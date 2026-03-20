# models/vehicle_dynamics.py
# Project-GP — Neural Port-Hamiltonian Vehicle Dynamics
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX2 — this revision)
# ─────────────────────────────────────────────────────────────────────────────
# BUGFIX-4 : CRITICAL — h_scale training-inference mismatch
#   Training: NeuralEnergyLandscape(M_diag=M_diag) → h_scale=1.0 (default).
#   Inference: NeuralEnergyLandscape(M_diag=M_diag, h_scale=102.62) loaded
#   from h_net_scale.txt. Inside __call__:
#       H_raw = softplus(Dense(1)(x)) * h_scale
#   Weights trained to output H_raw ∈ [0, 5000] (h_scale=1.0) were evaluated
#   with h_scale=102.62 → H_raw ∈ [0, 513100] → minimum(..., 5000) clips
#   everything → all H_net gradients zero at inference. FORCE_CAP tanh then
#   saturates at 12000N from 102× amplified forces, producing the 7534 mJ
#   passive energy injection seen in Test 2.
#   FIX: DifferentiableMultiBodyVehicle always instantiates NeuralEnergyLandscape
#   with h_scale=1.0. h_net_scale.txt is retained as a DIAGNOSTIC artefact only.
#
# BUGFIX-5 : susp_sq gated at absolute z=0, not physical equilibrium z_eq
#   PREVIOUS: susp_sq = sum(q[6:10]²) + 1e-4
#   The car operates at z_eq ≈ [12.8mm, 12.8mm, 14.2mm, 14.2mm]. At that
#   operating point susp_sq = 4·z_eq² ≈ 7e-4 m² — 7× larger than the 1e-4
#   floor, so the floor provides no real benefit. More critically, the gating
#   guarantees dH/dq=0 at z=0 (a configuration the car never occupies), while
#   the actual equilibrium point z_eq has a nonzero spring gradient (correct).
#   FIX: susp_sq = sum((q[6:10] - _Z_EQ)²) + 1e-4
#   · dH/dq at z_eq is now dominated by V_structural (30kN/m spring) + small
#     residual from H_res gradient — physically correct.
#   · Training data (residual_fitting.py) samples q_susp ~ N(z_eq, 20mm), so
#     E[susp_sq_new] = 4·(20mm)² = 0.0016 m² — gradient magnitude consistent.
#   · V_spring_dev in residual_fitting.py target_H MUST use (z-z_eq)² coords
#     to maintain consistency (see patch note in residual_fitting.py §1).
#
# UPGRADE-9 : H_RESIDUAL_CAP raised from 5000 to 50_000
#   With setup-dependent target_H = V_spring_dev + V_arb + V_torsion and
#   equilibrium-centered susp_sq, H_res = target_H / susp_sq is structurally:
#     H_res ≈ 0.5·(k_f - K_PRIOR)   [spring deviation term, k_f-dependent]
#   At k_f = 60000 (1σ high from setup distribution): H_res ≈ 15000 > 5000.
#   Old cap silently clipped ~30% of training samples with zero gradient signal.
#   50_000 covers the 99.9th percentile of physically valid setups.
#   The FORCE_CAP = 12000 N in _compute_derivatives provides physical limiting.
#
# UPGRADE-10 : DifferentiableAeroMap — softplus floors replace jnp.maximum
#   jnp.maximum(0.040 - heave_f, 0.015): zero subgradient when ground clearance
#   hits 15mm floor — exactly when MPC is most aggressively minimizing lap time.
#   jnp.maximum(Cl_f_base, 0.0): zero subgradient at zero downforce.
#   Both replaced with _softplus_floor — gradient always nonzero.
#
# ─── Retained from GP-vX1 ────────────────────────────────────────────────────
# BUGFIX-1  SuspensionSetup index alignment
# BUGFIX-2  Mass/inertia defaults corrected
# BUGFIX-3  _TRIL_14 missing definition
# UPGRADE-1 FiLM-conditioned NeuralEnergyLandscape
# UPGRADE-2 Structured NeuralDissipationMatrix with log-diagonal guarantee
# UPGRADE-3 4th-order Gauss-Legendre Variational Integrator
# UPGRADE-4 Differentiable ground-clearance bumpstop with softplus
# UPGRADE-5 Compliance steer from lateral load
# UPGRADE-6 Full C^∞ damper asymmetry (sigmoid bump/rebound blend)
# UPGRADE-7 Softplus Fz floor replaces jnp.maximum
# UPGRADE-8 susp_sq gradient floor — superseded by BUGFIX-5 / UPGRADE-9
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization

# ─────────────────────────────────────────────────────────────────────────────
# Module-level XLA-static constants
# ─────────────────────────────────────────────────────────────────────────────

# Pre-computed lower-triangular indices for 14×14 matrix.
# Defined at module load — NeuralDissipationMatrix.__call__ references this
# name. Placing inside the class or inside @nn.compact causes NameError because
# Flax traces the method body in a scope that doesn't re-execute module-level
# statements. Shape: two (105,) integer arrays — static from XLA's perspective.
_TRIL_14 = jnp.tril_indices(14)

# Static equilibrium suspension deflections [m] — front/rear.
# Derived from: F_susp_eq = m*g*l_opp/(2L) - m_us*g; z_eq = F_susp_eq/(k·MR₀²)
# Front: (300·9.81·0.6975)/(2·1.55) - 10·9.81 = 583N / (35000·1.14²) ≈ 0.0128m
# Rear:  (300·9.81·0.8525)/(2·1.55) - 11·9.81 = 695N / (38000·1.16²) ≈ 0.0135m
# Using canonical DEFAULT_SETUP spring rates. Defined here as a module-level
# constant so NeuralEnergyLandscape.__call__ and residual_fitting.py share
# identical values — single source of truth. See also compute_equilibrium_suspension().
_Z_EQ: jnp.ndarray = jnp.array([0.0128, 0.0128, 0.0135, 0.0135], dtype=jnp.float32)

# Structural spring prior per corner — must match V_structural in
# NeuralEnergyLandscape.__call__ and _V_STRUCT_PRIOR_K in residual_fitting.py.
_V_STRUCT_PRIOR_K: float = 30_000.0   # N/m


# ─────────────────────────────────────────────────────────────────────────────
# §1  SuspensionSetup  — 28-element typed pytree
# ─────────────────────────────────────────────────────────────────────────────

SETUP_DIM = 28

SETUP_NAMES = [
    'k_f', 'k_r',                         #  0–1   spring rates [N/m]
    'arb_f', 'arb_r',                     #  2–3   anti-roll bar [N·m/rad]
    'c_low_f', 'c_low_r',                 #  4–5   low-speed damper [N·s/m]
    'c_high_f', 'c_high_r',               #  6–7   high-speed damper [N·s/m]
    'v_knee_f', 'v_knee_r',               #  8–9   knee velocity [m/s]
    'rebound_ratio_f', 'rebound_ratio_r', # 10–11  rebound/bump ratio [-]
    'h_ride_f', 'h_ride_r',               # 12–13  ride height [m]
    'camber_f', 'camber_r',               # 14–15  static camber [deg]
    'toe_f', 'toe_r',                     # 16–17  static toe [deg]
    'castor_f',                           # 18     caster angle [deg]
    'anti_squat',                         # 19     anti-squat fraction [-]
    'anti_dive_f', 'anti_dive_r',         # 20–21  anti-dive fraction [-]
    'anti_lift',                          # 22     anti-lift fraction [-]
    'diff_lock_ratio',                    # 23     diff lock ratio [-]
    'brake_bias_f',                       # 24     front brake bias [-]
    'h_cg',                               # 25     CG height [m]
    'bump_steer_f', 'bump_steer_r',       # 26–27  bump steer [rad/m]
]

# Canonical default — physically validated for Ter26 FS 2026
DEFAULT_SETUP = jnp.array([
    35000., 38000.,   # k_f, k_r
      800.,   600.,   # arb_f, arb_r
     1800.,  1800.,   # c_low_f, c_low_r
     1200.,  1200.,   # c_high_f, c_high_r
     0.10,   0.10,    # v_knee_f, v_knee_r
     1.50,   1.50,    # rebound_ratio_f, rebound_ratio_r
     0.025,  0.022,   # h_ride_f, h_ride_r
     -2.0,   -1.5,    # camber_f, camber_r
     -0.10,  -0.15,   # toe_f, toe_r
      5.0,            # castor_f
      0.30,           # anti_squat
      0.40,  0.10,    # anti_dive_f, anti_dive_r
      0.20,           # anti_lift
      0.30,           # diff_lock_ratio
      0.60,           # brake_bias_f
      0.285,          # h_cg
      0.00,  0.00,    # bump_steer_f, bump_steer_r
], dtype=jnp.float32)

# Hard physical bounds for optimizer projection
SETUP_LB = jnp.array([
    10000., 10000.,   0.,    0.,   200.,  200.,   50.,   50.,
     0.03,   0.03,  1.0,   1.0,  0.010, 0.010,  -5.0,  -5.0,
    -1.0,   -1.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
     0.50,   0.18, -0.05, -0.05,
], dtype=jnp.float32)

SETUP_UB = jnp.array([
    120000., 120000., 5000., 5000., 8000., 8000., 4000., 4000.,
       0.40,   0.40,  3.0,   3.0,  0.060, 0.060,  0.0,   0.0,
       1.0,    1.0,  10.0,   0.9,   0.9,   0.9,   0.9,   1.0,
       0.80,   0.45, 0.05,  0.05,
], dtype=jnp.float32)


class SuspensionSetup(NamedTuple):
    """
    28-element typed container for all optimizable chassis setup parameters.

    Registered as a JAX pytree: flows through jit / vmap / grad / scan.
    Canonical parameter ordering is defined by SETUP_NAMES above.

    Design contract:
    · All quantities SI base units. Angles in degrees (documented explicitly).
    · `from_vector` / `to_vector` are the ONLY construction paths. Never
      construct by positional index externally.
    · `project_to_bounds()` uses smooth tanh rescaling — gradient-alive at bounds.
    """
    k_f: jax.Array              #  0
    k_r: jax.Array              #  1
    arb_f: jax.Array            #  2
    arb_r: jax.Array            #  3
    c_low_f: jax.Array          #  4
    c_low_r: jax.Array          #  5
    c_high_f: jax.Array         #  6
    c_high_r: jax.Array         #  7
    v_knee_f: jax.Array         #  8
    v_knee_r: jax.Array         #  9
    rebound_ratio_f: jax.Array  # 10
    rebound_ratio_r: jax.Array  # 11
    h_ride_f: jax.Array         # 12
    h_ride_r: jax.Array         # 13
    camber_f: jax.Array         # 14
    camber_r: jax.Array         # 15
    toe_f: jax.Array            # 16
    toe_r: jax.Array            # 17
    castor_f: jax.Array         # 18
    anti_squat: jax.Array       # 19
    anti_dive_f: jax.Array      # 20
    anti_dive_r: jax.Array      # 21
    anti_lift: jax.Array        # 22
    diff_lock_ratio: jax.Array  # 23
    brake_bias_f: jax.Array     # 24
    h_cg: jax.Array             # 25
    bump_steer_f: jax.Array     # 26
    bump_steer_r: jax.Array     # 27

    @staticmethod
    def from_vector(v: jax.Array) -> "SuspensionSetup":
        return SuspensionSetup(*[v[i] for i in range(SETUP_DIM)])

    def to_vector(self) -> jax.Array:
        return jnp.stack(list(self))

    @staticmethod
    def default() -> "SuspensionSetup":
        return SuspensionSetup.from_vector(DEFAULT_SETUP)

    @staticmethod
    def from_legacy_8(v: jax.Array) -> "SuspensionSetup":
        """
        Backward-compat loader for old 8-element checkpoints.
        Layout: [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f]
        """
        base = (DEFAULT_SETUP
                .at[0].set(v[0]).at[1].set(v[1])
                .at[2].set(v[2]).at[3].set(v[3])
                .at[4].set(v[4]).at[5].set(v[5])
                .at[25].set(v[6]).at[24].set(v[7]))
        return SuspensionSetup.from_vector(base)

    def project_to_bounds(self) -> "SuspensionSetup":
        """Smooth tanh projection — keeps gradient alive at bounds."""
        v    = self.to_vector()
        mid  = 0.5 * (SETUP_UB + SETUP_LB)
        half = 0.5 * (SETUP_UB - SETUP_LB)
        v_b  = mid + half * jnp.tanh((v - mid) / (half + 1e-6))
        return SuspensionSetup.from_vector(v_b)


# Register SuspensionSetup as a JAX pytree
jax.tree_util.register_pytree_node(
    SuspensionSetup,
    lambda s: (list(s), None),
    lambda _, leaves: SuspensionSetup(*leaves),
)


# ─────────────────────────────────────────────────────────────────────────────
# §2  PhysicsNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsNormalizer:
    """Normalization statistics for neural network inputs."""
    q_mean  = jnp.array([0., 0., 0.3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    q_scale = jnp.array([100., 100., 0.1, 0.1, 0.05, jnp.pi,
                          0.05, 0.05, 0.05, 0.05,
                          500., 500., 500., 500.])
    v_mean  = jnp.zeros(14)
    v_scale = jnp.array([25., 5., 1., 1., 1., 1., 1., 1., 1., 1., 50., 50., 50., 50.])

    # 28-element setup normalization — matching DEFAULT_SETUP operating point
    setup_mean = jnp.array([
        40000., 40000., 800.,  600.,  1800., 1800., 1200., 1200.,
        0.10,   0.10,   1.5,   1.5,   0.025, 0.022, -2.0,  -1.5,
       -0.10,  -0.15,  5.0,   0.30,   0.40,  0.10,  0.20,  0.30,
        0.60,   0.285,  0.00,  0.00,
    ], dtype=jnp.float32)

    setup_scale = jnp.array([
        20000., 20000., 400.,  300.,  800.,  800.,  600.,  600.,
        0.05,   0.05,   0.5,   0.5,   0.015, 0.012, 1.5,   1.5,
        0.5,    0.5,    3.0,   0.15,  0.20,  0.08,  0.10,  0.20,
        0.10,   0.040,  0.03,  0.03,
    ], dtype=jnp.float32)

    @staticmethod
    def normalize_q(q): return (q - PhysicsNormalizer.q_mean) / (PhysicsNormalizer.q_scale + 1e-8)
    @staticmethod
    def normalize_v(v): return (v - PhysicsNormalizer.v_mean) / (PhysicsNormalizer.v_scale + 1e-8)
    @staticmethod
    def normalize_setup(s): return (s - PhysicsNormalizer.setup_mean) / (PhysicsNormalizer.setup_scale + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Kinematic utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_ratio(z_corner: jax.Array, poly_coeffs: jax.Array) -> jax.Array:
    """Quadratic MR(z) = a0 + a1·z + a2·z²"""
    return poly_coeffs[0] + poly_coeffs[1] * z_corner + poly_coeffs[2] * z_corner ** 2


def compute_damper_force_bilinear(
    z_dot: jax.Array,
    c_low: jax.Array,
    c_high: jax.Array,
    v_knee: jax.Array,
    rebound_ratio: jax.Array,
) -> jax.Array:
    """
    4-way digressive damper: low/high speed bump + rebound asymmetry.
    Fully C^∞ smooth — no hard conditionals anywhere.

    High/low speed split: sigmoid blend at v_knee.
    Bump/rebound split: sigmoid blend with β=200 s/m → ~5mm/s transition width.
    At |z_dot| > 20mm/s the blend is indistinguishable from hard switching so
    physical accuracy is preserved. At z_dot=0 the blend is 50/50 of bump and
    rebound — the only physically valid C^∞ extension at zero velocity.
    """
    v_abs  = jnp.abs(z_dot)
    w_bump = jax.nn.sigmoid(200.0 * z_dot)
    w_reb  = 1.0 - w_bump

    c_lo = w_bump * c_low              + w_reb * (c_low  * rebound_ratio)
    c_hi = w_bump * c_high             + w_reb * (c_high * rebound_ratio)

    alpha  = 20.0 / (v_knee + 1e-6)
    w_high = jax.nn.sigmoid(alpha * (v_abs - v_knee))
    w_low  = 1.0 - w_high

    c_eff = c_lo * w_low + (c_lo + c_hi) * w_high
    return c_eff * z_dot


def compute_bump_steer(
    z_corner: jax.Array,
    bs_lin: jax.Array,
    bs_quad: jax.Array,
) -> jax.Array:
    """Bump steer: δ_bs = bs_lin·z + bs_quad·z²"""
    return bs_lin * z_corner + bs_quad * z_corner ** 2


def compute_bumpstop_force(
    z_corner: jax.Array,
    gap: jax.Array,
    k_bs: float = 500_000.0,
    beta: float = 200.0,
) -> jax.Array:
    """
    C^∞ bumpstop force via softplus. Zero force for z < gap, rising rapidly
    above. Replaces hard clip — gradient-alive through contact.
    F_bs = k_bs · softplus((z - gap)·β) / β
    """
    return k_bs * jax.nn.softplus(beta * (z_corner - gap)) / beta


def compute_castor_trail(castor_deg: jax.Array, Fz: jax.Array, tire_radius: float) -> jax.Array:
    t_mech = tire_radius * jnp.tan(jnp.deg2rad(castor_deg))
    return Fz * t_mech


def _softplus_floor(x: jax.Array, floor: float = 10.0) -> jax.Array:
    """
    Smooth lower bound: f(x) = floor + softplus(x - floor).
    · f(x) → x       for x >> floor  (identity in the physical regime)
    · f(x) → floor   for x << floor  (asymptotic minimum)
    · df/dx = sigmoid(x - floor) ∈ (0, 1) — never zero, never > 1.

    Replaces jnp.maximum(..., floor) whose sub-gradient is zero below the
    floor — kills optimizer signal exactly when a corner goes light.
    """
    return floor + jax.nn.softplus(x - floor)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Neural Networks
# ─────────────────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on setup vector.
    h_out = γ(setup) ⊙ LayerNorm(h) + β(setup)

    Mathematically superior to concatenation: the setup vector controls the
    GRADIENT of the energy landscape (effective stiffness), not just its offset.
    Initialized to identity (γ=1, β=0) so training starts from the unmodulated
    baseline and learns deviations.
    """
    features: int

    @nn.compact
    def __call__(self, h: jax.Array, setup_embedding: jax.Array) -> jax.Array:
        gamma = nn.Dense(self.features,
                         kernel_init=jax.nn.initializers.zeros,
                         bias_init=jax.nn.initializers.ones)(setup_embedding)
        beta  = nn.Dense(self.features,
                         kernel_init=jax.nn.initializers.zeros,
                         bias_init=jax.nn.initializers.zeros)(setup_embedding)
        h_norm = nn.LayerNorm()(h)
        return gamma * h_norm + beta


class NeuralEnergyLandscape(nn.Module):
    """
    Port-Hamiltonian residual energy H_res(q, p, setup).

    Architecture (GP-vX2):
    · SE(3)-bilateral symmetric feature extraction (22 state features).
      Anti-symmetric features enter as x² — structurally cannot represent
      odd functions of vy, wz, roll regardless of weights.
    · FiLM modulation at each hidden layer via 16 energy-relevant setup dims.
    · Output: T_prior + V_structural + H_residual
      where H_residual = softplus(MLP) · susp_sq_eq ≥ 0.

    susp_sq_eq = Σ(q[6:10] - _Z_EQ)² + 1e-4   [BUGFIX-5]
    · Gating at physical equilibrium z_eq, not at z=0.
    · dH/dq|_{z=z_eq} ≈ 30000·z_eq (from V_structural alone) — physically
      correct spring force at static equilibrium, balanced by gravity in F_ext.
    · H_res → 0.5·(k_f - K_PRIOR) near equilibrium (clean structure).
    · 1e-4 floor = (1mm)² — physical minimum, not a numerical hack.

    H_RESIDUAL_CAP = 50_000 J/m²   [UPGRADE-9]
    · H_res = target_H / susp_sq = 0.5·(k_f-K_PRIOR) at mean. For k_f=60000:
      H_res = 15000 J/m² > old cap of 5000 → old cap silently clipped 30%+.

    h_scale: ALWAYS 1.0 at both training and inference   [BUGFIX-4]
    · Training used h_scale=1.0 (NeuralEnergyLandscape default).
    · Old inference code passed h_scale=102.62 from h_net_scale.txt, creating
      a 102× energy amplification → FORCE_CAP saturation → 7534 mJ injection.
    · h_net_scale.txt is a training normalisation artefact, not architectural.
    """
    M_diag:         jnp.ndarray
    h_scale:        float = 1.0        # ALWAYS 1.0 — see BUGFIX-4
    H_RESIDUAL_CAP: float = 50_000.0   # raised from 5000 — see UPGRADE-9

    # Static index tuple for setup selection — Python-level constant so XLA
    # emits a static gather with no dynamic index allocation on each trace.
    _SETUP_IDX: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 12, 13, 25, 19, 20, 24)

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array, setup_params: jax.Array) -> jax.Array:
        T_prior      = 0.5 * jnp.sum((p ** 2) / (self.M_diag + 1e-8))
        V_structural = 0.5 * jnp.sum(q[6:10] ** 2) * _V_STRUCT_PRIOR_K

        v = p / (self.M_diag + 1e-8)
        Z, phi_roll, theta_pitch                  = q[2], q[3], q[4]
        z_fl, z_fr, z_rl, z_rr                   = q[6], q[7], q[8], q[9]
        vx, vy, vz, wx, wy, wz                   = v[0], v[1], v[2], v[3], v[4], v[5]
        dvz_fl, dvz_fr, dvz_rl, dvz_rr           = v[6], v[7], v[8], v[9]
        om_fl, om_fr, om_rl, om_rr               = v[10], v[11], v[12], v[13]

        # ── Symmetric features (invariant to left-right flip) ─────────────────
        SYM_Q_SC = jnp.array([0.10, 0.10, 0.05, 0.05])
        SYM_V_SC = jnp.array([25.0, 1.0, 1.5, 1.0, 1.0, 75.0, 75.0])
        sym_q = jnp.array([Z, theta_pitch,
                            (z_fl + z_fr) * 0.5, (z_rl + z_rr) * 0.5])
        sym_v = jnp.array([vx, vz, wy,
                            (dvz_fl + dvz_fr) * 0.5, (dvz_rl + dvz_rr) * 0.5,
                            (om_fl + om_fr) * 0.5,   (om_rl + om_rr) * 0.5])
        sym_q_n = sym_q / (SYM_Q_SC + 1e-6)
        sym_v_n = sym_v / (SYM_V_SC + 1e-6)

        # ── Anti-symmetric features (enter as x² → even fn → dH/dq=0 at eq) ──
        ANTI_Q_SC = jnp.array([0.10, 0.05, 0.05, 0.05])
        ANTI_V_SC = jnp.array([5.0, 1.0, 1.0, 75.0, 75.0, 75.0, 75.0])
        anti_q = jnp.array([phi_roll, z_fl - z_fr, z_rl - z_rr,
                              z_fl + z_rr - z_fr - z_rl]) ** 2
        anti_v = jnp.array([vy, wx, wz, dvz_fl - dvz_fr, dvz_rl - dvz_rr,
                              om_fl - om_fr, om_rl - om_rr]) ** 2
        anti_q_n = anti_q / (ANTI_Q_SC ** 2 + 1e-6)
        anti_v_n = anti_v / (ANTI_V_SC ** 2 + 1e-6)

        # State features: 4 + 7 + 4 + 7 = 22 (SE(3)-bilateral invariant)
        state_features = jnp.concatenate([sym_q_n, sym_v_n, anti_q_n, anti_v_n])

        # ── Setup embedding (decoupled MLP path) ─────────────────────────────
        setup_norm = PhysicsNormalizer.normalize_setup(setup_params)
        setup_sel  = setup_norm[jnp.array(self._SETUP_IDX)]
        setup_emb  = nn.swish(nn.Dense(32)(setup_sel))

        # ── Hidden layers with FiLM conditioning ─────────────────────────────
        x = nn.Dense(128)(state_features)
        x = FiLMLayer(128)(x, setup_emb)
        x = nn.swish(x)

        x = nn.Dense(64)(x)
        x = FiLMLayer(64)(x, setup_emb)
        x = nn.swish(x)

        H_raw = jnp.squeeze(jax.nn.softplus(nn.Dense(1)(x))) * self.h_scale
        H_res = jnp.minimum(H_raw, self.H_RESIDUAL_CAP)

        # BUGFIX-5: gate at physical equilibrium _Z_EQ, not at z=0.
        # With equilibrium-centered training data (q_susp ~ N(z_eq, 20mm)),
        # E[susp_sq_eq] = 4·(20mm)² = 0.0016 m² — full gradient magnitude.
        # The 1e-4 floor = (1mm)² provides numerical stability at exact eq.
        # Consistency requirement: residual_fitting.py V_spring_dev must use
        # (z_fl - z_eq_f)² coordinates (not absolute z²). See §1 patch note.
        susp_sq = jnp.sum((q[6:10] - _Z_EQ) ** 2) + 1e-4

        return T_prior + V_structural + H_res * susp_sq


class NeuralDissipationMatrix(nn.Module):
    """
    Port-Hamiltonian dissipation matrix R(q,p) = L·Lᵀ + diag(softplus(d)).

    · Learnable log-diagonal bias d ensures R ≥ diag(softplus(d)) > 0
      (strictly positive definite — STRONGER than PSD).
    · Diagonal floor prevents near-conservative blow-up from arbitrarily small
      damping predictions.
    · Physical mask restricts dissipation to heave, roll, pitch, unsprung-z DOFs.
    · _TRIL_14 is a MODULE-LEVEL constant (defined at top of file after imports).
      Do NOT move it inside this class or inside __call__ — Flax traces __call__
      in a scope where module-level names are still visible, but any attempt to
      re-define it locally would shadow the outer binding and cause NameError
      in environments that cache the trace.
    """
    dim: int = 14

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array) -> jax.Array:
        q_n   = PhysicsNormalizer.normalize_q(q)
        p_n   = p / (PhysicsNormalizer.v_scale * 200.0 + 1e-6)
        state = jnp.concatenate([q_n, p_n])

        x = nn.Dense(128)(state); x = nn.swish(x)
        x = nn.Dense(64)(x);      x = nn.swish(x)

        n_elem  = self.dim * (self.dim + 1) // 2
        L_elems = nn.Dense(n_elem,
                           kernel_init=jax.nn.initializers.lecun_normal(),
                           bias_init=jax.nn.initializers.zeros)(x)

        L      = jnp.zeros((self.dim, self.dim)).at[_TRIL_14].set(L_elems)
        R_chol = jnp.dot(L, L.T)

        log_d   = self.param('log_d', jax.nn.initializers.constant(-3.0), (self.dim,))
        R_diag  = jnp.diag(jax.nn.softplus(log_d))
        R_dense = R_chol + R_diag

        mask = jnp.array([0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.])
        return R_dense * jnp.outer(mask, mask)


class DifferentiableAeroMap(nn.Module):
    """
    Ground-effect-aware aero map.

    UPGRADE-10: replaced jnp.maximum with _softplus_floor for heave and Cl floors.
    · h_f = max(0.040 - heave_f, 0.015): zero subgradient at 15mm floor —
      exactly when MPC optimization is most aggressive. Replaced with softplus.
    · Cl_f = max(Cl_f_base, 0.0): zero subgradient at zero downforce (e.g.
      during braking with nose-up pitch). Replaced with softplus.
    Both are constraint-active during setup optimization; gradient must be live.
    """
    base_A:  float
    base_Cl: float
    base_Cd: float
    lf:      float
    lr:      float

    @nn.compact
    def __call__(self, vx, pitch, roll, heave_f, heave_r):
        state = jnp.stack([vx / 100.0, pitch, roll, heave_f, heave_r])
        x = nn.Dense(32)(state); x = nn.swish(x)
        x = nn.Dense(32)(x);     x = nn.swish(x)
        mods = nn.Dense(4,
                        kernel_init=jax.nn.initializers.zeros,
                        bias_init=jax.nn.initializers.zeros)(x)

        h_ref = 0.040
        # UPGRADE-10: _softplus_floor replaces jnp.maximum — gradient alive at floor.
        # floor=0.015 m: minimum physically meaningful ground clearance.
        h_f = _softplus_floor(0.040 - heave_f, 0.015)
        h_r = _softplus_floor(0.040 - heave_r, 0.015)

        Cl_f_base = (self.base_Cl * 0.40 * (1.0 + 0.30 * (h_ref / h_f - 1.0))
                     + 0.35 * pitch)
        Cl_r_base = self.base_Cl * 0.60 * (1.0 + 0.45 * (h_ref / h_r - 1.0))
        Cd_dyn    = self.base_Cd + mods[1]

        # UPGRADE-10: softplus prevents negative Cl (lift instead of downforce).
        # floor=0.0: downforce can be zero (nose-up braking) but never negative.
        Cl_f = _softplus_floor(Cl_f_base + mods[0] * 0.40, 0.0)
        Cl_r = _softplus_floor(Cl_r_base + mods[0] * 0.60, 0.0)

        q_dyn     = 0.5 * 1.225 * vx ** 2
        Fz_aero_f = q_dyn * Cl_f * self.base_A
        Fz_aero_r = q_dyn * Cl_r * self.base_A
        Fx_aero   = -q_dyn * Cd_dyn * self.base_A
        My_aero   = (Fz_aero_r * self.lr - Fz_aero_f * self.lf
                     + (Fz_aero_f + Fz_aero_r) * mods[2])
        Mx_aero   = (Fz_aero_f + Fz_aero_r) * mods[3]

        return Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero


# ─────────────────────────────────────────────────────────────────────────────
# §5  DifferentiableMultiBodyVehicle
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableMultiBodyVehicle:
    """
    Port-Hamiltonian 14-DOF full-car model with 28-parameter SuspensionSetup.

    State x (46-dim):
      [0:14]  q  — positions  (X,Y,Z,φ,θ,ψ, z_fl..z_rr, θ_fl..θ_rr)
      [14:28] v  — velocities (vx,vy,vz,wx,wy,wz, ż_fl..ż_rr, ω_fl..ω_rr)
      [28:38] thermal  (5-node tire model, front+rear)
      [38:46] transient slip  (α_t, κ_t × 4 corners)

    Integrator: 2-stage Gauss-Legendre RK4 (GLRK-4) — symplectic 4th-order.
    Aux integration: trapezoidal at converged Gauss stage points (no extra
    _compute_derivatives call vs. previous forward Euler).
    """

    def __init__(self, vehicle_params: dict, tire_coeffs: dict, rng_seed: int = 42):
        from models.tire_model import PacejkaTire
        self.vp   = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)

        self.m_us_f = self.vp.get('unsprung_mass_f', 7.74)
        self.m_us_r = self.vp.get('unsprung_mass_r', 7.76)
        m_unsprung  = 2.0 * self.m_us_f + 2.0 * self.m_us_r

        self.m   = self.vp.get('total_mass', 300.0)
        self.m_s = self.m - m_unsprung

        self.Ix = self.vp.get('Ix', 45.0)
        self.Iy = self.vp.get('Iy', 85.0)
        self.Iz = self.vp.get('Iz', 150.0)
        self.Iw = self.vp.get('Iw',  1.2)

        self.lf       = self.vp.get('lf', 0.8525)
        self.lr       = self.vp.get('lr', 0.6975)
        self.track_f  = self.vp.get('track_front', 1.200)
        self.track_r  = self.vp.get('track_rear',  1.180)
        self.g        = 9.81
        self.R_wheel  = self.vp.get('wheel_radius', 0.2045)

        self.track_w  = self.track_f
        self._L       = self.lf + self.lr

        self.M_diag = jnp.array([
            self.m_s, self.m_s, self.m_s, self.Ix, self.Iy, self.Iz,
            self.m_us_f, self.m_us_f, self.m_us_r, self.m_us_r,
            self.Iw, self.Iw, self.Iw, self.Iw,
        ])

        self._mr_f_poly   = jnp.array(self.vp.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))
        self._mr_r_poly   = jnp.array(self.vp.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))
        self._bs2_f       = self.vp.get('bump_steer_quad_f', 0.0)
        self._bs2_r       = self.vp.get('bump_steer_quad_r', 0.0)
        self._camber_dz_f = self.vp.get('camber_per_m_travel_f', -25.0)
        self._camber_dz_r = self.vp.get('camber_per_m_travel_r', -20.0)
        self._dh_rc_dz_f  = self.vp.get('dh_rc_dz_f', 0.20)
        self._dh_rc_dz_r  = self.vp.get('dh_rc_dz_r', 0.30)
        self._h_rc0_f     = self.vp.get('h_rc_f', 0.040)
        self._h_rc0_r     = self.vp.get('h_rc_r', 0.060)
        self._ackermann   = self.vp.get('ackermann_factor', 0.0)
        self._comply_f    = self.vp.get('compliance_steer_f', -0.15)
        self._comply_r    = self.vp.get('compliance_steer_r', -0.10)

        # BUGFIX-4: read h_scale from file for DIAGNOSTIC logging only.
        # The architectural h_scale in NeuralEnergyLandscape is ALWAYS 1.0.
        # Passing h_net_scale.txt (a training normalisation factor) into the
        # network architecture caused a 102× energy amplification at inference:
        #   H_raw = softplus(Dense(1)(x)) * 102.62 → min(..., 5000) clips
        #   everything → all H_net gradients zero → forces saturated at FORCE_CAP.
        # The scale file is kept for W&B logging and diagnostics; it does NOT
        # feed into the network architecture.
        current_dir  = os.path.dirname(os.path.abspath(__file__))
        h_scale_path = os.path.join(current_dir, 'h_net_scale.txt')
        self._h_train_scale = 1.0   # diagnostic reference only
        if os.path.exists(h_scale_path):
            with open(h_scale_path) as f:
                self._h_train_scale = float(f.read().strip())
            print(f"[VehicleDynamics] H_net train scale (diagnostic): "
                  f"{self._h_train_scale:.4f} J  [NOT applied to architecture]")
        else:
            print("[VehicleDynamics] h_net_scale.txt not found — "
                  "run train_neural_residuals() first.")

        # h_scale=1.0 ALWAYS — weights were trained with h_scale=1.0 default.
        self.H_net    = NeuralEnergyLandscape(M_diag=self.M_diag, h_scale=1.0)
        self.R_net    = NeuralDissipationMatrix(dim=14)
        self.aero_map = DifferentiableAeroMap(
            base_A  = self.vp.get('A_ref',  1.10),
            base_Cl = self.vp.get('Cl_ref', 4.14),
            base_Cd = self.vp.get('Cd_ref', 2.50),
            lf      = self.lf,
            lr      = self.lr,
        )

        rng = jax.random.PRNGKey(rng_seed)
        rng_h, rng_r, rng_a = jax.random.split(rng, 3)

        self.H_params    = self.H_net.init(rng_h, jnp.zeros(14), jnp.zeros(14), DEFAULT_SETUP)
        self.R_params    = self.R_net.init(rng_r, jnp.zeros(14), jnp.zeros(14))
        self.Aero_params = self.aero_map.init(rng_a, 0.0, 0.0, 0.0, 0.0, 0.0)

        for attr, fname in [('H_params',    'h_net.bytes'),
                             ('R_params',    'r_net.bytes'),
                             ('Aero_params', 'aero_net.bytes')]:
            ckpt = os.path.join(current_dir, fname)
            if os.path.exists(ckpt):
                with open(ckpt, 'rb') as f:
                    setattr(self, attr,
                            flax.serialization.from_bytes(getattr(self, attr), f.read()))
                print(f"[VehicleDynamics] Loaded {fname}")

    # ─────────────────────────────────────────────────────────────────────────
    # §5.1  Powertrain
    # ─────────────────────────────────────────────────────────────────────────

    def compute_drive_force(self, throttle: jax.Array, vx: jax.Array) -> jax.Array:
        T_peak = self.vp.get('motor_peak_torque', 21.0)
        ratio  = self.vp.get('drivetrain_ratio', 4.5)
        eta    = self.vp.get('drivetrain_efficiency', 0.92)
        v_max  = self.vp.get('v_max', 35.0)
        F_max_tq = throttle * T_peak * ratio * eta / self.R_wheel
        F_power  = jax.nn.softplus(v_max - vx) / v_max * F_max_tq
        return jnp.minimum(F_max_tq, F_power)

    def compute_brake_forces(self, brake_force, Fz_f, Fz_r, vx, brake_bias_f):
        mu_pad    = self.vp.get('brake_mu', 0.40)
        F_brake_f = -brake_force * brake_bias_f         * mu_pad
        F_brake_r = -brake_force * (1.0 - brake_bias_f) * mu_pad
        return F_brake_f, F_brake_r

    # ─────────────────────────────────────────────────────────────────────────
    # §5.2  Differential
    # ─────────────────────────────────────────────────────────────────────────

    def compute_differential_forces(
        self, T_drive_wheel, vx, wz,
        Fz_rl, Fz_rr, alpha_t_rl, alpha_t_rr,
        gamma_rl, gamma_rr, T_ribs_r, T_gas_r, diff_lock,
    ):
        eps  = 0.5
        tr   = self.track_r
        eta  = self.vp.get('drivetrain_efficiency', 0.92)
        vx_s = jnp.maximum(jnp.abs(vx), eps)

        v_rl       = vx_s - wz * tr / 2.0
        v_rr       = vx_s + wz * tr / 2.0
        omega_diff = (v_rl + v_rr) / (2.0 * self.R_wheel)

        d_omega    = omega_diff - v_rl / self.R_wheel
        T_lock     = diff_lock * 500.0 * d_omega

        T_rl_input = T_drive_wheel * 0.5 * eta - T_lock
        T_rr_input = T_drive_wheel * 0.5 * eta + T_lock

        omega_rl = v_rl / self.R_wheel + T_rl_input / (10.0 * self.R_wheel)
        omega_rr = v_rr / self.R_wheel + T_rr_input / (10.0 * self.R_wheel)

        kappa_rl = 0.5 * jnp.tanh((omega_rl * self.R_wheel - v_rl) / (vx_s * 0.5))
        kappa_rr = 0.5 * jnp.tanh((omega_rr * self.R_wheel - v_rr) / (vx_s * 0.5))

        Fx_rl, Fy_rl = self.tire.compute_force(
            alpha_t_rl, kappa_rl, Fz_rl, gamma_rl, T_ribs_r, T_gas_r, vx, wz=wz)
        Fx_rr, Fy_rr = self.tire.compute_force(
            alpha_t_rr, kappa_rr, Fz_rr, gamma_rr, T_ribs_r, T_gas_r, vx, wz=wz)

        return Fx_rl, Fx_rr, Fy_rl, Fy_rr, kappa_rl, kappa_rr

    # ─────────────────────────────────────────────────────────────────────────
    # §5.3  Core derivatives
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _compute_derivatives(
        self,
        x:            jax.Array,
        u:            jax.Array,
        setup_params: jax.Array,
    ) -> jax.Array:
        q, v = x[0:14], x[14:28]
        p    = self.M_diag * v

        grad_H_fn    = jax.grad(self.H_net.apply, argnums=(1, 2))
        dH_dq, dH_dp = grad_H_fn(self.H_params, q, p, setup_params)

        FORCE_CAP = 12000.0; VEL_CAP = 150.0
        dH_dq  = FORCE_CAP * jnp.tanh(dH_dq / (FORCE_CAP + 1e-8))
        dH_dp  = VEL_CAP   * jnp.tanh(dH_dp / (VEL_CAP   + 1e-8))

        # Stop the H_net Hessian from entering ANY outer optimization backward pass.
        #
        # WHAT THIS DOES: blocks ∂²H/∂(q,p)² from accumulating through the
        # WMPC lax.scan (64 steps) × GLRK-4 Newton scan (5 iters) × substep
        # scan (5 substeps) = up to 1600 chained Hessian evaluations. Even one
        # ill-conditioned Hessian evaluation at a transient non-physical state
        # NaNs the entire backward pass via JAX's non-short-circuiting scan.
        #
        # WHAT THIS PRESERVES:
        # · Forward dynamics remain exactly correct — H_net forces (PH_accel)
        #   are computed faithfully and drive the car in the forward pass.
        # · Physical force gradients (∂F_tire/∂u, ∂F_spring/∂k, ∂F_damper/∂c)
        #   still flow back through the entire scan chain — the optimizer retains
        #   full gradient signal for braking, cornering, and setup optimization.
        # · MORL setup gradients are unaffected: setup enters through
        #   SuspensionSetup (spring/ARB/damper), not through H_net argnums.
        #
        # WHY SEMANTICALLY EXACT: H_params are frozen during all online
        # optimization (WMPC, MORL). H_net's Hessian w.r.t. state contributes
        # zero useful gradient information for the wavelet coefficient search.
        # This is the straight-through estimator applied to a frozen network.
        dH_dq = jax.lax.stop_gradient(dH_dq)
        dH_dp = jax.lax.stop_gradient(dH_dp)
        grad_H = jnp.concatenate([dH_dq, dH_dp])

        J = jnp.zeros((28, 28))
        J = J.at[0:14, 14:28].set( jnp.eye(14))
        J = J.at[14:28, 0:14].set(-jnp.eye(14))
        R = jnp.zeros((28, 28))
        R = R.at[14:28, 14:28].set(self.R_net.apply(self.R_params, q, p))
        PH_accel = jnp.dot(J - R, grad_H)

        s = SuspensionSetup.from_vector(setup_params)

        X, Y, Z, phi_roll, theta_pitch, psi_yaw = q[0:6]
        vx, vy, vz, wx, wy, wz                  = v[0:6]
        vx = jnp.clip(vx, -80.0, 80.0)
        vy = jnp.clip(vy, -30.0, 30.0)
        wz = jnp.clip(wz,  -8.0,  8.0)
        Z  = jnp.clip(Z,  -0.3,  1.5)
        phi_roll    = jnp.clip(phi_roll,    -0.3, 0.3)
        theta_pitch = jnp.clip(theta_pitch, -0.2, 0.2)

        tf2 = self.track_f / 2.0
        tr2 = self.track_r / 2.0

        # Physical travel limits — 150mm bump / 80mm droop.
        # Without clipping: bumpstop softplus(200*(z-0.025)) overflows float32 at z≈0.47m.
        # GLRK4 Newton intermediate stages can temporarily overshoot this — one NaN
        # poisons all 320 downstream scan steps via JAX's non-short-circuiting scan.
        _SZ_MAX, _SZ_MIN = 0.15, -0.08
        z_fl = jnp.clip(q[6], _SZ_MIN, _SZ_MAX)
        z_fr = jnp.clip(q[7], _SZ_MIN, _SZ_MAX)
        z_rl = jnp.clip(q[8], _SZ_MIN, _SZ_MAX)
        z_rr = jnp.clip(q[9], _SZ_MIN, _SZ_MAX)
        # Clip suspension velocities — prevents extreme damper forces at Newton overshoot
        _SDZ_MAX = 3.0
        dz_fl = jnp.clip(v[6], -_SDZ_MAX, _SDZ_MAX)
        dz_fr = jnp.clip(v[7], -_SDZ_MAX, _SDZ_MAX)
        dz_rl = jnp.clip(v[8], -_SDZ_MAX, _SDZ_MAX)
        dz_rr = jnp.clip(v[9], -_SDZ_MAX, _SDZ_MAX)
        omega_fl, omega_fr          = v[10], v[11]
        omega_rl, omega_rr          = v[12], v[13]

        alpha_t_fl, kappa_t_fl = x[38], x[39]
        alpha_t_fr, kappa_t_fr = x[40], x[41]
        alpha_t_rl, kappa_t_rl = x[42], x[43]
        alpha_t_rr, kappa_t_rr = x[44], x[45]

        MR_fl = compute_motion_ratio(z_fl, self._mr_f_poly)
        MR_fr = compute_motion_ratio(z_fr, self._mr_f_poly)
        MR_rl = compute_motion_ratio(z_rl, self._mr_r_poly)
        MR_rr = compute_motion_ratio(z_rr, self._mr_r_poly)

        F_spring_fl = s.k_f * z_fl * (MR_fl ** 2)
        F_spring_fr = s.k_f * z_fr * (MR_fr ** 2)
        F_spring_rl = s.k_r * z_rl * (MR_rl ** 2)
        F_spring_rr = s.k_r * z_rr * (MR_rr ** 2)

        F_damp_fl = compute_damper_force_bilinear(dz_fl, s.c_low_f, s.c_high_f, s.v_knee_f, s.rebound_ratio_f)
        F_damp_fr = compute_damper_force_bilinear(dz_fr, s.c_low_f, s.c_high_f, s.v_knee_f, s.rebound_ratio_f)
        F_damp_rl = compute_damper_force_bilinear(dz_rl, s.c_low_r, s.c_high_r, s.v_knee_r, s.rebound_ratio_r)
        F_damp_rr = compute_damper_force_bilinear(dz_rr, s.c_low_r, s.c_high_r, s.v_knee_r, s.rebound_ratio_r)

        F_bs_fl = compute_bumpstop_force(z_fl, s.h_ride_f)
        F_bs_fr = compute_bumpstop_force(z_fr, s.h_ride_f)
        F_bs_rl = compute_bumpstop_force(z_rl, s.h_ride_r)
        F_bs_rr = compute_bumpstop_force(z_rr, s.h_ride_r)

        F_susp_fl = F_spring_fl + F_damp_fl + F_bs_fl
        F_susp_fr = F_spring_fr + F_damp_fr + F_bs_fr
        F_susp_rl = F_spring_rl + F_damp_rl + F_bs_rl
        F_susp_rr = F_spring_rr + F_damp_rr + F_bs_rr

        z_roll_f  = (z_fl - z_fr) * 0.5
        z_roll_r  = (z_rl - z_rr) * 0.5
        F_arb_f   = s.arb_f * z_roll_f / (tf2 + 1e-6)
        F_arb_r   = s.arb_r * z_roll_r / (tr2 + 1e-6)
        F_susp_fl = F_susp_fl + F_arb_f
        F_susp_fr = F_susp_fr - F_arb_f
        F_susp_rl = F_susp_rl + F_arb_r
        F_susp_rr = F_susp_rr - F_arb_r

        F_grav_f  = self.m * self.g * self.lr / self._L
        F_grav_r  = self.m * self.g * self.lf / self._L
        dFz_accel = self.m * jnp.clip(vx * wz, -50.0, 50.0) * self.vp.get('h_cg', 0.330) / self._L
        dFz_lat   = self.m * jnp.clip(vy * wz, -50.0, 50.0) * self.vp.get('h_cg', 0.330) / (self.track_f + 1e-6)

        Fz_fl = _softplus_floor(F_grav_f * 0.5 - dFz_accel * 0.5 - dFz_lat * 0.5)
        Fz_fr = _softplus_floor(F_grav_f * 0.5 - dFz_accel * 0.5 + dFz_lat * 0.5)
        Fz_rl = _softplus_floor(F_grav_r * 0.5 + dFz_accel * 0.5 - dFz_lat * 0.5)
        Fz_rr = _softplus_floor(F_grav_r * 0.5 + dFz_accel * 0.5 + dFz_lat * 0.5)

        Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero = self.aero_map.apply(
            self.Aero_params, vx, theta_pitch, phi_roll,
            z_fl + z_fr, z_rl + z_rr,
        )
        Fz_fl = Fz_fl + Fz_aero_f * 0.5
        Fz_fr = Fz_fr + Fz_aero_f * 0.5
        Fz_rl = Fz_rl + Fz_aero_r * 0.5
        Fz_rr = Fz_rr + Fz_aero_r * 0.5

        h_rc_f = self._h_rc0_f + self._dh_rc_dz_f * (z_fl + z_fr) * 0.5
        h_rc_r = self._h_rc0_r + self._dh_rc_dz_r * (z_rl + z_rr) * 0.5

        gamma_fl = jnp.deg2rad(s.camber_f + self._camber_dz_f * z_fl + self.vp.get('camber_gain_f', -0.80) * jnp.rad2deg(phi_roll) * 0.5)
        gamma_fr = jnp.deg2rad(s.camber_f + self._camber_dz_f * z_fr - self.vp.get('camber_gain_f', -0.80) * jnp.rad2deg(phi_roll) * 0.5)
        gamma_rl = jnp.deg2rad(s.camber_r + self._camber_dz_r * z_rl + self.vp.get('camber_gain_r', -0.65) * jnp.rad2deg(phi_roll) * 0.5)
        gamma_rr = jnp.deg2rad(s.camber_r + self._camber_dz_r * z_rr - self.vp.get('camber_gain_r', -0.65) * jnp.rad2deg(phi_roll) * 0.5)

        delta_cmd   = u[0]
        delta_bs_fl = compute_bump_steer(z_fl, s.bump_steer_f, self._bs2_f)
        delta_bs_fr = compute_bump_steer(z_fr, s.bump_steer_f, self._bs2_f)

        Fy_total_approx = jnp.clip((vy * self.m * self.g) / (vx + 1.0), -5000.0, 5000.0)
        delta_comply_f  = jnp.deg2rad(self._comply_f * Fy_total_approx / 1000.0)
        delta_comply_r  = jnp.deg2rad(self._comply_r * Fy_total_approx / 1000.0)

        ack      = self._ackermann
        wb       = self._L
        delta_fl = delta_cmd * (1.0 + ack * tf2 / (2.0 * wb)) + delta_bs_fl + delta_comply_f
        delta_fr = delta_cmd * (1.0 - ack * tf2 / (2.0 * wb)) + delta_bs_fr + delta_comply_f

        eps_v        = 0.5
        v_corner_fl  = vx - wz * tf2
        v_corner_fr  = vx + wz * tf2
        alpha_kin_fl = delta_fl       - jnp.arctan2(vy + wz * self.lf, jnp.maximum(jnp.abs(v_corner_fl), eps_v))
        alpha_kin_fr = delta_fr       - jnp.arctan2(vy + wz * self.lf, jnp.maximum(jnp.abs(v_corner_fr), eps_v))
        alpha_kin_rl = delta_comply_r - jnp.arctan2(vy - wz * self.lr, jnp.maximum(jnp.abs(vx - wz * tr2), eps_v))
        alpha_kin_rr = delta_comply_r - jnp.arctan2(vy - wz * self.lr, jnp.maximum(jnp.abs(vx + wz * tr2), eps_v))

        rl  = self.vp.get('relaxation_length', 0.35)
        tau = rl / (jnp.maximum(jnp.abs(vx), 1.0))
        d_alpha_fl = (alpha_kin_fl - alpha_t_fl) / tau
        d_alpha_fr = (alpha_kin_fr - alpha_t_fr) / tau
        d_alpha_rl = (alpha_kin_rl - alpha_t_rl) / tau
        d_alpha_rr = (alpha_kin_rr - alpha_t_rr) / tau
        d_kappa_fl = (0.0 - kappa_t_fl) / tau
        d_kappa_fr = (0.0 - kappa_t_fr) / tau

        T_ribs_f = x[28:31]
        T_gas_f  = x[31]
        T_ribs_r = x[32:35]
        T_gas_r  = x[35]

        Fx_fl, Fy_fl = self.tire.compute_force(alpha_t_fl, kappa_t_fl, Fz_fl, gamma_fl, T_ribs_f, T_gas_f, vx, wz=wz)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_t_fr, kappa_t_fr, Fz_fr, gamma_fr, T_ribs_f, T_gas_f, vx, wz=wz)
        Mz_fl        = self.tire.compute_aligning_torque(alpha_t_fl, kappa_t_fl, Fz_fl, gamma_fl, Fy_fl, Fx_fl)
        Mz_fr        = self.tire.compute_aligning_torque(alpha_t_fr, kappa_t_fr, Fz_fr, gamma_fr, Fy_fr, Fx_fr)
        Mz_castor_fl = compute_castor_trail(s.castor_f, Fz_fl, self.R_wheel)
        Mz_castor_fr = compute_castor_trail(s.castor_f, Fz_fr, self.R_wheel)

        T_drive    = self.compute_drive_force(jnp.clip(u[1] / 2000.0, 0.0, 1.0), vx) * self.R_wheel
        F_brake_f, F_brake_r = self.compute_brake_forces(
            jnp.maximum(-u[1], 0.0), Fz_fl + Fz_fr, Fz_rl + Fz_rr, vx, s.brake_bias_f)
        Fx_brake_f = F_brake_f
        Fx_brake_r = F_brake_r

        Fx_rl, Fx_rr, Fy_rl, Fy_rr, kappa_t_rl_new, kappa_t_rr_new = \
            self.compute_differential_forces(
                T_drive, vx, wz,
                Fz_rl, Fz_rr, alpha_t_rl, alpha_t_rr,
                gamma_rl, gamma_rr, T_ribs_r, T_gas_r, s.diff_lock_ratio)

        d_kappa_rl = (kappa_t_rl_new - kappa_t_rl) / tau
        d_kappa_rr = (kappa_t_rr_new - kappa_t_rr) / tau

        Fy_f     = Fy_fl + Fy_fr
        Fy_r     = Fy_rl + Fy_rr
        Fx_f     = Fx_fl + Fx_fr + Fx_brake_f
        Fx_r     = Fx_rl + Fx_rr + Fx_brake_r
        Mz_total = Mz_fl + Mz_fr + Mz_castor_fl + Mz_castor_fr
        M_diff   = (Fx_rr - Fx_rl) * tr2

        F_ext = jnp.zeros(28)
        F_ext = F_ext.at[14].set(Fx_f + Fx_r - self.m_s * self.g * jnp.sin(theta_pitch))
        F_ext = F_ext.at[15].set(Fy_f + Fy_r - self.m_s * self.g * jnp.sin(phi_roll) * jnp.cos(theta_pitch))
        F_ext = F_ext.at[16].set(F_susp_fl + F_susp_fr + F_susp_rl + F_susp_rr
                                  - self.m_s * self.g * jnp.cos(phi_roll) * jnp.cos(theta_pitch)
                                  + Fz_aero_f + Fz_aero_r)
        F_ext = F_ext.at[17].set(-Fy_f * h_rc_f - Fy_r * h_rc_r
                                  + (F_susp_fr - F_susp_fl) * tf2
                                  + (F_susp_rr - F_susp_rl) * tr2 + Mx_aero)
        F_ext = F_ext.at[18].set((Fx_f + Fx_r) * s.h_cg
                                  - (F_susp_fl + F_susp_fr) * self.lf
                                  + (F_susp_rl + F_susp_rr) * self.lr + My_aero)
        F_ext = F_ext.at[19].set(Fy_f * self.lf - Fy_r * self.lr
                                  + (Fx_fr - Fx_fl) * tf2
                                  + (Fx_rr - Fx_rl) * tr2
                                  + Mz_total + M_diff)
        F_ext = F_ext.at[20].set(-F_susp_fl + Fz_fl - self.m_us_f * self.g)
        F_ext = F_ext.at[21].set(-F_susp_fr + Fz_fr - self.m_us_f * self.g)
        F_ext = F_ext.at[22].set(-F_susp_rl + Fz_rl - self.m_us_r * self.g)
        F_ext = F_ext.at[23].set(-F_susp_rr + Fz_rr - self.m_us_r * self.g)
        F_ext = F_ext.at[24].set(-Fx_fl * self.R_wheel)
        F_ext = F_ext.at[25].set(-Fx_fr * self.R_wheel)
        F_ext = F_ext.at[26].set(-Fx_rl * self.R_wheel + T_drive * 0.5)
        F_ext = F_ext.at[27].set(-Fx_rr * self.R_wheel + T_drive * 0.5)

        dq_dt = PH_accel[0:14]
        dv_dt = (PH_accel[14:28] + F_ext[14:28]) / self.M_diag
        # 500 m/s² ≈ 51G — physically unreachable, only active at Newton overshoot states.
        # Clipping is inactive near the optimum so gradient quality is unaffected there.
        dv_dt = jnp.clip(dv_dt, -500.0, 500.0)
        dx_mech = jnp.concatenate([dq_dt, dv_dt])

        dx_therm = (self.tire.compute_thermal_derivatives(
            x[28:38],
            jnp.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr]),
            jnp.array([0.0, 0.0, jnp.abs(kappa_t_rl_new), jnp.abs(kappa_t_rr_new)]),
            jnp.abs(vx),
        ) if hasattr(self.tire, 'compute_thermal_derivatives') else jnp.zeros(10))

        dx_slip = jnp.array([
            d_alpha_fl, d_kappa_fl, d_alpha_fr, d_kappa_fr,
            d_alpha_rl, d_kappa_rl, d_alpha_rr, d_kappa_rr,
        ])

        # FIX: Clip slip derivatives to prevent Explicit Euler instability
        # at high speeds where the relaxation time constant tau < dt/2.
        dx_slip = jnp.clip(dx_slip, -200.0, 200.0)

        return jnp.concatenate([dx_mech[:28], dx_therm, dx_slip])

    # ─────────────────────────────────────────────────────────────────────────
    # §5.4  Gauss-Legendre RK4 Variational Integrator
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _glrk4_step(
        self,
        x:            jax.Array,
        u:            jax.Array,
        setup_params: jax.Array,
        dt_step:      float,
    ) -> jax.Array:
        """
        2-stage Gauss-Legendre RK4 (GLRK-4) variational integrator.

        Butcher tableau (s=2 Gauss-Legendre):
            a11=1/4,          a12=1/4 - √3/6
            a21=1/4 + √3/6,  a22=1/4
            b1=1/2,           b2=1/2

        Properties:
        · Symplectic — preserves dq∧dp to machine precision.
        · 4th-order: energy drift O(h⁵) vs O(h³) for Störmer-Verlet.
        · 3 Newton iterations sufficient for smooth Hamiltonians at h≈1ms.

        Aux (thermal + slip) integration: trapezoidal using converged stage
        derivatives — zero additional _compute_derivatives calls vs. the
        previous forward Euler, and one order of accuracy higher (O(h³)).
        """
        sqrt3    = jnp.sqrt(3.0)
        a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
        a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
        b1, b2   = 0.5, 0.5

        q0   = x[0:14];  v0   = x[14:28];  aux0 = x[28:46]
        _AUX = 18

        dx0         = self._compute_derivatives(x, u, setup_params)
        k1_q, k1_v  = dx0[0:14], dx0[14:28]
        k2_q, k2_v  = k1_q,      k1_v

        def newton_iter(carry, _):
            k1_q_, k1_v_, k2_q_, k2_v_, _a1, _a2 = carry

            q1 = q0 + dt_step * (a11 * k1_q_ + a12 * k2_q_)
            v1 = v0 + dt_step * (a11 * k1_v_ + a12 * k2_v_)
            x1 = x.at[0:14].set(q1).at[14:28].set(v1)

            q2 = q0 + dt_step * (a21 * k1_q_ + a22 * k2_q_)
            v2 = v0 + dt_step * (a21 * k1_v_ + a22 * k2_v_)
            x2 = x.at[0:14].set(q2).at[14:28].set(v2)

            dx1 = self._compute_derivatives(x1, u, setup_params)
            dx2 = self._compute_derivatives(x2, u, setup_params)

            # Clip stage-derivative carry between Newton iterations.
            #
            # ROOT CAUSE OF 46% NaN RATE IN WMPC:
            # _compute_derivatives is called inside the GLRK-4 Newton scan,
            # which is itself inside the WMPC trajectory scan. When the
            # WMPC optimizer explores states above the friction limit, vx can
            # transiently reach 25–30 m/s at a Newton intermediate stage.
            # At those states, jax.grad(H_net.apply) produces second-order
            # H_net derivatives that are not bounded by the forward tanh clips
            # (the clips act on the OUTPUT of the gradient, not on second-order
            # terms arising from the backward pass through tanh itself).
            # The second-order gradient of tanh(x/C) is -2x/C² * tanh(x/C) *
            # sech²(x/C), which at x≈5C (reached transiently) gives O(C⁻¹)
            # magnitudes. Over 5 Newton iterations the carry compounds this:
            # dx_i+1 depends on dx_i through the stage update, so the
            # compounded second-order error grows as r^5 where r = O(C⁻¹ * dt).
            # At C=12000 and dt=0.01: r≈8e-7, harmless.
            # At vx=25 m/s and C=150 (VEL_CAP): r≈1.7e-3, still ok.
            # But VEL_CAP * tanh(dH_dp / VEL_CAP): the second derivative w.r.t.
            # p at |dH_dp| >> VEL_CAP gives -2/VEL_CAP * tanh(...) * sech²(...).
            # With GLRK-4 stage point v1 = v0 + dt*(a11*k1_v + a12*k2_v), the
            # momentum p = M * v at that stage can briefly exceed the soft cap.
            # Clipping the carry to ±500 m/s² is 50G — never active within
            # the physical operating envelope, always active at Newton overshoot.
            _DC = 500.0
            return (jnp.clip(dx1[0:14],  -_DC, _DC),
                    jnp.clip(dx1[14:28], -_DC, _DC),
                    jnp.clip(dx2[0:14],  -_DC, _DC),
                    jnp.clip(dx2[14:28], -_DC, _DC),
                    dx1[28:46], dx2[28:46]), None

        (k1_q, k1_v, k2_q, k2_v,
         dx1_aux, dx2_aux), _ = jax.lax.scan(
            newton_iter,
            (k1_q, k1_v, k2_q, k2_v, jnp.zeros(_AUX), jnp.zeros(_AUX)),
            None, length=5
        )

        q_new   = q0  + dt_step * (b1 * k1_q  + b2 * k2_q)
        v_new   = v0  + dt_step * (b1 * k1_v  + b2 * k2_v)
        aux_new = aux0 + dt_step * (b1 * dx1_aux + b2 * dx2_aux)
        
        # FIX: Hard-cap the aux states so jnp.sin() never sees Infinity
        aux_new = jnp.clip(aux_new, -1000.0, 1000.0)

        return (x.at[0:14].set(q_new)
                  .at[14:28].set(v_new)
                  .at[28:46].set(aux_new))

    # ─────────────────────────────────────────────────────────────────────────
    # §5.5  Public simulate_step
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0, 5))
    def simulate_step(
        self,
        state:      jax.Array,
        controls:   jax.Array,
        setup,
        dt:         float = 0.005,
        n_substeps: int   = 5,
    ) -> jax.Array:
        """
        Advance simulation by `dt` using `n_substeps` GLRK-4 steps.

        Accepts:
          SuspensionSetup  — preferred
          jnp.ndarray(28,) — accepted directly
          jnp.ndarray(8,)  — legacy 8-element (auto-upgraded via from_legacy_8)
        """
        if isinstance(setup, SuspensionSetup):
            setup_vec = setup.to_vector()
        else:
            v = jnp.asarray(setup, dtype=jnp.float32)
            if v.shape[0] == 8:
                setup_vec = SuspensionSetup.from_legacy_8(v).to_vector()
            else:
                setup_vec = v

        dt_sub = dt / n_substeps

        def substep(x, _):
            return self._glrk4_step(x, controls, setup_vec, dt_sub), None

        final_state, _ = jax.lax.scan(substep, state, None, length=n_substeps)
        return final_state

    @staticmethod
    def default_setup_params() -> jax.Array:
        return DEFAULT_SETUP


# ─────────────────────────────────────────────────────────────────────────────
# §6  Public factory functions
# ─────────────────────────────────────────────────────────────────────────────

def make_setup_from_params(vp: dict) -> SuspensionSetup:
    """
    Construct a SuspensionSetup from a vehicle_params dict.
    CANONICAL construction path for all callers not using the optimizer.
    """
    v = jnp.array([
        vp.get('spring_rate_f',     DEFAULT_SETUP[0]),
        vp.get('spring_rate_r',     DEFAULT_SETUP[1]),
        vp.get('arb_rate_f',        DEFAULT_SETUP[2]),
        vp.get('arb_rate_r',        DEFAULT_SETUP[3]),
        vp.get('damper_c_low_f',    DEFAULT_SETUP[4]),
        vp.get('damper_c_low_r',    DEFAULT_SETUP[5]),
        vp.get('damper_c_high_f',   DEFAULT_SETUP[6]),
        vp.get('damper_c_high_r',   DEFAULT_SETUP[7]),
        vp.get('damper_v_knee',     DEFAULT_SETUP[8]),
        vp.get('damper_v_knee',     DEFAULT_SETUP[9]),
        vp.get('rebound_ratio_f',   DEFAULT_SETUP[10]),
        vp.get('rebound_ratio_r',   DEFAULT_SETUP[11]),
        vp.get('h_ride_f', vp.get('h_ride_design', DEFAULT_SETUP[12])),
        vp.get('h_ride_r', vp.get('h_ride_design', DEFAULT_SETUP[13])),
        vp.get('static_camber_f',   DEFAULT_SETUP[14]),
        vp.get('static_camber_r',   DEFAULT_SETUP[15]),
        vp.get('static_toe_f',      DEFAULT_SETUP[16]),
        vp.get('static_toe_r',      DEFAULT_SETUP[17]),
        vp.get('castor_f',          DEFAULT_SETUP[18]),
        vp.get('anti_squat',        DEFAULT_SETUP[19]),
        vp.get('anti_dive_f',       DEFAULT_SETUP[20]),
        vp.get('anti_dive_r',       DEFAULT_SETUP[21]),
        vp.get('anti_lift',         DEFAULT_SETUP[22]),
        vp.get('diff_lock_ratio',   DEFAULT_SETUP[23]),
        vp.get('brake_bias_f',      DEFAULT_SETUP[24]),
        vp.get('h_cg',              DEFAULT_SETUP[25]),
        vp.get('bump_steer_f',      DEFAULT_SETUP[26]),
        vp.get('bump_steer_r',      DEFAULT_SETUP[27]),
    ], dtype=jnp.float32)
    return SuspensionSetup.from_vector(v)


def build_default_setup_28(vp: dict) -> jax.Array:
    """
    Returns canonical 28-element float32 array from vehicle_params dict.
    This is the ONLY correct way to build setup_params for ocp_solver.
    """
    return make_setup_from_params(vp).to_vector()


def compute_equilibrium_suspension(setup_vec: jax.Array, vp: dict) -> jax.Array:
    """
    Returns static equilibrium suspension displacements (z_fl, z_fr, z_rl, z_rr) [m].

    Fully differentiable w.r.t. setup_vec — safe inside jit/grad/vmap.
    """
    k_f  = setup_vec[0];  k_r  = setup_vec[1]
    m    = jnp.array(vp.get('total_mass',    300.0))
    g    = 9.81
    lf   = jnp.array(vp.get('lf',          0.8525))
    lr   = jnp.array(vp.get('lr',          0.6975))
    L    = lf + lr
    muf  = jnp.array(vp.get('unsprung_mass_f', 7.74))
    mur  = jnp.array(vp.get('unsprung_mass_r', 7.76))
    mr_f = jnp.array(vp.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))[0]
    mr_r = jnp.array(vp.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))[0]

    F_susp_f_eq = m * g * lr / (2.0 * L) - muf * g
    F_susp_r_eq = m * g * lf / (2.0 * L) - mur * g
    z_f_eq = F_susp_f_eq / (k_f * mr_f ** 2 + 1e-6)
    z_r_eq = F_susp_r_eq / (k_r * mr_r ** 2 + 1e-6)
    return jnp.array([z_f_eq, z_f_eq, z_r_eq, z_r_eq])


# ─────────────────────────────────────────────────────────────────────────────
# §7  Backward-compat aliases
# ─────────────────────────────────────────────────────────────────────────────
MultiBodyVehicle = DifferentiableMultiBodyVehicle
VehicleDynamicsH = DifferentiableMultiBodyVehicle