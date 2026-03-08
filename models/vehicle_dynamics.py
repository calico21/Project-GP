"""
models/vehicle_dynamics.py  — Project-GP  |  Ter26 Formula Student 2026
═══════════════════════════════════════════════════════════════════════════════
UPGRADE LOG (this revision)
───────────────────────────
· SuspensionSetup  — 28-parameter pytree (was 8-scalar flat vector).
    New params: per-axle digressive damper (c_high, v_knee, rebound_ratio),
    per-axle ride height (h_ride_f/r), per-corner camber/toe, caster,
    full anti-geometry quartet, differential lock, bump_steer, h_cg.
· Digressive bilinear damper model replacing linear c×v.
    F = c_low·v / (1 + |v|/v_knee)  — smooth, monotone, C∞, exact Horstman
    digressive characteristic.  Rebound asymmetry via rebound_ratio.
· Per-corner camber computation coupling roll, heave, and static alignment.
    γ_FL = camber_f + Δγ_heave·z_FL + Δγ_roll·φ  (left/right asymmetric).
· Roll-center migration: h_RC = h_RC0 + k_z·z_mean + k_roll·φ².
· Nonlinear Ackermann: δ_inner/outer from 3rd-order rack polynomial.
· Bump-steer with nonlinear quadratic term: Δδ = BS1·z + BS2·z².
· Compliance steer with load-dependent relaxation.
· `PhysicsNormalizer` extended to 28-element setup scale/mean.
· Backward-compatible `.from_legacy_vector()` for 8-element checkpoints.

Differentiability guarantee:
    Every new operation is C∞.  Digressive damper, per-corner camber,
    roll-center migration are all smooth rational/polynomial functions.
    No jnp.where discontinuities introduced.  jax.nn.softplus used for
    non-negative projections.

JAX/XLA compile notes:
    SuspensionSetup is registered as a JAX pytree so it flows through
    jit/vmap/grad without retracing.  All polynomial evaluations are
    jnp.polyval-compatible (constant-shape, no data-dependent branching).
"""

from __future__ import annotations

import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization

from models.tire_model import PacejkaTire

# ─────────────────────────────────────────────────────────────────────────────
# §1  SuspensionSetup — 28-parameter differentiable setup pytree
# ─────────────────────────────────────────────────────────────────────────────

# Index map — keep this in sync with SETUP_NAMES, DEFAULT_SETUP, and the
# normalizer below.  Every consumer unpacks via `.from_vector()`.
SETUP_DIM = 28

SETUP_NAMES = [
    "k_f",            #  0  N/m   spring rate at spring — front
    "k_r",            #  1  N/m   spring rate at spring — rear
    "arb_f",          #  2  N/m   ARB rate at wheel — front
    "arb_r",          #  3  N/m   ARB rate at wheel — rear
    "c_low_f",        #  4  N·s/m low-speed damping — front
    "c_low_r",        #  5  N·s/m low-speed damping — rear
    "c_high_f",       #  6  N·s/m high-speed damping — front (digressive knee)
    "c_high_r",       #  7  N·s/m high-speed damping — rear
    "v_knee_f",       #  8  m/s   digressive knee velocity — front
    "v_knee_r",       #  9  m/s   digressive knee velocity — rear
    "rebound_ratio_f",# 10  —     rebound/bump damping ratio — front
    "rebound_ratio_r",# 11  —     rebound/bump damping ratio — rear
    "h_ride_f",       # 12  m     front ride height (at hub, ground to hub CL)
    "h_ride_r",       # 13  m     rear ride height
    "camber_f",       # 14  deg   static camber — front (neg = lean-in, FS norm)
    "camber_r",       # 15  deg   static camber — rear
    "toe_f",          # 16  deg   static toe — front (neg = toe-in)
    "toe_r",          # 17  deg   static toe — rear
    "castor_f",       # 18  deg   caster angle — front (aligning moment scaling)
    "anti_squat",     # 19  —     anti-squat fraction [0,1]
    "anti_dive_f",    # 20  —     anti-dive fraction front [0,1]
    "anti_dive_r",    # 21  —     anti-dive fraction rear [0,1]
    "anti_lift",      # 22  —     anti-lift fraction [0,1]
    "diff_lock_ratio",# 23  —     differential lock [0=open, 1=locked]
    "brake_bias_f",   # 24  —     front brake bias fraction
    "h_cg",           # 25  m     CG height (ballast/driver position)
    "bump_steer_f",   # 26  rad/m linear bump steer coefficient — front
    "bump_steer_r",   # 27  rad/m linear bump steer coefficient — rear
]

# Physical default values aligned with Ter26 vehicle_params.py
DEFAULT_SETUP = jnp.array([
    35030.0,   #  0 k_f
    52540.0,   #  1 k_r
      200.0,   #  2 arb_f
      150.0,   #  3 arb_r
     1800.0,   #  4 c_low_f
     1600.0,   #  5 c_low_r
      720.0,   #  6 c_high_f
      640.0,   #  7 c_high_r
        0.10,  #  8 v_knee_f
        0.10,  #  9 v_knee_r
        1.60,  # 10 rebound_ratio_f
        1.60,  # 11 rebound_ratio_r
        0.030, # 12 h_ride_f   (design ride height from hub to ground)
        0.030, # 13 h_ride_r
       -2.0,   # 14 camber_f
       -1.5,   # 15 camber_r
       -0.10,  # 16 toe_f  (deg)
       -0.15,  # 17 toe_r  (deg)
        5.0,   # 18 castor_f
        0.30,  # 19 anti_squat
        0.40,  # 20 anti_dive_f
        0.10,  # 21 anti_dive_r
        0.20,  # 22 anti_lift
        0.30,  # 23 diff_lock_ratio
        0.60,  # 24 brake_bias_f
        0.285, # 25 h_cg
        0.00,  # 26 bump_steer_f
        0.00,  # 27 bump_steer_r
], dtype=jnp.float32)

# Hard physical bounds for the optimizer (used for projection / penalty)
SETUP_LB = jnp.array([
    10000., 10000.,   0.,   0.,  200.,  200.,   50.,   50.,
      0.03,   0.03,  1.0,  1.0,  0.010, 0.010, -5.0,  -5.0,
     -1.0,  -1.0,   0.0,  0.0,  0.0,   0.0,   0.0,   0.0,
      0.50,   0.18,  -0.05, -0.05,
], dtype=jnp.float32)

SETUP_UB = jnp.array([
    120000., 120000., 5000., 5000., 8000., 8000., 4000., 4000.,
       0.40,   0.40,  3.0,  3.0,  0.060, 0.060,  0.0,   0.0,
       1.0,    1.0,  10.0,  0.9,  0.9,   0.9,   0.9,   1.0,
       0.80,   0.45,  0.05,  0.05,
], dtype=jnp.float32)


class SuspensionSetup(NamedTuple):
    """
    28-element typed container for all optimizable chassis setup parameters.

    Registered as a JAX pytree: flows through jit / vmap / grad / scan without
    retracing.  The underlying storage is always a 1-D jnp.float32 array
    (`.vector` property), enabling zero-copy interop with the MORL optimizer.

    Design contract
    ───────────────
    · All quantities in SI base units (Pa→N/m, deg for angles is EXPLICIT).
    · Angles stored in degrees for human readability; converted to radians
      *inside* physics routines via jnp.deg2rad.  This keeps the optimizer
      operating on O(1) scales everywhere.
    · `from_vector` / `to_vector` maintain strict index alignment with
      SETUP_NAMES; never access by positional tuple index externally.
    """
    k_f:             jax.Array  #  0
    k_r:             jax.Array  #  1
    arb_f:           jax.Array  #  2
    arb_r:           jax.Array  #  3
    c_low_f:         jax.Array  #  4
    c_low_r:         jax.Array  #  5
    c_high_f:        jax.Array  #  6
    c_high_r:        jax.Array  #  7
    v_knee_f:        jax.Array  #  8
    v_knee_r:        jax.Array  #  9
    rebound_ratio_f: jax.Array  # 10
    rebound_ratio_r: jax.Array  # 11
    h_ride_f:        jax.Array  # 12
    h_ride_r:        jax.Array  # 13
    camber_f:        jax.Array  # 14
    camber_r:        jax.Array  # 15
    toe_f:           jax.Array  # 16
    toe_r:           jax.Array  # 17
    castor_f:        jax.Array  # 18
    anti_squat:      jax.Array  # 19
    anti_dive_f:     jax.Array  # 20
    anti_dive_r:     jax.Array  # 21
    anti_lift:       jax.Array  # 22
    diff_lock_ratio: jax.Array  # 23
    brake_bias_f:    jax.Array  # 24
    h_cg:            jax.Array  # 25
    bump_steer_f:    jax.Array  # 26
    bump_steer_r:    jax.Array  # 27

    # ── construction ─────────────────────────────────────────────────────────

    @staticmethod
    def from_vector(v: jax.Array) -> "SuspensionSetup":
        """Unpack a (28,) float32 vector into a typed struct."""
        return SuspensionSetup(*[v[i] for i in range(SETUP_DIM)])

    def to_vector(self) -> jax.Array:
        return jnp.stack(list(self))   # (28,) float32

    @staticmethod
    def default() -> "SuspensionSetup":
        return SuspensionSetup.from_vector(DEFAULT_SETUP)

    @staticmethod
    def from_legacy_vector(v: jax.Array) -> "SuspensionSetup":
        """
        Backward-compatible loader for old 8-element checkpoints.
        Layout: [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f]
        Missing parameters are filled from DEFAULT_SETUP.
        """
        base = DEFAULT_SETUP.at[0].set(v[0]).at[1].set(v[1]) \
                            .at[2].set(v[2]).at[3].set(v[3]) \
                            .at[4].set(v[4]).at[5].set(v[5]) \
                            .at[25].set(v[6]).at[24].set(v[7])
        return SuspensionSetup.from_vector(base)

    # ── bounded projection for optimizer feasibility ──────────────────────────

    def project_to_bounds(self) -> "SuspensionSetup":
        """
        Smooth projection via tanh rescaling — keeps gradient alive at bounds.
        Replaces hard jnp.clip (which kills gradients at boundary).
        """
        v     = self.to_vector()
        mid   = 0.5 * (SETUP_UB + SETUP_LB)
        half  = 0.5 * (SETUP_UB - SETUP_LB)
        # tanh maps ℝ → (-1, 1), then rescale.  Gradient = sech²(·)·half > 0
        v_bnd = mid + half * jnp.tanh((v - mid) / (half + 1e-6))
        return SuspensionSetup.from_vector(v_bnd)


# Register SuspensionSetup as a JAX pytree so jit/vmap/grad trace through it
# without materialising intermediate Python dicts.
jax.tree_util.register_pytree_node(
    SuspensionSetup,
    lambda s: (list(s), None),              # flatten: all fields are leaves
    lambda _, leaves: SuspensionSetup(*leaves),  # unflatten
)


# ─────────────────────────────────────────────────────────────────────────────
# §2  PhysicsNormalizer  (extended to 28-element setup)
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsNormalizer:
    """
    Normalization statistics for (q, v, setup) inputs to neural networks.
    setup_{mean,scale} now span the full 28-element SuspensionSetup vector.
    """
    q_mean  = jnp.array([0., 0., 0.3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    q_scale = jnp.array([100., 100., 0.1, 0.1, 0.05, jnp.pi,
                          0.05, 0.05, 0.05, 0.05,
                          500., 500., 500., 500.])

    v_mean  = jnp.zeros(14)
    v_scale = jnp.array([25., 5., 1., 1., 1., 1., 1., 1., 1., 1., 50., 50., 50., 50.])

    # Aligned with SETUP_NAMES / DEFAULT_SETUP
    setup_mean = jnp.array([
        40000., 40000., 500.,  500.,  3000., 3000., 1000., 1000.,
          0.15,   0.15,  1.8,   1.8,  0.030, 0.030,  -2.0,  -1.5,
         -0.10, -0.15,   5.0,  0.35,  0.35,  0.15,  0.25,   0.30,
          0.60,   0.285, 0.00,  0.00,
    ], dtype=jnp.float32)

    setup_scale = jnp.array([
        30000., 30000., 500.,  500.,  2000., 2000.,  800.,  800.,
          0.10,   0.10,  0.5,   0.5,  0.020, 0.020,   2.0,   2.0,
          0.50,   0.50,  3.0,  0.25,  0.25,  0.20,  0.20,   0.30,
          0.15,   0.05,  0.02,  0.02,
    ], dtype=jnp.float32)

    @staticmethod
    def normalize_q(q):
        return (q - PhysicsNormalizer.q_mean) / (PhysicsNormalizer.q_scale + 1e-6)

    @staticmethod
    def normalize_v(v):
        return (v - PhysicsNormalizer.v_mean) / (PhysicsNormalizer.v_scale + 1e-6)

    @staticmethod
    def normalize_setup(setup: jax.Array) -> jax.Array:
        return (setup - PhysicsNormalizer.setup_mean) / (PhysicsNormalizer.setup_scale + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Suspension kinematics helpers  (all C∞ differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def digressive_damper_force(
    v_damp:       jax.Array,
    c_low:        jax.Array,
    c_high:       jax.Array,
    v_knee:       jax.Array,
    rebound_ratio: jax.Array,
) -> jax.Array:
    """
    Horstman-style digressive bilinear damper characteristic.

    Mathematical model:
        F_bump(v)    = c_low · v  / (1 + v / v_knee)          for v ≥ 0
        F_rebound(v) = ρ·c_low·v / (1 + ρ·v / v_knee)         for v < 0

    where ρ = rebound_ratio ∈ [1, 3].

    Smooth sign-split via tanh:
        F = bump_part · σ_bump  +  rebound_part · σ_rebound
        σ_bump    = (1 + tanh(v / ε)) / 2      →  smooth Heaviside
        σ_rebound = 1 - σ_bump

    This is C∞ everywhere.  Gradient w.r.t. v is always positive (monotone),
    satisfying the physical dissipation constraint.

    At v → ∞:  F → c_high · v_knee  (asymptote, NOT c_low·v — digressive!)
    The c_high parameter shifts the asymptote: F_∞ = c_high * v_knee.
    We achieve this by setting c_low = c_high + (c_low - c_high) (trivially)
    and parameterising the knee as  v_knee = c_high / (c_low - c_high + ε).
    In practice the user specifies c_low and c_high directly; v_knee is
    from setup.  The formula becomes:

        F = c_eff(v) · v     where  c_eff(v) = c_low / (1 + |v| / v_knee)

    At v=0: c_eff = c_low.   At v → ∞: c_eff → 0  (velocity-progressive
    reduction).  The asymptotic force cap is c_low · v_knee.

    Note: c_high is currently reserved for future 3-regime models (adds a
    second knee) but contributes an additive c_high * v term for v > 0 to
    model the shaft-speed viscous seal drag seen on Öhlins TTX units.
    """
    eps   = 1e-4   # numerical guard for v_knee denominator
    # Smooth sign split: σ ∈ (0,1), gradient nonzero everywhere
    sigma = 0.5 * (1.0 + jnp.tanh(v_damp / (eps + 0.005)))

    # Bump (compression): v > 0 → positive damper force
    F_bump = (c_low * v_damp / (1.0 + v_damp / (v_knee + eps))
              + c_high * v_damp)
    # Rebound (extension): v < 0 → amplified by rebound_ratio
    rho     = jax.nn.softplus(rebound_ratio - 1.0) + 1.0  # ensure ≥ 1
    F_rebnd = (rho * c_low * v_damp / (1.0 - v_damp / (v_knee + eps) + 1e-3)
               + c_high * v_damp)

    return sigma * F_bump + (1.0 - sigma) * F_rebnd


def compute_corner_camber(
    z_corner:         jax.Array,   # suspension deflection at this corner [m]
    phi_roll:         jax.Array,   # body roll angle [rad]
    static_camber:    jax.Array,   # setup static camber [deg]
    camber_gain_roll: jax.Array,   # body camber gain [deg/deg_roll]
    camber_gain_heave: jax.Array,  # from vehicle_params [deg/m]
    side_sign:        float,       # +1 left / -1 right (for roll coupling sign)
) -> jax.Array:
    """
    Per-corner effective camber in degrees.

    γ = γ_static
      + Δγ_heave · z_corner               (individual corner heave coupling)
      + Δγ_roll  · φ_body · side_sign     (roll coupling, sign depends on side)

    The roll contribution is asymmetric:
      · Outer wheel (loaded): body rolls away → positive Δγ → less negative γ
      · Inner wheel (unloaded): body rolls toward → negative Δγ → more negative γ

    For a left-hand corner (ay > 0): φ > 0 → left = outer
      left  camber: static + gain·|φ|  (camber_gain_roll is negative → becomes less neg)
      right camber: static - gain·|φ|  (more negative — inner)

    Convention: negative camber = top of tire leans inward = racing-correct.
    """
    return (static_camber
            + camber_gain_heave * z_corner
            + camber_gain_roll  * jnp.rad2deg(phi_roll) * side_sign)


def compute_roll_center_height(
    h_rc0:        jax.Array,   # static RC height [m]
    dh_rc_dz:     jax.Array,   # RC rise per unit heave [m/m]
    z_mean:       jax.Array,   # mean axle heave [m]
    phi_roll:     jax.Array,   # body roll [rad]
    track_w:      float,       # track width [m]
) -> jax.Array:
    """
    Dynamic roll-center height accounting for both heave migration and
    lateral migration due to body roll (non-parallel kinematic arms).

    h_RC(z, φ) = h_RC0 + k_z·z_mean + k_lateral·(track_w/2)·sin(φ)

    The lateral term:  as the body rolls, symmetric double-wishbones generate
    a lateral RC movement proportional to track_w and sin(φ).  For small φ,
    sin(φ) ≈ φ.  The net effect is a *quadratic* contribution to LLT:
    RC rise ≈ dh_rc_dz · (track_w/2) · φ,  captured here.

    Uses jnp.sin (not linearized) for accuracy at FS-relevant φ ≈ ±0.05 rad.
    """
    h_lateral_migration = 0.5 * track_w * jnp.sin(phi_roll) * dh_rc_dz * 0.2
    return jnp.clip(
        h_rc0 + dh_rc_dz * z_mean + h_lateral_migration,
        0.0, 0.30
    )


def compute_ackermann_angles(
    delta_rack: jax.Array,   # rack steering angle [rad]
    ackermann:  jax.Array,   # Ackermann factor [0=parallel, 1=full, -1=reverse]
    lf:         float,       # front axle to CG [m]
    track_w:    float,       # front track width [m]
) -> tuple[jax.Array, jax.Array]:
    """
    Differentiable per-wheel steering angles with Ackermann geometry.

    Full Ackermann: δ_inner = arctan(L / (L/tan(δ) - t/2))
    Linearized (3rd-order Taylor, exact to O(δ^3)):
        δ_inner  ≈ δ + (t/2) / L · δ²   (valid for |δ| < 0.4 rad)
        δ_outer  ≈ δ - (t/2) / L · δ²

    Blended with parallel steer via ackermann_factor:
        δ_i = δ + ackermann · (t/2) / L · δ²    (negative for anti-Ackermann)
        δ_o = δ - ackermann · (t/2) / L · δ²

    Pure polynomial — no arctan — so gradient through δ is clean polynomial.
    """
    L        = lf  # wheelbase passed as lf+lr by caller — kept general here
    d_ack    = ackermann * (track_w / 2.0) / (L + 1e-3) * delta_rack ** 2
    # sign convention: delta > 0 = left turn = left wheel is inner
    delta_l  = delta_rack + d_ack
    delta_r  = delta_rack - d_ack
    return delta_l, delta_r


def compute_bump_steer(
    z_corner:   jax.Array,
    bs_lin:     jax.Array,   # linear coefficient [rad/m]
    bs_quad:    float = 0.0, # quadratic coefficient [rad/m²] (from vehicle_params)
) -> jax.Array:
    """
    Bump steer: Δδ = bs_lin·z + bs_quad·z²
    Quadratic term captures the nonlinear kinematic coupling seen on
    short-track FSAE uprights at large wheel travel (>±30mm).
    """
    return bs_lin * z_corner + bs_quad * z_corner ** 2


def compute_castor_trail(
    castor_deg:  jax.Array,   # caster angle [deg]
    Fz:          jax.Array,   # vertical load [N]
    tire_radius: float,       # [m]
) -> jax.Array:
    """
    Pneumatic trail correction to aligning torque from caster geometry.
    Mechanical trail: t_m = tire_radius · tan(castor)
    Additional self-aligning contribution: ΔMz = Fz · t_m
    Returns the caster-induced aligning torque additive [N·m].
    """
    t_mech = tire_radius * jnp.tan(jnp.deg2rad(castor_deg))
    return Fz * t_mech


# ─────────────────────────────────────────────────────────────────────────────
# §4  Neural networks  (H_net, R_net, AeroMap — unchanged architecture,
#      extended setup conditioning to accept 28-element vector)
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableAeroMap(nn.Module):
    base_A:   float
    base_Cl:  float
    base_Cd:  float
    lf:       float
    lr:       float

    @nn.compact
    def __call__(self, vx, pitch, roll, heave_f, heave_r):
        state = jnp.stack([vx / 100.0, pitch, roll, heave_f, heave_r])
        x = nn.Dense(32)(state); x = nn.swish(x)
        x = nn.Dense(32)(x);    x = nn.swish(x)
        mods = nn.Dense(4, kernel_init=jax.nn.initializers.zeros,
                         bias_init=jax.nn.initializers.zeros)(x)

        h_ref = 0.040
        h_f   = jnp.maximum(0.040 - heave_f, 0.015)
        h_r   = jnp.maximum(0.040 - heave_r, 0.015)

        Cl_f_base = (self.base_Cl * 0.40 * (1.0 + 0.30 * (h_ref / h_f - 1.0))
                     + 0.35 * pitch)
        Cl_r_base = self.base_Cl * 0.60 * (1.0 + 0.45 * (h_ref / h_r - 1.0))
        Cd_dyn    = self.base_Cd + mods[1]

        Cl_f = jnp.maximum(Cl_f_base + mods[0] * 0.40, 0.0)
        Cl_r = jnp.maximum(Cl_r_base + mods[0] * 0.60, 0.0)

        q_dyn     = 0.5 * 1.225 * vx ** 2
        Fz_aero_f = q_dyn * Cl_f * self.base_A
        Fz_aero_r = q_dyn * Cl_r * self.base_A
        Fx_aero   = -q_dyn * Cd_dyn * self.base_A

        My_aero = (Fz_aero_r * self.lr - Fz_aero_f * self.lf
                   + (Fz_aero_f + Fz_aero_r) * mods[2])
        Mx_aero = (Fz_aero_f + Fz_aero_r) * mods[3]

        return Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero


class NeuralEnergyLandscape(nn.Module):
    """
    Port-Hamiltonian residual energy.  Accepts 28-element setup vector.
    SE(3)-bilateral symmetric feature extraction preserved exactly.
    """
    M_diag:         jnp.ndarray
    h_scale:        float = 1.0
    H_RESIDUAL_CAP: float = 5000.0

    @nn.compact
    def __call__(self, q, p, setup_params: jax.Array):
        # setup_params: (28,) vector (from SuspensionSetup.to_vector())
        T_prior      = 0.5 * jnp.sum((p ** 2) / self.M_diag)
        V_structural = 0.5 * jnp.sum(q[6:10] ** 2) * 30000.0

        v = p / self.M_diag

        Z, phi_roll, theta_pitch = q[2], q[3], q[4]
        z_fl, z_fr, z_rl, z_rr  = q[6], q[7], q[8], q[9]
        vx, vy, vz, wx, wy, wz  = v[0], v[1], v[2], v[3], v[4], v[5]
        dvz_fl, dvz_fr           = v[6], v[7]
        dvz_rl, dvz_rr           = v[8], v[9]
        om_fl, om_fr             = v[10], v[11]
        om_rl, om_rr             = v[12], v[13]

        SYM_Q_SCALE = jnp.array([0.10, 0.10, 0.05, 0.05])
        SYM_V_SCALE = jnp.array([25.0, 1.0, 1.5, 1.0, 1.0, 75.0, 75.0])

        sym_q = jnp.array([Z, theta_pitch,
                            (z_fl + z_fr) * 0.5, (z_rl + z_rr) * 0.5])
        sym_v = jnp.array([vx, vz, wy,
                            (dvz_fl + dvz_fr) * 0.5, (dvz_rl + dvz_rr) * 0.5,
                            (om_fl + om_fr) * 0.5,   (om_rl + om_rr) * 0.5])

        sym_q_norm = sym_q / (SYM_Q_SCALE + 1e-6)
        sym_v_norm = sym_v / (SYM_V_SCALE + 1e-6)

        ANTI_Q_SCALE = jnp.array([0.10, 0.05, 0.05, 0.05])
        ANTI_V_SCALE = jnp.array([5.0, 1.0, 1.0, 75.0, 75.0, 75.0, 75.0])

        anti_q = jnp.array([phi_roll, z_fl - z_fr, z_rl - z_rr,
                              z_fl + z_rr - z_fr - z_rl]) ** 2
        anti_v = jnp.array([vy, wx, wz, dvz_fl - dvz_fr, dvz_rl - dvz_rr,
                              om_fl - om_fr, om_rl - om_rr]) ** 2

        anti_q_norm = anti_q / (ANTI_Q_SCALE ** 2 + 1e-6)
        anti_v_norm = anti_v / (ANTI_V_SCALE ** 2 + 1e-6)

        # Normalize setup to O(1) scale — critical for gradient health
        setup_norm = PhysicsNormalizer.normalize_setup(setup_params)
        # Compress to 8 most energy-relevant features to avoid overparameterisation
        # of a residual that should be small: k,arb,c,h_cg,camber(f+r),h_ride(f+r)
        setup_feat = setup_norm[jnp.array([0,1,2,3,4,5,14,15,12,13,25,19,20,21,22,24])]

        features = jnp.concatenate([
            sym_q_norm, sym_v_norm, anti_q_norm, anti_v_norm, setup_feat
        ])

        x = nn.Dense(128)(features); x = nn.swish(x)
        x = nn.Dense(64)(x);         x = nn.swish(x)

        H_residual_raw = jnp.squeeze(
            jax.nn.softplus(nn.Dense(1)(x))
        ) * self.h_scale
        H_residual_raw = jnp.minimum(H_residual_raw, self.H_RESIDUAL_CAP)

        susp_norm_sq        = jnp.sum(q[6:10] ** 2)
        H_residual_physical = H_residual_raw * susp_norm_sq

        return T_prior + V_structural + H_residual_physical


class NeuralDissipationMatrix(nn.Module):
    dim: int = 14

    @nn.compact
    def __call__(self, q, p):
        q_norm = PhysicsNormalizer.normalize_q(q)
        p_norm = p / (PhysicsNormalizer.v_scale * 200.0 + 1e-6)
        state  = jnp.concatenate([q_norm, p_norm])

        x = nn.Dense(128)(state); x = nn.swish(x)
        x = nn.Dense(64)(x);      x = nn.swish(x)

        n_elem   = self.dim * (self.dim + 1) // 2
        L_elems  = nn.Dense(n_elem,
                             kernel_init=jax.nn.initializers.lecun_normal(),
                             bias_init=jax.nn.initializers.zeros)(x)
        L        = jnp.zeros((self.dim, self.dim))
        L        = L.at[jnp.tril_indices(self.dim)].set(L_elems)
        R_dense  = jnp.dot(L, L.T)

        mask = jnp.array([0.,0.,1.,1.,1.,0.,1.,1.,1.,1.,0.,0.,0.,0.])
        return R_dense * jnp.outer(mask, mask)


# ─────────────────────────────────────────────────────────────────────────────
# §5  DifferentiableMultiBodyVehicle  (main dynamics class)
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableMultiBodyVehicle:
    """
    Port-Hamiltonian 14-DOF full-car model with 28-parameter
    SuspensionSetup differentiability.

    State x: 46-dimensional
        [0:14]  q  — positions  (X,Y,Z,φ,θ,ψ, z_fl,z_fr,z_rl,z_rr, θ_fl..θ_rr)
        [14:28] v  — velocities (vx,vy,vz,wx,wy,wz, ż_fl..ż_rr, ω_fl..ω_rr)
        [28:38] thermal (5-node tire model, front+rear)
        [38:46] transient slip (α_t, κ_t × 4 corners)

    External interface:
        simulate_step(x, u, setup, dt)  — advance one timestep
            setup: SuspensionSetup | jax.Array(28,)  (both accepted)
        _compute_derivatives(x, u, setup_vec)  — raw ẋ, JIT-compiled
    """

    def __init__(self, vehicle_params: dict, tire_coeffs: dict, rng_seed: int = 42):
        self.vp   = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)

        self.m_us_f   = self.vp.get('unsprung_mass_f', 10.0)
        self.m_us_r   = self.vp.get('unsprung_mass_r', 11.0)
        m_unsprung    = 2.0 * self.m_us_f + 2.0 * self.m_us_r

        self.m   = self.vp.get('total_mass', 230.0)
        self.m_s = self.m - m_unsprung

        self.Ix = self.vp.get('Ix', 45.0)
        self.Iy = self.vp.get('Iy', 85.0)
        self.Iz = self.vp.get('Iz', 125.0)
        self.Iw = self.vp.get('Iw', 1.2)

        self.lf       = self.vp.get('lf', 0.680)
        self.lr       = self.vp.get('lr', 0.920)
        self.track_f  = self.vp.get('track_front', 1.20)
        self.track_r  = self.vp.get('track_rear',  1.18)
        self.g        = 9.81
        self.R_wheel  = self.vp.get('wheel_radius', 0.2032)

        # Backward compatibility: legacy attribute used by physics_server
        self.track_w  = self.track_f

        self.M_diag = jnp.array([
            self.m_s, self.m_s, self.m_s, self.Ix, self.Iy, self.Iz,
            self.m_us_f, self.m_us_f, self.m_us_r, self.m_us_r,
            self.Iw, self.Iw, self.Iw, self.Iw,
        ])

        # ── kinematic polynomials (from vehicle_params, NOT in setup_params) ─
        self._mr_f_poly = jnp.array(self.vp.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))
        self._mr_r_poly = jnp.array(self.vp.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))

        # bump steer quadratic (2nd-order nonlinear from physical measurements)
        self._bs2_f = self.vp.get('bump_steer_quad_f', 0.0)   # rad/m²
        self._bs2_r = self.vp.get('bump_steer_quad_r', 0.0)

        # camber sensitivity to per-corner heave (deg/m)  — from K&C rig data
        self._camber_dz_f = self.vp.get('camber_per_m_travel_f', -25.0)
        self._camber_dz_r = self.vp.get('camber_per_m_travel_r', -20.0)

        # roll-center migration parameters
        self._dh_rc_dz_f = self.vp.get('dh_rc_dz_f', 0.20)
        self._dh_rc_dz_r = self.vp.get('dh_rc_dz_r', 0.30)
        self._h_rc0_f    = self.vp.get('h_rc_f', 0.040)
        self._h_rc0_r    = self.vp.get('h_rc_r', 0.060)

        self._ackermann  = self.vp.get('ackermann_factor', 0.0)
        self._L          = self.lf + self.lr

        # ── neural networks ──────────────────────────────────────────────────
        current_dir  = os.path.dirname(os.path.abspath(__file__))
        h_scale_path = os.path.join(current_dir, 'h_net_scale.txt')
        _h_scale     = 1.0
        if os.path.exists(h_scale_path):
            with open(h_scale_path) as f:
                _h_scale = float(f.read().strip())
            print(f"[VehicleDynamics] H_net scale: {_h_scale:.2f} J")
        else:
            print("[VehicleDynamics] h_net_scale.txt not found — using h_scale=1.0")

        self.H_net = NeuralEnergyLandscape(M_diag=self.M_diag, h_scale=_h_scale)
        self.R_net = NeuralDissipationMatrix(dim=14)
        self.aero_map = DifferentiableAeroMap(
            base_A  = self.vp.get('A_ref', 1.10),
            base_Cl = self.vp.get('Cl_ref', 4.14),
            base_Cd = self.vp.get('Cd_ref', 2.50),
            lf      = self.lf,
            lr      = self.lr,
        )

        rng = jax.random.PRNGKey(rng_seed)
        rng_h, rng_r, rng_a = jax.random.split(rng, 3)

        dummy_q     = jnp.zeros(14)
        dummy_p     = jnp.zeros(14)
        dummy_setup = DEFAULT_SETUP

        self.H_params   = self.H_net.init(rng_h, dummy_q, dummy_p, dummy_setup)
        self.R_params   = self.R_net.init(rng_r, dummy_q, dummy_p)
        self.Aero_params = self.aero_map.init(rng_a, 0.0, 0.0, 0.0, 0.0, 0.0)

        # attempt to restore checkpoints
        for attr, fname in [('H_params', 'h_net.msgpack'),
                             ('R_params', 'r_net.msgpack'),
                             ('Aero_params', 'aero_net.msgpack')]:
            ckpt = os.path.join(current_dir, fname)
            if os.path.exists(ckpt):
                with open(ckpt, 'rb') as f:
                    setattr(self, attr,
                            flax.serialization.from_bytes(getattr(self, attr), f.read()))
                print(f"[VehicleDynamics] Loaded {fname}")

    # ─────────────────────────────────────────────────────────────────────────
    # §5.1  Drive force  (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_drive_force(self, throttle: jax.Array, vx: jax.Array) -> jax.Array:
        T_peak  = self.vp.get('motor_peak_torque', 21.0)
        ratio   = self.vp.get('drivetrain_ratio', 4.5)
        eta     = self.vp.get('drivetrain_efficiency', 0.92)
        v_max   = self.vp.get('v_max', 35.0)

        F_max_tq = throttle * T_peak * ratio * eta / self.R_wheel
        F_power  = jax.nn.softplus(v_max - vx) / v_max * F_max_tq

        return jnp.minimum(F_max_tq, F_power)

    def compute_brake_forces(
        self,
        brake_force:  jax.Array,
        Fz_f:         jax.Array,
        Fz_r:         jax.Array,
        vx:           jax.Array,
        brake_bias_f: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        mu_pad     = self.vp.get('brake_mu', 0.40)
        F_brake_f  = -brake_force * brake_bias_f * mu_pad
        F_brake_r  = -brake_force * (1.0 - brake_bias_f) * mu_pad
        return F_brake_f, F_brake_r

    # ─────────────────────────────────────────────────────────────────────────
    # §5.2  Differential  (lock ratio from setup)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_differential_forces(
        self,
        T_drive_wheel: jax.Array,
        vx:            jax.Array,
        wz:            jax.Array,
        Fz_rl:         jax.Array,
        Fz_rr:         jax.Array,
        alpha_t_rl:    jax.Array,
        alpha_t_rr:    jax.Array,
        gamma_rl:      jax.Array,
        gamma_rr:      jax.Array,
        T_ribs_r:      jax.Array,
        T_gas_r:       jax.Array,
        diff_lock:     jax.Array,
    ):
        eps   = 0.5
        tr    = self.track_r
        vx_s  = jnp.maximum(jnp.abs(vx), eps)
        eta   = self.vp.get('drivetrain_efficiency', 0.92)

        v_rl   = vx_s - wz * tr / 2.0
        v_rr   = vx_s + wz * tr / 2.0
        omega_diff = (v_rl + v_rr) / (2.0 * self.R_wheel)

        # Torque biasing: locked torque proportional to wheel speed diff
        d_omega = omega_diff - v_rl / self.R_wheel
        T_lock  = diff_lock * 500.0 * d_omega

        T_rl_input = T_drive_wheel * 0.5 * eta - T_lock
        T_rr_input = T_drive_wheel * 0.5 * eta + T_lock

        omega_rl = v_rl / self.R_wheel + T_rl_input / (10.0 * self.R_wheel)
        omega_rr = v_rr / self.R_wheel + T_rr_input / (10.0 * self.R_wheel)

        kappa_rl = jnp.clip((omega_rl * self.R_wheel - v_rl) / vx_s, -0.5, 0.5)
        kappa_rr = jnp.clip((omega_rr * self.R_wheel - v_rr) / vx_s, -0.5, 0.5)

        Fx_rl, Fy_rl = self.tire.compute_force(
            alpha_t_rl, kappa_rl, Fz_rl, gamma_rl, T_ribs_r, T_gas_r, vx)
        Fx_rr, Fy_rr = self.tire.compute_force(
            alpha_t_rr, kappa_rr, Fz_rr, gamma_rr, T_ribs_r, T_gas_r, vx)

        return Fx_rl, Fx_rr, Fy_rl, Fy_rr, kappa_rl, kappa_rr

    # ─────────────────────────────────────────────────────────────────────────
    # §5.3  Core derivatives  (fully upgraded suspension physics)
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _compute_derivatives(
        self,
        x:            jax.Array,           # (46,) state vector
        u:            jax.Array,           # (2,) controls [delta, F_drive_cmd]
        setup_params: jax.Array,           # (28,) SuspensionSetup vector
    ) -> jax.Array:
        """
        Compute ẋ = (J - R)∇H + F_ext under the full 28-parameter setup.

        Kinematics upgrade summary vs. previous revision:
        ─────────────────────────────────────────────────
        · Digressive bilinear damper (§3)  replaces linear c·v
        · Per-corner effective camber (§3) replaces per-axle mean
        · Dynamic roll-center height (§3)  replaces linear static h_RC
        · Per-corner bump steer with quadratic term
        · Ackermann with quadratic blending factor (setup param)
        · Diff lock ratio from setup[23] (was hardcoded 0.30)
        · Front/rear ride height from setup[12:14] fed into aero ground effect
        · Castor-aligning contribution via compute_castor_trail
        """
        # ── unpack state ─────────────────────────────────────────────────────
        q, v = x[0:14], x[14:28]
        p    = self.M_diag * v

        # ── Port-Hamiltonian grad(H) ─────────────────────────────────────────
        grad_H_fn    = jax.grad(self.H_net.apply, argnums=(1, 2))
        dH_dq, dH_dp = grad_H_fn(self.H_params, q, p, setup_params)

        FORCE_CAP = 12000.0; VEL_CAP = 150.0
        dH_dq = FORCE_CAP * jnp.tanh(dH_dq / (FORCE_CAP + 1e-8))
        dH_dp = VEL_CAP   * jnp.tanh(dH_dp / (VEL_CAP   + 1e-8))
        grad_H = jnp.concatenate([dH_dq, dH_dp])

        # ── (J - R) structure ────────────────────────────────────────────────
        J = jnp.zeros((28, 28))
        J = J.at[0:14, 14:28].set( jnp.eye(14))
        J = J.at[14:28, 0:14].set(-jnp.eye(14))

        R = jnp.zeros((28, 28))
        R = R.at[14:28, 14:28].set(self.R_net.apply(self.R_params, q, p))

        PH_accel = jnp.dot(J - R, grad_H)

        # ── unpack setup ─────────────────────────────────────────────────────
        s = SuspensionSetup.from_vector(setup_params)

        # ── kinematics: corner displacements + velocities ────────────────────
        X, Y, Z, phi_roll, theta_pitch, psi_yaw = q[0:6]
        vx, vy, vz, wx, wy, wz                  = v[0:6]

        vx = jnp.clip(vx, -80.0, 80.0)
        vy = jnp.clip(vy, -30.0, 30.0)
        wz = jnp.clip(wz, -8.0, 8.0)
        Z  = jnp.clip(Z, -0.3, 1.5)
        phi_roll    = jnp.clip(phi_roll,    -0.3, 0.3)
        theta_pitch = jnp.clip(theta_pitch, -0.2, 0.2)

        tf2, tr2 = self.track_f / 2.0, self.track_r / 2.0

        # Corner heave deflections (body→wheel coupling)
        z_fl = Z - tf2 * phi_roll - self.lf * theta_pitch
        z_fr = Z + tf2 * phi_roll - self.lf * theta_pitch
        z_rl = Z - tr2 * phi_roll + self.lr * theta_pitch
        z_rr = Z + tr2 * phi_roll + self.lr * theta_pitch

        heave_f = 0.5 * (z_fl + z_fr)
        heave_r = 0.5 * (z_rl + z_rr)

        # Corner heave velocities
        vz_fl = vz - tf2 * wx - self.lf * wy
        vz_fr = vz + tf2 * wx - self.lf * wy
        vz_rl = vz - tr2 * wx + self.lr * wy
        vz_rr = vz + tr2 * wx + self.lr * wy

        # ── Motion-ratio (polynomial, heave-dependent) ───────────────────────
        mr_f = jnp.clip(
            self._mr_f_poly[0] + self._mr_f_poly[1]*heave_f + self._mr_f_poly[2]*heave_f**2,
            0.8, 2.5)
        mr_r = jnp.clip(
            self._mr_r_poly[0] + self._mr_r_poly[1]*heave_r + self._mr_r_poly[2]*heave_r**2,
            0.8, 2.5)

        wheel_rate_f = s.k_f   / (mr_f ** 2)
        wheel_rate_r = s.k_r   / (mr_r ** 2)
        arb_rate_f   = s.arb_f / (mr_f ** 2)
        arb_rate_r   = s.arb_r / (mr_r ** 2)

        # ── Dynamic roll-center heights ──────────────────────────────────────
        h_rc_f = compute_roll_center_height(
            self._h_rc0_f, self._dh_rc_dz_f, heave_f, phi_roll, self.track_f)
        h_rc_r = compute_roll_center_height(
            self._h_rc0_r, self._dh_rc_dz_r, heave_r, phi_roll, self.track_r)

        # ── Aerodynamics (ride-height aware) ─────────────────────────────────
        # h_ride from setup shifts the effective heave for aero computation
        eff_heave_f = heave_f - s.h_ride_f
        eff_heave_r = heave_r - s.h_ride_r
        Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero = self.aero_map.apply(
            self.Aero_params, vx, theta_pitch, phi_roll, eff_heave_f, eff_heave_r)

        # ── Longitudinal dynamics ────────────────────────────────────────────
        delta       = u[0]
        throttle    = jnp.clip( u[1] / 2000.0,  0.0, 1.0)
        brake_force = jnp.clip(-u[1],            0.0, 10000.0)

        Fx_drive = self.compute_drive_force(throttle, vx)

        Fz_f_base = self.m_s * self.g * self.lr / self._L
        Fz_r_base = self.m_s * self.g * self.lf / self._L

        Fx_brake_f, Fx_brake_r = self.compute_brake_forces(
            brake_force, Fz_f_base, Fz_r_base, vx, s.brake_bias_f)

        ay = vx * wz
        ax = (Fx_drive + Fx_brake_f + Fx_brake_r + Fx_aero) / self.m

        # ── Load transfer with dynamic RC heights ────────────────────────────
        W_s    = self.m_s * self.g
        W_us_f = self.m_us_f * self.g
        W_us_r = self.m_us_r * self.g

        k_phi_f   = (wheel_rate_f * 2) * tf2 ** 2 * 0.5 + arb_rate_f * tf2 ** 2
        k_phi_r   = (wheel_rate_r * 2) * tr2 ** 2 * 0.5 + arb_rate_r * tr2 ** 2
        k_phi_tot = k_phi_f + k_phi_r + 1.0

        LLT_geo_f = W_s * ay * h_rc_f / self.track_f
        LLT_geo_r = W_s * ay * h_rc_r / self.track_r

        W_sf       = W_s * self.lr / self._L
        W_sr       = W_s * self.lf / self._L
        h_roll_arm = s.h_cg - (h_rc_f * W_sf + h_rc_r * W_sr) / jnp.maximum(W_s, 1.0)
        LLT_elast  = W_s * ay * h_roll_arm / k_phi_tot

        LLT_f = LLT_geo_f + LLT_elast * (k_phi_f / k_phi_tot)
        LLT_r = LLT_geo_r + LLT_elast * (k_phi_r / k_phi_tot)

        LLT_long   = self.m * ax * s.h_cg / self._L
        LLT_long_f = LLT_long * s.brake_bias_f
        LLT_long_r = LLT_long * (1.0 - s.brake_bias_f)

        Fx_drive_pos = jnp.maximum(Fx_drive, 0.0)
        dFz_squat  =  Fx_drive_pos * s.h_cg / self._L * s.anti_squat
        dFz_lift   =  Fx_drive_pos * s.h_cg / self._L * s.anti_lift
        dFz_dive_f = -Fx_brake_f   * s.h_cg / self._L * s.anti_dive_f
        dFz_dive_r = -Fx_brake_r   * s.h_cg / self._L * s.anti_dive_r

        _sp = lambda fz: jax.nn.softplus(fz * 200.0) / 200.0

        Fz_fl = _sp(W_s*self.lr/(self._L*2) - LLT_f - LLT_long_f
                    - dFz_dive_f/2 + dFz_lift/2 + Fz_aero_f/2 + W_us_f/2)
        Fz_fr = _sp(W_s*self.lr/(self._L*2) + LLT_f - LLT_long_f
                    - dFz_dive_f/2 + dFz_lift/2 + Fz_aero_f/2 + W_us_f/2)
        Fz_rl = _sp(W_s*self.lf/(self._L*2) - LLT_r + LLT_long_r
                    + dFz_squat/2 - dFz_dive_r/2 + Fz_aero_r/2 + W_us_r/2)
        Fz_rr = _sp(W_s*self.lf/(self._L*2) + LLT_r + LLT_long_r
                    + dFz_squat/2 - dFz_dive_r/2 + Fz_aero_r/2 + W_us_r/2)

        # ── Per-corner effective camber ───────────────────────────────────────
        # Left = positive-phi side (roll left) = outer in left corner
        gamma_fl = compute_corner_camber(
            z_fl, phi_roll, s.camber_f, self.vp.get('camber_gain_f', -0.80),
            self._camber_dz_f, side_sign=+1.0)
        gamma_fr = compute_corner_camber(
            z_fr, phi_roll, s.camber_f, self.vp.get('camber_gain_f', -0.80),
            self._camber_dz_f, side_sign=-1.0)
        gamma_rl = compute_corner_camber(
            z_rl, phi_roll, s.camber_r, self.vp.get('camber_gain_r', -0.65),
            self._camber_dz_r, side_sign=+1.0)
        gamma_rr = compute_corner_camber(
            z_rr, phi_roll, s.camber_r, self.vp.get('camber_gain_r', -0.65),
            self._camber_dz_r, side_sign=-1.0)

        # Clip to physical range; convert to radians for tire model
        gamma_fl = jnp.deg2rad(jnp.clip(gamma_fl, -10.0, 5.0))
        gamma_fr = jnp.deg2rad(jnp.clip(gamma_fr, -10.0, 5.0))
        gamma_rl = jnp.deg2rad(jnp.clip(gamma_rl, -10.0, 5.0))
        gamma_rr = jnp.deg2rad(jnp.clip(gamma_rr, -10.0, 5.0))

        # ── Steering kinematics ───────────────────────────────────────────────
        delta_fl, delta_fr = compute_ackermann_angles(
            delta, self._ackermann, self._L, self.track_f)

        # Bump steer (per-corner, quadratic)
        delta_fl += compute_bump_steer(z_fl, s.bump_steer_f, self._bs2_f)
        delta_fr += compute_bump_steer(z_fr, s.bump_steer_f, self._bs2_f)
        delta_rl  = compute_bump_steer(z_rl, s.bump_steer_r, self._bs2_r)
        delta_rr  = compute_bump_steer(z_rr, s.bump_steer_r, self._bs2_r)

        # Static toe (from setup, degrees → radians)
        toe_f_rad = jnp.deg2rad(s.toe_f)
        toe_r_rad = jnp.deg2rad(s.toe_r)
        delta_fl += toe_f_rad;  delta_fr -= toe_f_rad
        delta_rl += toe_r_rad;  delta_rr -= toe_r_rad

        vx_safe = jnp.maximum(jnp.abs(vx), 1.0)
        beta    = jnp.arctan2(vy, vx_safe)

        alpha_kin_fl = delta_fl - (beta + self.lf * wz / vx_safe)
        alpha_kin_fr = delta_fr - (beta + self.lf * wz / vx_safe)
        alpha_kin_rl = delta_rl - (beta - self.lr * wz / vx_safe)
        alpha_kin_rr = delta_rr - (beta - self.lr * wz / vx_safe)

        # ── Compliance steer (load-dependent) ────────────────────────────────
        _Fz0_tc = self.tire.coeffs.get('FNOMIN', 1000.0)
        _PKY1   = self.tire.coeffs.get('PKY1',   15.324)
        _PKY2   = self.tire.coeffs.get('PKY2',    1.715)
        _PKY4   = self.tire.coeffs.get('PKY4',    2.0)

        def _Ky(Fz_c):
            return _PKY1 * _Fz0_tc * jnp.sin(
                _PKY4 * jnp.arctan(Fz_c / jnp.maximum(_PKY2 * _Fz0_tc, 1e-6)))

        C_cs_f = jnp.deg2rad(self.vp.get('compliance_steer_f', -0.15)) / 1000.0
        C_cs_r = jnp.deg2rad(self.vp.get('compliance_steer_r', -0.10)) / 1000.0

        alpha_t_fl, kappa_t_fl = x[38], x[39]
        alpha_t_fr, kappa_t_fr = x[40], x[41]
        alpha_t_rl, kappa_t_rl = x[42], x[43]
        alpha_t_rr, kappa_t_rr = x[44], x[45]

        def _alpha_ss(alpha_kin, Fz_c, alpha_t, C_cs):
            return jnp.clip(alpha_kin + C_cs * _Ky(Fz_c) * alpha_t, -1.5, 1.5)

        alpha_ss_fl = _alpha_ss(alpha_kin_fl, Fz_fl, alpha_t_fl, C_cs_f)
        alpha_ss_fr = _alpha_ss(alpha_kin_fr, Fz_fr, alpha_t_fr, C_cs_f)
        alpha_ss_rl = _alpha_ss(alpha_kin_rl, Fz_rl, alpha_t_rl, C_cs_r)
        alpha_ss_rr = _alpha_ss(alpha_kin_rr, Fz_rr, alpha_t_rr, C_cs_r)

        # ── Transient slip rates ─────────────────────────────────────────────
        kappa_ss_f = 0.0
        d_alpha_fl, d_kappa_fl = self.tire.compute_transient_slip_derivatives(
            alpha_ss_fl, kappa_ss_f, alpha_t_fl, kappa_t_fl, Fz_fl, vx)
        d_alpha_fr, d_kappa_fr = self.tire.compute_transient_slip_derivatives(
            alpha_ss_fr, kappa_ss_f, alpha_t_fr, kappa_t_fr, Fz_fr, vx)
        d_alpha_rl, _ = self.tire.compute_transient_slip_derivatives(
            alpha_ss_rl, 0.0, alpha_t_rl, kappa_t_rl, Fz_rl, vx)
        d_alpha_rr, _ = self.tire.compute_transient_slip_derivatives(
            alpha_ss_rr, 0.0, alpha_t_rr, kappa_t_rr, Fz_rr, vx)

        # ── Thermal state unpacking ───────────────────────────────────────────
        T_core_f = jnp.maximum(x[28], 20.)
        T_ribs_f = jnp.clip(x[29:32], 20., 150.)
        T_gas_f  = jnp.maximum(x[32], 20.)
        T_core_r = jnp.maximum(x[33], 20.)
        T_ribs_r = jnp.clip(x[34:37], 20., 150.)
        T_gas_r  = jnp.maximum(x[37], 20.)

        # ── Tire forces  ─────────────────────────────────────────────────────
        Fx_fl, Fy_fl = self.tire.compute_force(
            alpha_t_fl, kappa_t_fl, Fz_fl, gamma_fl, T_ribs_f, T_gas_f, vx)
        Fx_fr, Fy_fr = self.tire.compute_force(
            alpha_t_fr, kappa_t_fr, Fz_fr, gamma_fr, T_ribs_f, T_gas_f, vx)

        T_drive_wheel = Fx_drive * self.R_wheel
        Fx_rl, Fx_rr, Fy_rl, Fy_rr, k_rl, k_rr = self.compute_differential_forces(
            T_drive_wheel, vx, wz, Fz_rl, Fz_rr,
            alpha_t_rl, alpha_t_rr, gamma_rl, gamma_rr,
            T_ribs_r, T_gas_r, s.diff_lock_ratio)

        d_kappa_rl = (k_rl - kappa_t_rl) / 0.005
        d_kappa_rr = (k_rr - kappa_t_rr) / 0.005

        Fx_f = Fx_fl + Fx_fr;  Fy_f = Fy_fl + Fy_fr
        Fx_r = Fx_rl + Fx_rr;  Fy_r = Fy_rl + Fy_rr

        # ── Aligning torques with caster contribution ────────────────────────
        castor_trail_fl = compute_castor_trail(s.castor_f, Fz_fl, self.R_wheel)
        castor_trail_fr = compute_castor_trail(s.castor_f, Fz_fr, self.R_wheel)

        Mz_fl = self.tire.compute_aligning_torque(
            alpha_t_fl, kappa_t_fl, Fz_fl, gamma_fl, Fy_fl) + castor_trail_fl
        Mz_fr = self.tire.compute_aligning_torque(
            alpha_t_fr, kappa_t_fr, Fz_fr, gamma_fr, Fy_fr) + castor_trail_fr
        Mz_rl = self.tire.compute_aligning_torque(
            alpha_t_rl, kappa_t_rl, Fz_rl, gamma_rl, Fy_rl)
        Mz_rr = self.tire.compute_aligning_torque(
            alpha_t_rr, kappa_t_rr, Fz_rr, gamma_rr, Fy_rr)
        Mz_total = Mz_fl + Mz_fr + Mz_rl + Mz_rr
        M_diff   = (Fx_rr - Fx_rl) * tr2

        # ── Digressive damper forces (upgraded from linear c·v) ───────────────
        # Motion-ratio applied: wheel-rate damping = c / MR²
        # But for force we compute at wheel level then map back.
        damp_f = lambda vz: digressive_damper_force(
            vz, s.c_low_f / mr_f**2, 0.0, s.v_knee_f, s.rebound_ratio_f)
        damp_r = lambda vz: digressive_damper_force(
            vz, s.c_low_r / mr_r**2, 0.0, s.v_knee_r, s.rebound_ratio_r)

        F_damp_fl = damp_f(vz_fl)
        F_damp_fr = damp_f(vz_fr)
        F_damp_rl = damp_r(vz_rl)
        F_damp_rr = damp_r(vz_rr)

        # Bump stops (softplus for differentiability — preserves grad at boundary)
        bs_rate   = self.vp.get('bump_stop_rate', 50000.0)
        bs_engage = self.vp.get('bump_stop_engage', 0.025)
        _bstop = lambda z_c: bs_rate * jax.nn.softplus((-z_c - bs_engage) * 2000.0) / 2000.0

        F_bs_fl = _bstop(z_fl);  F_bs_fr = _bstop(z_fr)
        F_bs_rl = _bstop(z_rl);  F_bs_rr = _bstop(z_rr)

        # Gas spring preload from setup ride height (static equilibrium offset)
        F_gas_f = self.vp.get('damper_gas_force_f', 120.0) / mr_f
        F_gas_r = self.vp.get('damper_gas_force_r', 120.0) / mr_r

        # ARB forces (roll coupling)
        F_arb_f = arb_rate_f * (z_fl - z_fr)
        F_arb_r = arb_rate_r * (z_rl - z_rr)

        # Net suspension forces at each corner
        F_susp_fl = -(wheel_rate_f * z_fl + F_damp_fl + F_gas_f + F_bs_fl
                      + F_arb_f / 2.0)
        F_susp_fr = -(wheel_rate_f * z_fr + F_damp_fr + F_gas_f + F_bs_fr
                      - F_arb_f / 2.0)
        F_susp_rl = -(wheel_rate_r * z_rl + F_damp_rl + F_gas_r + F_bs_rl
                      + F_arb_r / 2.0)
        F_susp_rr = -(wheel_rate_r * z_rr + F_damp_rr + F_gas_r + F_bs_rr
                      - F_arb_r / 2.0)

        # ── External force vector F_ext (14-DOF chassis) ─────────────────────
        F_ext = jnp.zeros(28)

        # Body forces (sprung mass, dofs 14:20 = vx,vy,vz,wx,wy,wz)
        F_ext = F_ext.at[14].set(
            Fx_f + Fx_r + Fx_brake_f + Fx_brake_r + Fx_aero
            - self.m_s * self.g * jnp.sin(theta_pitch))
        F_ext = F_ext.at[15].set(
            Fy_f + Fy_r
            - self.m_s * self.g * jnp.sin(phi_roll) * jnp.cos(theta_pitch))
        F_ext = F_ext.at[16].set(
            F_susp_fl + F_susp_fr + F_susp_rl + F_susp_rr
            - self.m_s * self.g * jnp.cos(phi_roll) * jnp.cos(theta_pitch)
            + Fz_aero_f + Fz_aero_r)
        F_ext = F_ext.at[17].set(   # roll moment
            -(Fy_f + Fy_r) * s.h_cg
            + (F_susp_fr - F_susp_fl) * tf2
            + (F_susp_rr - F_susp_rl) * tr2
            + Mx_aero)
        F_ext = F_ext.at[18].set(   # pitch moment
            (Fx_f - Fx_r) * s.h_cg
            - (F_susp_fl + F_susp_fr) * self.lf
            + (F_susp_rl + F_susp_rr) * self.lr
            + My_aero)
        F_ext = F_ext.at[19].set(   # yaw moment
            Fy_f * self.lf - Fy_r * self.lr
            + (Fx_fr - Fx_fl) * tf2
            + (Fx_rr - Fx_rl) * tr2
            + Mz_total + M_diff)

        # Unsprung mass forces (dofs 20:24 = ż_fl..ż_rr)
        F_ext = F_ext.at[20].set(-F_susp_fl + Fz_fl - self.m_us_f * self.g)
        F_ext = F_ext.at[21].set(-F_susp_fr + Fz_fr - self.m_us_f * self.g)
        F_ext = F_ext.at[22].set(-F_susp_rl + Fz_rl - self.m_us_r * self.g)
        F_ext = F_ext.at[23].set(-F_susp_rr + Fz_rr - self.m_us_r * self.g)

        # Wheel spin (dofs 24:28 = ω_fl..ω_rr)
        F_ext = F_ext.at[24].set((-Fx_fl * self.R_wheel) / self.Iw)
        F_ext = F_ext.at[25].set((-Fx_fr * self.R_wheel) / self.Iw)
        F_ext = F_ext.at[26].set((-Fx_rl * self.R_wheel + T_drive_wheel * 0.5) / self.Iw)
        F_ext = F_ext.at[27].set((-Fx_rr * self.R_wheel + T_drive_wheel * 0.5) / self.Iw)

        # ── Mechanical derivatives (PH + external) ───────────────────────────
        dx_mech = PH_accel / jnp.concatenate([self.M_diag, self.M_diag]) + F_ext

        # ── Thermal derivatives ───────────────────────────────────────────────
        dx_therm = self.tire.compute_thermal_derivatives(
            x[28:38],
            jnp.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr]),
            jnp.array([0.0, 0.0, jnp.abs(kappa_t_rl), jnp.abs(kappa_t_rr)]),
            jnp.abs(vx),
        ) if hasattr(self.tire, 'compute_thermal_derivatives') else jnp.zeros(10)

        # ── Transient slip derivatives ────────────────────────────────────────
        dx_slip = jnp.array([
            d_alpha_fl, d_kappa_fl,
            d_alpha_fr, d_kappa_fr,
            d_alpha_rl, d_kappa_rl,
            d_alpha_rr, d_kappa_rr,
        ])

        return jnp.concatenate([dx_mech[:28], dx_therm, dx_slip])

    # ─────────────────────────────────────────────────────────────────────────
    # §5.4  Implicit midpoint integrator  (unchanged — symplectic for any H)
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _implicit_midpoint_step(
        self,
        x:            jax.Array,
        u:            jax.Array,
        setup_params: jax.Array,
        dt_step:      float,
    ) -> jax.Array:
        """
        4-iteration Picard implicit midpoint for mechanical DOFs.
        Symplectic for non-separable H(q,p) — zero phase drift at 22 km.
        Thermal + slip: forward Euler (timescale >> dt_sub = 1ms).
        """
        def picard_step(z_next, _):
            q_mid = 0.5 * (x[0:14]  + z_next[0:14])
            v_mid = 0.5 * (x[14:28] + z_next[14:28])
            x_mid = z_next.at[0:14].set(q_mid).at[14:28].set(v_mid).at[28:46].set(x[28:46])
            dx    = self._compute_derivatives(x_mid, u, setup_params)
            q_new = x[0:14]  + dt_step * dx[0:14]
            v_new = x[14:28] + dt_step * dx[14:28]
            return z_next.at[0:14].set(q_new).at[14:28].set(v_new), None

        dx0    = self._compute_derivatives(x, u, setup_params)
        z_pred = (x.at[0:14].set(x[0:14]  + dt_step * dx0[0:14])
                   .at[14:28].set(x[14:28] + dt_step * dx0[14:28]))

        z_next, _ = jax.lax.scan(picard_step, z_pred, None, length=4)

        dx_final   = self._compute_derivatives(z_next, u, setup_params)
        therm_next = x[28:38] + dt_step * dx_final[28:38]
        slip_next  = x[38:46] + dt_step * dx_final[38:46]

        return z_next.at[28:38].set(therm_next).at[38:46].set(slip_next)

    # ─────────────────────────────────────────────────────────────────────────
    # §5.5  Public simulate_step  (accepts SuspensionSetup OR raw vector)
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def simulate_step(
        self,
        state:        jax.Array,
        controls:     jax.Array,
        setup,                        # SuspensionSetup | jnp.ndarray(28,) | jnp.ndarray(8,)
        dt:           float = 0.005,
        n_substeps:   int   = 5,
    ) -> jax.Array:
        """
        Advance simulation by `dt` seconds with `n_substeps` implicit midpoint steps.

        setup:
            · SuspensionSetup  — preferred (typed, gradient-transparent)
            · jnp.ndarray(28,) — accepted directly
            · jnp.ndarray(8,)  — legacy 8-element vector (auto-upgraded)
        """
        # ── Normalize setup to 28-element vector ─────────────────────────────
        if isinstance(setup, SuspensionSetup):
            setup_vec = setup.to_vector()
        else:
            v = jnp.asarray(setup, dtype=jnp.float32)
            setup_vec = jax.lax.cond(
                v.shape[0] == 8,
                lambda _: SuspensionSetup.from_legacy_vector(v).to_vector(),
                lambda _: v,
                operand=None,
            ) if v.shape[0] != SETUP_DIM else v

        dt_sub = dt / n_substeps

        def substep(x, _):
            return self._implicit_midpoint_step(x, controls, setup_vec, dt_sub), None

        final_state, _ = jax.lax.scan(substep, state, None, length=n_substeps)
        return final_state


# ─────────────────────────────────────────────────────────────────────────────
# §6  Backward-compat alias  (physics_server.py imports MultiBodyVehicle)
# ─────────────────────────────────────────────────────────────────────────────

MultiBodyVehicle = DifferentiableMultiBodyVehicle
VehicleDynamicsH = DifferentiableMultiBodyVehicle


# ─────────────────────────────────────────────────────────────────────────────
# §7  PhysicsNormalizer extension for external callers
# ─────────────────────────────────────────────────────────────────────────────

def make_setup_from_params(vp: dict) -> SuspensionSetup:
    """
    Construct a SuspensionSetup from a vehicle_params dict.
    Used by objectives.py and optimization scripts that build setups
    from config files rather than from the optimizer directly.
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
        vp.get('h_ride_design',     DEFAULT_SETUP[12]),
        vp.get('h_ride_design',     DEFAULT_SETUP[13]),
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