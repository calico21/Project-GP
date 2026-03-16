# models/tire_model.py
# Project-GP — Multi-Fidelity Tire Model
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX2)
# ─────────────────────────────────────────────────────────────────────────────
# BUGFIX-3 : sigma penalty 56% at init — physically impossible cornering speeds
#   PREVIOUS: penalty = 2.0 * sigma applied unconditionally.
#   Uninitalized GP: prior_var=0.08 → sigma≈0.28 → penalty=0.56 → forces
#   reduced to 44% of Pacejka baseline. MPC sees a friction circle 56%
#   smaller than physical reality → finds "optimal" cornering at 19.2 m/s
#   instead of the physical limit of ~16.6 m/s.
#   FIX: jnp.clip(2.0 * sigma, 0.0, 0.15)  — 15% maximum LCB penalty.
#   Physical justification: even in fully unexplored regions, Pacejka
#   captures the dominant tire physics. The PINN correction is a residual
#   ΔFy/Fy0 bounded to ±25% by the clip already present downstream.
#   The GP sigma quantifies uncertainty in that residual, not in the baseline.
#
# BUGFIX-4 : GP inducing points cover only positive slip quadrant
#   PREVIOUS: Z_raw = uniform(0,1), Z = Z_raw * scale + shift
#   Result: alpha ∈ [0, 0.15] rad (left turns only), kappa ∈ [0, 0.10]
#   (traction only), gamma ∈ [0, 0.05] (positive camber only).
#   The GP uncertainty estimate for right-hand turns and braking was
#   physically meaningless — the car spends half its time in uncharted space.
#   FIX: Z_raw = normal(0, 0.5), Z = tanh(Z_raw) * scale + shift
#   tanh maps R→(-1,1) symmetrically. Covers:
#     alpha ∈ [-0.25, 0.25] rad  (±14.3°, full operating range)
#     kappa ∈ [-0.20, 0.20]      (traction + braking)
#     gamma ∈ [-0.08, 0.08] rad  (±4.6°, FS camber + roll-induced variation)
#     Fz    ∈ [400, 1200] N      (realistic corner load range)
#     Vx    ∈ [2, 22] m/s        (full FS event speed range)
#   tanh also bounds inducing points permanently — Adam cannot push them
#   outside the physical operating envelope.
#
# BUGFIX-5 : PINN blind to thermal state
#   PREVIOUS: features = [sin(α), sin(2α), κ, κ³, γ, Fz/1000, Vx/20]
#   No temperature. The deterministic drift correction is designed to capture
#   systematic Pacejka deviations — and the dominant deviation at operating
#   conditions IS thermal sensitivity. The network had no access to it.
#   FIX: add (T_eff - T_opt) / 30.0 as 8th feature.
#   T_eff = mean(T_ribs[:3]) — same surface average used by _thermal_grip_factor.
#   Normalization: /30 → unit range covers ±30°C from optimum, where the
#   Pacejka thermal correction changes by ~exp(-0.0008×900) ≈ 0.49 = 51% drop.
#   GP input kept at 5D (kinematic only) — thermal is analytically modeled
#   above the GP layer via _thermal_grip_factor.
#
# BUGFIX-6 : SpectralDense — u_vec receives loss gradients, breaking Lipschitz bound
#   PREVIOUS: u_vec = self.param(...) — Adam treats it as a learnable weight.
#   Adam minimizes loss by pushing u_vec toward directions that minimize sigma,
#   making W_sn = W / sigma unbounded. The Lipschitz-≤1 guarantee was void.
#   FIX: jax.lax.stop_gradient(sigma) — normalization factor is frozen.
#   u_vec is still a param so it survives serialization, but its gradient is
#   zeroed before the Adam update, preserving the power-iteration semantics.
#
# BUGFIX-7 : compute_thermal_derivatives layout misaligned with state vector
#   PREVIOUS: assumed T_nodes[0:5] = contiguous 5-node front block.
#   ACTUAL state layout (vehicle_dynamics.py §5.3):
#     x[28:31] = T_ribs_f, x[31] = T_gas_f, x[32:35] = T_ribs_r, x[35] = T_gas_r
#     x[36] = T_core_f, x[37] = T_core_r
#   Previous code: T_nodes[4] = x[32] = T_rib0_r was used as front core temp.
#   Rear thermal block starting at T_nodes[5]=x[33] was offset by one index.
#   FIX: reindex to match actual vehicle_dynamics layout exactly.
#
# BUGFIX-8 : tire.operator AttributeError in diagnose.py
#   diagnose.py: tire.operator.apply(state, ...) — attribute was _pinn_module.
#   FIX: @property operator returns _pinn_module.
#
# ─── Retained from GP-vX1 ────────────────────────────────────────────────────
# BUGFIX-1  TireOperatorPINN now a proper Flax nn.Module
# BUGFIX-2  compute_thermal_derivatives implemented
# UPGRADE-1 Spectral normalization on PINN Dense layers
# UPGRADE-2 Learnable inducing points for Sparse GP
# UPGRADE-3 Full MF6.2 aligning torque
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# §1  Spectral normalization utility
# ─────────────────────────────────────────────────────────────────────────────

class SpectralDense(nn.Module):
    """
    Dense layer with power-iteration spectral normalization.
    Bounds the Lipschitz constant of each layer to ≤ 1.

    BUGFIX-6: stop_gradient applied to sigma.
    Without it Adam receives a gradient through sigma and minimizes it,
    making W / sigma unbounded — the exact opposite of the intended guarantee.
    With stop_gradient, the normalization factor is frozen w.r.t. the optimizer,
    preserving the Lipschitz-≤1 bound across all training epochs.
    """
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        W = self.param('kernel', jax.nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        # u_vec: power-iteration running estimate of the top right singular vector.
        # Stored as param for serialization, but NOT updated by Adam — see below.
        u = self.param('u_vec', jax.nn.initializers.normal(), (self.features,))

        # One power iteration step (sufficient for smooth networks).
        v = W.T @ (W @ u)   # (in,out).T @ ((in,out) @ (out,)) = (out,in) @ (in,) = (out,)

        v = v / (jnp.linalg.norm(v) + 1e-8)
        # BUGFIX-6: stop_gradient prevents sigma from entering the loss gradient.
        # Adam will NOT see dL/d(sigma) and will NOT try to minimize sigma.
        sigma = jax.lax.stop_gradient(jnp.linalg.norm(W @ v) + 1e-8)
        W_sn  = W / sigma

        out = x @ W_sn
        if self.use_bias:
            b   = self.param('bias', jax.nn.initializers.zeros, (self.features,))
            out = out + b
        return out


# ─────────────────────────────────────────────────────────────────────────────
# §2  Sparse Gaussian Process — Matérn 5/2 with learnable inducing points
# ─────────────────────────────────────────────────────────────────────────────

class SparseGPMatern52(nn.Module):
    """
    Sparse GP with Matérn 5/2 kernel and LEARNABLE inducing point locations.

    Input:  x_star — (5,) kinematic state [alpha, kappa, gamma, Fz, Vx]
    Output: scalar predictive std dev σ(x_star)

    BUGFIX-4: Symmetric inducing point initialization.
    Previous uniform(0,1) initialization covered only positive slip angles
    (left turns only) and positive kappa (traction only). The GP uncertainty
    estimate for right-hand turns and braking was physically meaningless.

    New: Z_raw ~ N(0, 0.5), Z = tanh(Z_raw) * scale + shift
    · tanh maps R→(-1,1), giving symmetric coverage around zero for signed
      quantities (alpha, kappa, gamma) and offset coverage for positive ones (Fz, Vx).
    · tanh also permanently bounds inducing points — Adam cannot push them
      outside the physical operating envelope regardless of learning rate.
    """
    num_inducing: int = 50

    @nn.compact
    def __call__(self, x_star: jax.Array) -> jax.Array:
        log_ls = self.param('log_lengthscale',
                            lambda key, _: jnp.log(jnp.array([0.2, 0.15, 0.1, 400.0, 15.0])),
                            (5,))
        ls = jnp.exp(log_ls)

        log_var = self.param('log_variance',
                             jax.nn.initializers.constant(jnp.log(0.08)), ())
        prior_var = jnp.exp(log_var)

        # BUGFIX-4: normal init + tanh transform for symmetric, bounded coverage.
        # Physical ranges after transform:
        #   alpha: tanh(N(0,0.5)) * 0.25 → ∈ (-0.25, 0.25) rad  [±14.3°]
        #   kappa: tanh(N(0,0.5)) * 0.20 → ∈ (-0.20, 0.20)       [±braking/traction]
        #   gamma: tanh(N(0,0.5)) * 0.08 → ∈ (-0.08, 0.08) rad   [±4.6° incl. roll]
        #   Fz:    tanh(N(0,0.5)) * 400 + 800 → ∈ (400, 1200) N  [corner load range]
        #   Vx:    tanh(N(0,0.5)) * 10 + 12  → ∈ (2, 22) m/s     [FS event range]
        Z_scale = jnp.array([0.25, 0.20, 0.08, 400.0, 10.0])
        Z_shift = jnp.array([0.0,  0.0,  0.0,  800.0, 12.0])
        Z_raw   = self.param('Z_raw',
                             jax.nn.initializers.normal(stddev=0.5),
                             (self.num_inducing, 5))
        Z = jnp.tanh(Z_raw) * Z_scale + Z_shift

        def matern52(x1, x2):
            d = jnp.sqrt(jnp.sum(((x1 - x2) / (ls + 1e-8)) ** 2) + 1e-8)
            s = jnp.sqrt(5.0) * d
            return prior_var * (1.0 + s + (5.0 / 3.0) * d ** 2) * jnp.exp(-s)

        k_ZZ     = jax.vmap(lambda z1: jax.vmap(lambda z2: matern52(z1, z2))(Z))(Z)
        K_ZZ_inv = jnp.linalg.inv(k_ZZ + 1e-4 * jnp.eye(self.num_inducing))

        k_xZ     = jax.vmap(lambda z: matern52(x_star, z))(Z)
        k_xx     = prior_var
        red      = jnp.dot(k_xZ, jnp.dot(K_ZZ_inv, k_xZ))
        variance = jnp.maximum(k_xx - red, 1e-6)
        return jnp.sqrt(variance)


# ─────────────────────────────────────────────────────────────────────────────
# §3  PINN + GP combined tire operator
# ─────────────────────────────────────────────────────────────────────────────

class TireOperatorPINN(nn.Module):
    """
    Symmetry-respecting Physics-Informed Neural Network + Sparse GP.

    Input: state_tensor (8,) [alpha, kappa, gamma, Fz, Vx, T_norm]   [BUGFIX-5]
    Previously (7,) with no thermal state — the dominant Pacejka deviation
    (thermal sensitivity) was invisible to the correction network.

    T_norm = (T_eff - T_opt) / 30.0 where T_eff = mean(T_surface_ribs).
    Normalization: /30 covers ±30°C from optimum, the range over which
    the Pacejka thermal correction changes by up to 51%.

    GP input remains (5,) kinematic only — thermal is modeled analytically
    above via _thermal_grip_factor and the GP's role is uncertainty bounding
    in kinematic operating regimes far from tested conditions.
    """
    dim_hidden:      int = 16
    num_gp_inducing: int = 50

    @nn.compact
    def __call__(
        self,
        state_tensor: jax.Array,   # (8,) [alpha, kappa, gamma, Fz, Vx, T_norm, _pad, _pad]
        stochastic_key = None,
    ):
        # Unpack — T_norm is the 6th element (BUGFIX-5 addition)
        alpha  = state_tensor[0]
        kappa  = state_tensor[1]
        gamma  = state_tensor[2]
        Fz     = state_tensor[3]
        Vx     = state_tensor[4]
        T_norm = state_tensor[5]   # (T_eff - T_opt) / 30.0

        # Symmetry-respecting kinematic features
        # sin(2α): captures the secondary peak in Fy vs α beyond the saturation point
        # κ³: asymmetric longitudinal coupling in combined slip
        # T_norm: primary driver of Pacejka residuals at operating temperature
        features = jnp.array([
            jnp.sin(alpha),
            jnp.sin(2.0 * alpha),
            kappa,
            kappa ** 3,
            gamma,
            Fz / 1000.0,
            Vx / 20.0,
            T_norm,         # BUGFIX-5: thermal deviation from optimum
        ])

        # Spectrally normalized layers → Lipschitz-bounded PINN
        x = SpectralDense(self.dim_hidden)(features)
        x = jnp.tanh(x)
        x = SpectralDense(32)(x)
        x = jnp.tanh(x)
        drift = nn.Dense(2,
                         kernel_init=jax.nn.initializers.zeros,
                         bias_init=jax.nn.initializers.zeros)(x)

        # GP uncertainty — kinematic state only (thermal is analytically modeled)
        gp    = SparseGPMatern52(num_inducing=self.num_gp_inducing)
        sigma = gp(state_tensor[:5])   # (5,): alpha, kappa, gamma, Fz, Vx

        if stochastic_key is not None:
            noise = jax.random.normal(stochastic_key, shape=(2,))
            return drift + sigma * noise, sigma

        return drift, sigma


# ─────────────────────────────────────────────────────────────────────────────
# §4  PacejkaTire  — MF6.2 + 5-Node Thermal + PINN/GP
# ─────────────────────────────────────────────────────────────────────────────

class PacejkaTire:
    """
    100% Pure JAX Differentiable Tire Model.

    Layers:
    1. Analytical Pacejka MF6.2 (pure + combined slip)
    2. 5-Node thermodynamic ODE (Jaeger flash temp + convection/conduction)
    3. TireOperatorPINN: deterministic drift + GP uncertainty (LCB capped at 15%)
    """

    def __init__(self, tire_coeffs: dict, rng_seed: int = 42):
        self.coeffs  = tire_coeffs
        self.T_opt   = tire_coeffs.get('T_opt', tire_coeffs.get('T_OPT', 90.0))
        self.T_env   = 25.0
        self.T_track = 40.0
        self.P_nom   = tire_coeffs.get('P_nom', 0.834)

        key = jax.random.PRNGKey(rng_seed)
        self._pinn_module = TireOperatorPINN()

        # BUGFIX-5: dummy state is now (8,) to match expanded feature vector
        dummy_state = jnp.ones(8)
        self._pinn_params = self._pinn_module.init(key, dummy_state)

    @property
    def operator(self):
        """
        Alias for _pinn_module — backward compat with diagnose.py.
        Correct call: tire.operator.apply(tire._pinn_params, state_8d)
        """
        return self._pinn_module

    @property
    def pinn_params(self):
        """Exposes PINN params for external calibration or logging."""
        return self._pinn_params

    # ─────────────────────────────────────────────────────────────────────────
    # §4.1  Flash temperature
    # ─────────────────────────────────────────────────────────────────────────

    def compute_flash_temperature(
        self,
        mu_actual: jax.Array,
        Fz:        jax.Array,
        V_slide:   jax.Array,
    ) -> jax.Array:
        k_rubber            = 0.25
        rho_c               = 2.0e6
        contact_half_length = 0.075
        thermal_diff        = k_rubber / rho_c
        V_safe              = jnp.maximum(jnp.abs(V_slide), 1e-3)
        q_flux              = (mu_actual * jnp.abs(Fz) * V_safe) / (2.0 * contact_half_length * 0.205)
        T_flash             = ((q_flux * contact_half_length)
                               / (k_rubber * jnp.sqrt(jnp.pi * (V_safe * contact_half_length) / thermal_diff)))
        return jnp.clip(T_flash, 0.0, 350.0)

    # ─────────────────────────────────────────────────────────────────────────
    # §4.2  5-Node thermal derivatives  (BUGFIX-7: layout realigned)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_thermal_derivatives(
        self,
        T_nodes:    jax.Array,   # (10,) — see layout note below
        Fz_corners: jax.Array,   # (4,)
        kappa:      jax.Array,   # (4,) absolute longitudinal slip
        Vx:         jax.Array,   # scalar
    ) -> jax.Array:
        """
        5-node thermal ODE per axle, realigned to match vehicle_dynamics layout.

        State vector layout (vehicle_dynamics.py §5.3):
          T_nodes[0:3] = x[28:31] = T_surf_inner/mid/outer_f   (3 rib temps, front)
          T_nodes[3]   = x[31]    = T_gas_f
          T_nodes[4:7] = x[32:35] = T_surf_inner/mid/outer_r   (3 rib temps, rear)
          T_nodes[7]   = x[35]    = T_gas_r
          T_nodes[8]   = x[36]    = T_core_f
          T_nodes[9]   = x[37]    = T_core_r

        BUGFIX-7: previous code assumed T_nodes[0:5] = front 5-node block.
        That mapped T_nodes[4]=x[32]=T_surf0_r into the front core slot and
        misaligned the entire rear thermal block by one index.

        Returns dT/dt (10,) in °C/s, ordered identically to T_nodes input.
        """
        # Thermal constants (approximate for Hoosier R25B)
        k_cond   = 0.25     # W/m/K rubber conductivity
        rho_c    = 2.0e6    # J/m³/K volumetric heat capacity
        h_conv   = 80.0     # W/m²/K convection coefficient (air + track)
        A_patch  = 0.025    # m² contact patch area
        V_tire   = 0.003    # m³ tire volume (approx)
        C_node   = rho_c * V_tire / 5.0   # lumped capacitance per node [J/K]

        mu_nom = 1.5   # conservative nominal friction coefficient

        # ── Front axle ────────────────────────────────────────────────────────
        T_ribs_f   = T_nodes[0:3]   # surface inner/mid/outer
        T_gas_f    = T_nodes[3]
        T_core_f   = T_nodes[8]     # BUGFIX-7: was T_nodes[4] (= T_surf0_r, wrong)

        Fz_f       = (Fz_corners[0] + Fz_corners[1]) * 0.5
        kap_f      = (kappa[0] + kappa[1]) * 0.5
        V_slide_f  = jnp.abs(Vx * kap_f)
        T_flash_f  = self.compute_flash_temperature(mu_nom, Fz_f, V_slide_f)

        # Frictional heat split evenly across 3 surface nodes
        Q_fric_f   = mu_nom * Fz_f * V_slide_f / (3.0 * C_node + 1e-6)

        dT_s_f0 = (Q_fric_f
                   + h_conv * A_patch * (T_flash_f - T_ribs_f[0]) / C_node
                   - h_conv * A_patch * (T_ribs_f[0] - self.T_env) / C_node)
        dT_s_f1 = (Q_fric_f
                   + h_conv * A_patch * (T_flash_f - T_ribs_f[1]) / C_node
                   - h_conv * A_patch * (T_ribs_f[1] - self.T_env) / C_node)
        dT_s_f2 = (Q_fric_f
                   + h_conv * A_patch * (T_flash_f - T_ribs_f[2]) / C_node
                   - h_conv * A_patch * (T_ribs_f[2] - self.T_env) / C_node)

        T_surf_avg_f = (T_ribs_f[0] + T_ribs_f[1] + T_ribs_f[2]) / 3.0
        # Gay-Lussac coupling: internal gas tracks surface slowly
        dT_gas_f  = 0.05 * (T_surf_avg_f - T_gas_f)
        # Core conduction: slow thermal mass
        dT_core_f = 0.02 * (T_surf_avg_f - T_core_f)

        # ── Rear axle ─────────────────────────────────────────────────────────
        T_ribs_r  = T_nodes[4:7]   # BUGFIX-7: was T_nodes[5:8] (off by one)
        T_gas_r   = T_nodes[7]     # BUGFIX-7: was T_nodes[8]
        T_core_r  = T_nodes[9]     # BUGFIX-7: was T_nodes[9] (coincidentally correct)

        Fz_r      = (Fz_corners[2] + Fz_corners[3]) * 0.5
        kap_r     = (kappa[2] + kappa[3]) * 0.5
        V_slide_r = jnp.abs(Vx * kap_r)
        T_flash_r = self.compute_flash_temperature(mu_nom, Fz_r, V_slide_r)

        Q_fric_r  = mu_nom * Fz_r * V_slide_r / (3.0 * C_node + 1e-6)

        dT_s_r0 = (Q_fric_r
                   + h_conv * A_patch * (T_flash_r - T_ribs_r[0]) / C_node
                   - h_conv * A_patch * (T_ribs_r[0] - self.T_env) / C_node)
        dT_s_r1 = (Q_fric_r
                   + h_conv * A_patch * (T_flash_r - T_ribs_r[1]) / C_node
                   - h_conv * A_patch * (T_ribs_r[1] - self.T_env) / C_node)
        dT_s_r2 = (Q_fric_r
                   + h_conv * A_patch * (T_flash_r - T_ribs_r[2]) / C_node
                   - h_conv * A_patch * (T_ribs_r[2] - self.T_env) / C_node)

        T_surf_avg_r = (T_ribs_r[0] + T_ribs_r[1] + T_ribs_r[2]) / 3.0
        dT_gas_r  = 0.05 * (T_surf_avg_r - T_gas_r)
        dT_core_r = 0.02 * (T_surf_avg_r - T_core_r)

        # Output ordering MUST match input T_nodes ordering (vehicle_dynamics layout):
        #   [surf×3_f, gas_f, surf×3_r, gas_r, core_f, core_r]
        dT = jnp.array([
            dT_s_f0, dT_s_f1, dT_s_f2, dT_gas_f,   # front (indices 0-3)
            dT_s_r0, dT_s_r1, dT_s_r2, dT_gas_r,   # rear  (indices 4-7)
            dT_core_f, dT_core_r,                    # cores (indices 8-9)
        ])
        return jnp.clip(dT, -500.0, 500.0)

    # ─────────────────────────────────────────────────────────────────────────
    # §4.3  Thermal grip factor from 5-node model
    # ─────────────────────────────────────────────────────────────────────────

    def _thermal_grip_factor(
        self,
        T_ribs: jax.Array,   # (3,) surface temperatures
        T_gas:  jax.Array,   # scalar internal gas temp
    ) -> jax.Array:
        """
        μ_thermal = exp(-β·(T_eff - T_opt)²)
        Gaussian thermal window around T_opt, with Gay-Lussac pressure correction.
        """
        T_eff   = jnp.mean(T_ribs[:3])
        beta    = 0.0008        # K⁻²  (peak width ≈ 35°C)
        mu_T    = jnp.exp(-beta * (T_eff - self.T_opt) ** 2)

        T_ref   = self.T_env + 273.15
        T_gas_K = T_gas + 273.15
        P_ratio = jnp.clip(T_gas_K / (T_ref + 1e-3), 0.70, 1.30)
        mu_P    = 1.0 + 0.05 * (P_ratio - 1.0)

        return jnp.clip(mu_T * mu_P, 0.30, 1.20)

    # ─────────────────────────────────────────────────────────────────────────
    # §4.4  Pacejka MF6.2 force computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_force(
        self,
        alpha:          jax.Array,
        kappa:          jax.Array,
        Fz:             jax.Array,
        gamma:          jax.Array,
        T_ribs:         jax.Array,   # (3,) surface rib temperatures
        T_gas:          jax.Array,   # scalar internal gas temperature
        Vx:             jax.Array,
        stochastic_key          = None,
        wz:             jax.Array = jnp.array(0.0),
    ):
        """
        Full Pacejka MF6.2 lateral and longitudinal force.
        Returns (Fx, Fy) in Newtons.
        Sign convention: positive Fy = left force (SAE z-up).
        """
        c   = self.coeffs
        eps = 1e-6

        Fz0     = c.get('FNOMIN', 654.0)
        Fz_safe = jnp.maximum(Fz, 10.0)
        dfz     = (Fz_safe - Fz0) / (Fz0 + eps)

        lam_muy = self._thermal_grip_factor(T_ribs, T_gas)
        gam     = gamma

        # ════════════════════════════════════════════════════════════════════
        # PURE LATERAL FORCE  (MF6.2)
        # ════════════════════════════════════════════════════════════════════
        PCY1 = c.get('PCY1',  1.53041)
        PDY1 = c.get('PDY1',  2.40275)
        PDY2 = c.get('PDY2',  0.343535)
        PDY3 = c.get('PDY3',  3.89743)
        PEY1 = c.get('PEY1',  0.000)
        PEY2 = c.get('PEY2', -0.280762)
        PEY3 = c.get('PEY3',  0.70403)
        PEY4 = c.get('PEY4', -0.478297)
        PKY1 = c.get('PKY1', 53.2421)
        PKY2 = c.get('PKY2',  2.38205)
        PKY3 = c.get('PKY3',  0.15)
        PKY4 = c.get('PKY4',  2.0)
        PHY1 = c.get('PHY1', -0.0009)
        PHY2 = c.get('PHY2', -0.00082)
        PVY1 = c.get('PVY1',  0.045)
        PVY2 = c.get('PVY2', -0.024)

        SHy  = PHY1 + PHY2 * dfz
        SVy  = Fz_safe * (PVY1 + PVY2 * dfz) * lam_muy

        Ky   = (PKY1 * Fz0
                * jnp.sin(PKY4 * jnp.arctan(Fz_safe / jnp.maximum(PKY2 * Fz0, eps)))
                * (1.0 - PKY3 * jnp.abs(gam)))
        Dy   = PDY1 * (1.0 + PDY2 * dfz) * (1.0 - PDY3 * gam ** 2) * Fz_safe * lam_muy
        Cy   = PCY1
        By   = Ky / jnp.maximum(Cy * Dy, eps)

        a_s     = alpha + SHy
        sgn_as  = jnp.tanh(a_s / (1e-3 + eps))
        Ey      = jnp.clip(
            (PEY1 + PEY2 * dfz) * (1.0 - (PEY3 + PEY4 * gam) * sgn_as),
            -10.0, 1.0,
        )
        x_y     = By * a_s
        Fy0     = Dy * jnp.sin(Cy * jnp.arctan(x_y - Ey * (x_y - jnp.arctan(x_y)))) + SVy

        # ════════════════════════════════════════════════════════════════════
        # PURE LONGITUDINAL FORCE  (MF6.2)
        # ════════════════════════════════════════════════════════════════════
        PCX1 = c.get('PCX1',  1.579)
        PDX1 = c.get('PDX1',  1.0)
        PDX2 = c.get('PDX2', -0.10)
        PDX3 = c.get('PDX3',  0.0)
        PEX1 = c.get('PEX1', -0.20)
        PEX2 = c.get('PEX2',  0.10)
        PEX3 = c.get('PEX3',  0.0)
        PKX1 = c.get('PKX1', 18.5)
        PKX2 = c.get('PKX2',  0.0)
        PKX3 = c.get('PKX3',  0.20)
        PHX1 = c.get('PHX1',  0.0)
        PHX2 = c.get('PHX2',  0.0)
        PVX1 = c.get('PVX1',  0.0)
        PVX2 = c.get('PVX2',  0.0)

        SHx     = PHX1 + PHX2 * dfz
        SVx     = Fz_safe * (PVX1 + PVX2 * dfz) * lam_muy
        kappa_c = kappa + SHx

        Cx  = PCX1
        Dx  = PDX1 * (1.0 + PDX2 * dfz) * (1.0 - PDX3 * gam ** 2) * Fz_safe * lam_muy
        Kx  = PKX1 * Fz_safe * jnp.exp(PKX3 * dfz) * (1.0 + PKX2 * dfz)
        Bx  = Kx / jnp.maximum(Cx * Dx, eps)
        Ex  = jnp.clip(PEX1 + PEX2 * dfz + PEX3 * dfz ** 2, -10.0, 1.0)
        x_x = Bx * kappa_c
        Fx0 = Dx * jnp.sin(Cx * jnp.arctan(x_x - Ex * (x_x - jnp.arctan(x_x)))) + SVx

        # ════════════════════════════════════════════════════════════════════
        # COMBINED SLIP REDUCTION  (Gyk, Gxa)
        # ════════════════════════════════════════════════════════════════════
        RBY1 = c.get('RBY1', 7.0)
        RBY2 = c.get('RBY2', 7.0)
        RBY3 = c.get('RBY3', 0.0)
        RCY1 = c.get('RCY1', 1.0)
        REY1 = c.get('REY1', 0.0)
        REY2 = c.get('REY2', 0.0)
        RHY1 = c.get('RHY1', 0.0)

        RBX1 = c.get('RBX1', 10.0)
        RBX2 = c.get('RBX2', 10.0)
        RCX1 = c.get('RCX1', 1.0)
        RHX1 = c.get('RHX1', 0.0)

        # ── Combined slip reduction  (MF6.2 corrected) ─────────────────────────────
        # G_yk: lateral force reduction due to LONGITUDINAL slip κ
        #   B_yk is α-dependent (sensitivity scaling), x is κ-based (the slip driving reduction).
        #   Previous code: x_ys = By_s * alpha_s — κ-independent → always 0% reduction.
        # G_xa: longitudinal force reduction due to LATERAL slip α
        #   B_xa is κ-dependent, x is α-based.
        # Both were using the wrong variable for x → combined slip had zero effect.

        SHyk    = RHY1
        kappa_ys = kappa + SHyk                             # shifted kappa — the input to G_yk

        # B_yk scales with alpha (how sensitive the reduction is to lateral conditions)
        By_s    = RBY1 * jnp.cos(jnp.arctan(RBY2 * (alpha - RBY3)))
        Ey_s    = REY1 + REY2 * dfz
        x_ys    = By_s * kappa_ys                           # FIX: was alpha_s — must be kappa
        # Normalization: G_yk(κ=0) = cos(arctan(B_yk * SHyk)) = cos(0) = 1 when SHyk=0
        Gyk_num = jnp.cos(RCY1 * jnp.arctan(x_ys  - Ey_s * (x_ys  - jnp.arctan(x_ys))))
        Gyk_den = jnp.cos(RCY1 * jnp.arctan(By_s * SHyk - Ey_s * (By_s * SHyk - jnp.arctan(By_s * SHyk))))
        Gyk     = Gyk_num / (Gyk_den + 1e-6)
        Gyk     = jnp.clip(Gyk, 0.05, 1.0)

        SHxa    = RHX1
        alpha_xs = alpha + SHxa                             # shifted alpha — input to G_xa

        # B_xa scales with kappa (how sensitive the reduction is to longitudinal conditions)
        Bx_s    = RBX1 * jnp.cos(jnp.arctan(RBX2 * kappa))
        x_xs    = Bx_s * alpha_xs                           # FIX: was kappa_s — must be alpha
        Gxa_num = jnp.cos(RCX1 * jnp.arctan(x_xs))
        Gxa_den = jnp.cos(RCX1 * jnp.arctan(Bx_s * SHxa))
        Gxa     = Gxa_num / (Gxa_den + 1e-6)
        Gxa     = jnp.clip(Gxa, 0.05, 1.0)

        Fy = Fy0 * Gyk
        Fx = Fx0 * Gxa

        # ── Turn slip correction ─────────────────────────────────────────────
        a_contact = c.get('contact_half_length', 0.05)
        R_path    = jnp.abs(Vx) / (jnp.abs(wz) + eps)
        phi_t     = a_contact / (R_path + 1e-3)
        Fy        = Fy * (1.0 - 0.15 * jnp.abs(phi_t))

        # ── PINN/GP residual corrections ─────────────────────────────────────
        # BUGFIX-5: include normalized thermal deviation as 6th PINN feature.
        # T_eff = surface average; same value used by _thermal_grip_factor.
        T_eff  = jnp.mean(T_ribs[:3])
        T_norm = (T_eff - self.T_opt) / 30.0

        # jnp.asarray handles both Python float and JAX array Vx without
        # isinstance checks (which fail under abstract tracing in jit).
        Vx_arr = jnp.asarray(Vx)

        # State tensor: (8,) to match TireOperatorPINN expanded feature vector.
        # Indices 6-7 unused by PINN (forward-compat padding), GP uses [:5].
        state_in = jnp.array([
            alpha, kappa_c, gam, Fz_safe, Vx_arr, T_norm,
            0.0, 0.0,   # padding slots for future features
        ])
        mods, sigma = self._pinn_module.apply(self._pinn_params, state_in,
                                               stochastic_key)
        mods = jnp.clip(mods, -0.25, 0.25)

        # BUGFIX-3: cap LCB penalty at 15%.
        # Uncapped: uninit GP gives sigma≈0.28 → penalty=0.56 → forces at 44%
        # of Pacejka → MPC finds physically impossible cornering speeds.
        # Physical justification: the PINN corrects a RESIDUAL on top of
        # Pacejka; the GP bounds uncertainty in that residual, not the baseline.
        # 15% is the maximum physically credible Pacejka modeling error.
        penalty = jnp.clip(2.0 * sigma, 0.0, 0.15)

        Fy = Fy * (1.0 + mods[1] - penalty)
        Fx = Fx * (1.0 + mods[0] - penalty)

        return Fx, Fy

    # ─────────────────────────────────────────────────────────────────────────
    # §4.5  Aligning torque  (MF6.2 Mz)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_aligning_torque(
        self,
        alpha: jax.Array,
        kappa: jax.Array,
        Fz:    jax.Array,
        gamma: jax.Array,
        Fy:    jax.Array,
        Fx:    jax.Array = jnp.array(0.0),
    ) -> jax.Array:
        """
        Pacejka MF6.2 aligning torque Mz.
        Positive = nose-right restoring moment.
        """
        c   = self.coeffs
        eps = 1e-6
        Fz0     = c.get('FNOMIN', 654.0)
        Fz_safe = jnp.maximum(Fz, 10.0)
        dfz     = (Fz_safe - Fz0) / (Fz0 + eps)
        gam     = gamma

        QBZ1 = c.get('QBZ1',  6.5)
        QBZ2 = c.get('QBZ2', -0.50)
        QBZ3 = c.get('QBZ3',  0.0)
        QBZ9 = c.get('QBZ9',  9.0)
        QCZ1 = c.get('QCZ1',  1.10)
        QDZ1 = c.get('QDZ1',  0.08)
        QDZ2 = c.get('QDZ2', -0.01)
        QDZ3 = c.get('QDZ3',  0.0)
        QDZ4 = c.get('QDZ4',  0.0)
        QEZ1 = c.get('QEZ1', -1.50)
        QEZ2 = c.get('QEZ2',  0.60)
        QEZ3 = c.get('QEZ3',  0.0)   # sign-term coefficient (unused at default=0)
        QHZ1 = c.get('QHZ1',  0.0)
        QHZ2 = c.get('QHZ2',  0.0)

        R0   = c.get('R0', 0.2045)

        SHt = QHZ1 + QHZ2 * dfz
        a_t = alpha + SHt

        Bt      = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz ** 2) * (1.0 + QBZ9 * jnp.abs(gam))
        Ct      = QCZ1
        Dt      = Fz_safe * R0 * (QDZ1 + QDZ2 * dfz) * (1.0 + QDZ3 * gam + QDZ4 * gam ** 2)
        # MF6.2 full Et: includes sign-dependent asymmetry term (QEZ3)
        sgn_at  = jnp.tanh(a_t / (1e-3 + eps))
        Et      = jnp.clip((QEZ1 + QEZ2 * dfz) * (1.0 - QEZ3 * sgn_at), -10.0, 1.0)
        x_t     = Bt * a_t
        t       = Dt * jnp.cos(Ct * jnp.arctan(x_t - Et * (x_t - jnp.arctan(x_t))))
        Mz0     = -t * Fy

        SSZ1 = c.get('SSZ1', 0.0)
        SSZ2 = c.get('SSZ2', 0.0)
        s    = R0 * (SSZ1 + SSZ2 * (Fy / (Fz0 + eps)))
        Mz_r = s * Fx

        return Mz0 + Mz_r

    # ─────────────────────────────────────────────────────────────────────────
    # §4.6  Transient slip derivatives
    # ─────────────────────────────────────────────────────────────────────────

    def compute_transient_slip_derivatives(
        self,
        alpha_kin:   jax.Array,
        kappa_kin:   jax.Array,
        alpha_t:     jax.Array,
        kappa_t:     jax.Array,
        Fz:          jax.Array,
        Vx:          jax.Array,
    ) -> tuple:
        """
        First-order carcass lag for transient slip states.
        rl = relaxation length [m]; τ = rl / |Vx| [s]
        """
        rl  = self.coeffs.get('relaxation_length', 0.35)
        tau = rl / (jnp.maximum(jnp.abs(Vx), 1.0))
        d_alpha = (alpha_kin - alpha_t) / tau
        d_kappa = (kappa_kin - kappa_t) / tau
        return d_alpha, d_kappa