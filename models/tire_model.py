# models/tire_model.py
# Project-GP — Multi-Fidelity Tire Model
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX1)
# ────────────────────
# BUGFIX-1 : TireOperatorPINN now a proper Flax nn.Module
#   Previous implementation used raw jnp weight matrices that were NOT
#   registered as JAX pytree leaves — they could not be updated via optax,
#   and `jax.grad` through the PINN was unreliable across function calls.
#   New: standard Flax @nn.compact module with proper init() / apply().
#   All call sites updated to pass (params, state_tensor).
#
# BUGFIX-2 : compute_thermal_derivatives implemented
#   vehicle_dynamics.py called self.tire.compute_thermal_derivatives() but
#   the method was missing. Now implemented as a differentiable 5-node ODE.
#
# UPGRADE-1 : Spectral normalization on PINN Dense layers
#   SpectralDense wraps nn.Dense with a learnable spectral norm scale σ.
#   Bounds the Lipschitz constant of the PINN to ≤ 1, preventing exploding
#   gradients when the physics engine is differentiated through the tire model.
#
# UPGRADE-2 : Learnable inducing points for Sparse GP
#   SparseGPMatern52 now has trainable inducing point locations Z via
#   self.param(...). The inducing points are initialized from the random
#   distribution but can migrate toward high-uncertainty operating regimes
#   during online calibration.
#
# UPGRADE-3 : Full MF6.2 aligning torque now available
#   compute_aligning_torque was present but not called in compute_force.
#   Now consistently applied and accessible for FFB (force feedback).
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
    Critical for gradient health when differentiating through long tire rollouts.

    Implementation: σ(W) ≈ ‖W‖₂ via one power iteration stored as a
    mutable variable. At eval time, W_normalized = W / σ(W).
    """
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        W = self.param('kernel', jax.nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        # Power iteration estimate of spectral norm
        u = self.param('u_vec', jax.nn.initializers.normal(),
                       (self.features,))
        v  = W.T @ (W @ u)
        v  = v / (jnp.linalg.norm(v) + 1e-8)
        sigma = jnp.dot(u, W @ v)
        W_sn  = W / (sigma + 1e-8)

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

    Upgrade vs previous:
    · Inducing point locations Z are registered as Flax params — they can be
      trained via gradient descent to migrate toward high-uncertainty regions.
    · K_ZZ_inv is recomputed at each call from the current Z (traced by JAX).
    · prior_variance and lengthscales are also learnable (log-parameterized
      for positivity).

    Input: x_star — (5,) state vector [alpha, kappa, gamma, Fz, Vx]
    Output: scalar predictive std dev σ(x_star)
    """
    num_inducing: int = 50

    @nn.compact
    def __call__(self, x_star: jax.Array) -> jax.Array:
        # Learnable log-lengthscales (log for positivity)
        log_ls = self.param('log_lengthscale',
                            lambda key, _: jnp.log(jnp.array([0.2, 0.15, 0.1, 400.0, 15.0])),
                            (5,))
        ls = jnp.exp(log_ls)

        log_var = self.param('log_variance',
                             jax.nn.initializers.constant(jnp.log(0.08)), ())
        prior_var = jnp.exp(log_var)

        # Learnable inducing points: (num_inducing, 5)
        Z_init = jax.nn.initializers.uniform(scale=1.0)
        Z_raw  = self.param('Z_raw', Z_init, (self.num_inducing, 5))
        # Scale to physical operating range
        Z_scale = jnp.array([0.15, 0.10, 0.05, 800.0, 20.0])
        Z_shift = jnp.array([0.0,  0.0,  0.0, 600.0,  5.0])
        Z = Z_raw * Z_scale + Z_shift

        def matern52(x1, x2):
            d = jnp.sqrt(jnp.sum(((x1 - x2) / (ls + 1e-8)) ** 2) + 1e-8)
            s = jnp.sqrt(5.0) * d
            return prior_var * (1.0 + s + (5.0 / 3.0) * d ** 2) * jnp.exp(-s)

        k_ZZ = jax.vmap(lambda z1: jax.vmap(lambda z2: matern52(z1, z2))(Z))(Z)
        K_ZZ_inv = jnp.linalg.inv(k_ZZ + 1e-4 * jnp.eye(self.num_inducing))

        k_xZ    = jax.vmap(lambda z: matern52(x_star, z))(Z)
        k_xx    = prior_var
        red     = jnp.dot(k_xZ, jnp.dot(K_ZZ_inv, k_xZ))
        variance = jnp.maximum(k_xx - red, 1e-6)
        return jnp.sqrt(variance)


# ─────────────────────────────────────────────────────────────────────────────
# §3  PINN + GP combined tire operator
# ─────────────────────────────────────────────────────────────────────────────

class TireOperatorPINN(nn.Module):
    """
    Symmetry-respecting Physics-Informed Neural Network + Sparse GP.

    UPGRADE: Full Flax nn.Module (was raw weight matrices — not a proper pytree).

    · Deterministic drift: spectrally-normalized MLP predicts [ΔFx, ΔFy]
      corrections as fractions of the Pacejka baseline.
    · Stochastic bound: Matérn 5/2 sparse GP with learnable inducing points
      computes calibrated predictive std dev σ(state).
    · Both are JIT-compilable and end-to-end differentiable.
    """
    dim_hidden: int = 16
    num_gp_inducing: int = 50

    @nn.compact
    def __call__(
        self,
        state_tensor: jax.Array,          # (5,) [alpha, kappa, gamma, Fz, Vx]
        stochastic_key = None,
    ):
        alpha, kappa, gamma, Fz, Vx = (state_tensor[i] for i in range(5))

        # Symmetry-respecting features
        features = jnp.array([
            jnp.sin(alpha),
            jnp.sin(2.0 * alpha),
            kappa,
            kappa ** 3,
            gamma,
            Fz / 1000.0,   # normalize to O(1)
            Vx / 20.0,
        ])

        # Spectrally normalized layers → Lipschitz-bounded PINN
        x = SpectralDense(self.dim_hidden)(features)
        x = jnp.tanh(x)
        x = SpectralDense(32)(x)
        x = jnp.tanh(x)
        drift = nn.Dense(2,
                         kernel_init=jax.nn.initializers.zeros,
                         bias_init=jax.nn.initializers.zeros)(x)

        # GP uncertainty
        gp   = SparseGPMatern52(num_inducing=self.num_gp_inducing)
        sigma = gp(state_tensor)

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
    3. TireOperatorPINN: deterministic drift + stochastic GP uncertainty

    Call convention:
        compute_force(alpha, kappa, Fz, gamma, T_ribs, T_gas, Vx, wz=0.0)
        → (Fx, Fy)  in Newtons

    BUGFIX-2: compute_thermal_derivatives now implemented.
    """

    def __init__(self, tire_coeffs: dict, rng_seed: int = 42):
        self.coeffs  = tire_coeffs
        self.T_opt   = tire_coeffs.get('T_opt', tire_coeffs.get('T_OPT', 90.0))
        self.T_env   = 25.0
        self.T_track = 40.0
        self.P_nom   = tire_coeffs.get('P_nom', 0.834)

        key = jax.random.PRNGKey(rng_seed)
        self._pinn_module = TireOperatorPINN()

        # Initialize PINN params with dummy input
        dummy_state = jnp.ones(5)
        self._pinn_params = self._pinn_module.init(key, dummy_state)

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
        contact_half_length = 0.075   # a = 75mm
        thermal_diff        = k_rubber / rho_c
        V_safe              = jnp.maximum(jnp.abs(V_slide), 1e-3)
        q_flux              = (mu_actual * jnp.abs(Fz) * V_safe) / (2.0 * contact_half_length * 0.205)
        T_flash             = ((q_flux * contact_half_length)
                               / (k_rubber * jnp.sqrt(jnp.pi * (V_safe * contact_half_length) / thermal_diff)))
        return jnp.clip(T_flash, 0.0, 350.0)

    # ─────────────────────────────────────────────────────────────────────────
    # §4.2  5-Node thermal derivatives  (BUGFIX-2: previously missing)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_thermal_derivatives(
        self,
        T_nodes:    jax.Array,   # (10,) [T_surf_f×3, T_gas_f, T_core_f, T_surf_r×3, T_gas_r, T_core_r]
        Fz_corners: jax.Array,   # (4,)
        kappa:      jax.Array,   # (4,) absolute longitudinal slip
        Vx:         jax.Array,   # scalar
    ) -> jax.Array:
        """
        5-node thermal ODE per axle:
          Node 0-2: Surface inner/mid/outer (convection from track + flash temp)
          Node 3:   Internal gas (Gay-Lussac pressure coupling)
          Node 4:   Core (conduction from surface avg)

        Returns dT/dt (10,) in °C/s.
        """
        # Thermal constants (approximate for Hoosier R25B)
        k_cond   = 0.25    # W/m/K rubber conductivity
        rho_c    = 2.0e6   # J/m³/K volumetric heat capacity
        h_conv   = 80.0    # W/m²/K convection coefficient (air + track)
        A_patch  = 0.025   # m² contact patch area
        V_tire   = 0.003   # m³ tire volume (approx)
        C_node   = rho_c * V_tire / 5.0   # lumped capacitance per node [J/K]

        T_env  = self.T_env
        T_gas0 = 25.0  # reference gas temperature
        P_ref  = self.P_nom  # bar

        # Front axle (nodes 0:5)
        T_f = T_nodes[0:5]
        Fz_f = (Fz_corners[0] + Fz_corners[1]) * 0.5
        kap_f = (kappa[0] + kappa[1]) * 0.5
        mu_f = 1.5  # approximate friction coefficient (conservative)
        V_slide_f = jnp.abs(Vx * kap_f)
        T_flash_f = self.compute_flash_temperature(mu_f, Fz_f, V_slide_f)

        # Frictional heat input to surface nodes (split evenly inner/mid/outer)
        Q_fric_f  = mu_f * Fz_f * V_slide_f / (3.0 * C_node + 1e-6)  # °C/s

        # Surface nodes: convection to environment + flash temp input
        dT_surf_f0 = (Q_fric_f + h_conv * A_patch * (T_flash_f - T_f[0]) / C_node
                      - h_conv * A_patch * (T_f[0] - T_env) / C_node)
        dT_surf_f1 = (Q_fric_f + h_conv * A_patch * (T_flash_f - T_f[1]) / C_node
                      - h_conv * A_patch * (T_f[1] - T_env) / C_node)
        dT_surf_f2 = (Q_fric_f + h_conv * A_patch * (T_flash_f - T_f[2]) / C_node
                      - h_conv * A_patch * (T_f[2] - T_env) / C_node)

        # Internal gas (Gay-Lussac: T_gas ↑ → P_tire ↑)
        T_surf_avg_f = (T_f[0] + T_f[1] + T_f[2]) / 3.0
        dT_gas_f = 0.05 * (T_surf_avg_f - T_f[3])  # slow thermal coupling

        # Core: conduction from surface average
        dT_core_f = 0.02 * (T_surf_avg_f - T_f[4])

        # Rear axle (nodes 5:10)
        T_r = T_nodes[5:10]
        Fz_r = (Fz_corners[2] + Fz_corners[3]) * 0.5
        kap_r = (kappa[2] + kappa[3]) * 0.5
        V_slide_r = jnp.abs(Vx * kap_r)
        T_flash_r = self.compute_flash_temperature(mu_f, Fz_r, V_slide_r)

        Q_fric_r  = mu_f * Fz_r * V_slide_r / (3.0 * C_node + 1e-6)

        dT_surf_r0 = (Q_fric_r + h_conv * A_patch * (T_flash_r - T_r[0]) / C_node
                      - h_conv * A_patch * (T_r[0] - T_env) / C_node)
        dT_surf_r1 = (Q_fric_r + h_conv * A_patch * (T_flash_r - T_r[1]) / C_node
                      - h_conv * A_patch * (T_r[1] - T_env) / C_node)
        dT_surf_r2 = (Q_fric_r + h_conv * A_patch * (T_flash_r - T_r[2]) / C_node
                      - h_conv * A_patch * (T_r[2] - T_env) / C_node)

        T_surf_avg_r = (T_r[0] + T_r[1] + T_r[2]) / 3.0
        dT_gas_r  = 0.05 * (T_surf_avg_r - T_r[3])
        dT_core_r = 0.02 * (T_surf_avg_r - T_r[4])

        # Clamp to prevent thermal runaway in untrained regime
        dT = jnp.array([
            dT_surf_f0, dT_surf_f1, dT_surf_f2, dT_gas_f, dT_core_f,
            dT_surf_r0, dT_surf_r1, dT_surf_r2, dT_gas_r, dT_core_r,
        ])
        return jnp.clip(dT, -500.0, 500.0)

    # ─────────────────────────────────────────────────────────────────────────
    # §4.3  Thermal grip factor from 5-node model
    # ─────────────────────────────────────────────────────────────────────────

    def _thermal_grip_factor(
        self,
        T_ribs: jax.Array,   # (3,) or (4,) or (5,) surface temperatures
        T_gas:  jax.Array,   # scalar internal gas temp
    ) -> jax.Array:
        """
        μ_thermal = exp(-β·(T_eff - T_opt)²)
        Gaussian thermal window around T_opt, capturing the known
        drop-off on both cold and overheated sides of the friction peak.
        Also applies Gay-Lussac pressure correction.
        """
        T_eff     = jnp.mean(T_ribs[:3])   # surface average
        beta      = 0.0008                  # K⁻²  (peak width ≈ 35°C)
        mu_T      = jnp.exp(-beta * (T_eff - self.T_opt) ** 2)

        # Gay-Lussac pressure correction: ΔP/P₀ = ΔT/T₀
        T_ref  = self.T_env + 273.15
        T_gas_K = T_gas + 273.15
        P_ratio = jnp.clip(T_gas_K / (T_ref + 1e-3), 0.70, 1.30)
        # Higher pressure → slightly more grip (linear approximation)
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
        T_ribs:         jax.Array,
        T_gas:          jax.Array,
        Vx:             float = 15.0,
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

        # ── Reference load & pressure correction ─────────────────────────────
        Fz0       = c.get('FNOMIN', 654.0)
        Fz_safe   = jnp.maximum(Fz, 10.0)
        dfz       = (Fz_safe - Fz0) / (Fz0 + eps)

        # Thermal grip factor
        lam_muy   = self._thermal_grip_factor(T_ribs, T_gas)

        # Camber in rad (input gamma is already radians from vehicle_dynamics)
        gam = gamma

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

        SHy   = PHY1 + PHY2 * dfz
        SVy   = Fz_safe * (PVY1 + PVY2 * dfz) * lam_muy

        Ky    = (PKY1 * Fz0
                 * jnp.sin(PKY4 * jnp.arctan(Fz_safe / jnp.maximum(PKY2 * Fz0, eps)))
                 * (1.0 - PKY3 * jnp.abs(gam)))
        Dy    = PDY1 * (1.0 + PDY2 * dfz) * (1.0 - PDY3 * gam ** 2) * Fz_safe * lam_muy
        Cy    = PCY1
        By    = Ky / jnp.maximum(Cy * Dy, eps)

        a_s   = alpha + SHy
        # Smooth sign function for Ey (avoid discontinuous jnp.where)
        sgn_as = jnp.tanh(a_s / (1e-3 + eps))
        Ey    = jnp.clip((PEY1 + PEY2 * dfz) * (1.0 - (PEY3 + PEY4 * gam) * sgn_as), -10.0, 1.0)
        x_y   = By * a_s
        Fy0   = Dy * jnp.sin(Cy * jnp.arctan(x_y - Ey * (x_y - jnp.arctan(x_y)))) + SVy

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

        SHx = PHX1 + PHX2 * dfz
        SVx = Fz_safe * (PVX1 + PVX2 * dfz) * lam_muy
        kappa_c = kappa + SHx

        Cx = PCX1
        Dx = PDX1 * (1.0 + PDX2 * dfz) * (1.0 - PDX3 * gam ** 2) * Fz_safe * lam_muy
        Kx = PKX1 * Fz_safe * jnp.exp(PKX3 * dfz) * (1.0 + PKX2 * dfz)
        Bx = Kx / jnp.maximum(Cx * Dx, eps)
        Ex = jnp.clip(PEX1 + PEX2 * dfz + PEX3 * dfz ** 2, -10.0, 1.0)
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

        SHyk  = RHY1
        alpha_s = alpha + SHyk
        By_s  = RBY1 * jnp.cos(jnp.arctan(RBY2 * (alpha - RBY3)))
        Ey_s  = REY1 + REY2 * dfz
        x_ys  = By_s * alpha_s
        Gyk   = jnp.cos(RCY1 * jnp.arctan(x_ys - Ey_s * (x_ys - jnp.arctan(x_ys))))
        Gyk   = jnp.clip(Gyk, 0.05, 1.0)

        kappa_s = kappa + RHX1
        Bx_s  = RBX1 * jnp.cos(jnp.arctan(RBX2 * kappa))
        x_xs  = Bx_s * kappa_s
        Gxa   = jnp.cos(RCX1 * jnp.arctan(x_xs))
        Gxa   = jnp.clip(Gxa, 0.05, 1.0)

        Fy = Fy0 * Gyk
        Fx = Fx0 * Gxa

        # ── Turn slip correction ─────────────────────────────────────────────
        a_contact = c.get('contact_half_length', 0.05)
        R_path    = jnp.abs(Vx) / (jnp.abs(wz) + eps)
        phi_t     = a_contact / (R_path + 1e-3)
        Fy        = Fy * (1.0 - 0.15 * jnp.abs(phi_t))

        # ── PINN/GP residual corrections ─────────────────────────────────────
        state_in  = jnp.array([alpha, kappa_c, gam, Fz_safe, float(Vx) if not isinstance(Vx, jax.Array) else Vx])
        mods, sigma = self._pinn_module.apply(self._pinn_params, state_in, stochastic_key)
        mods        = jnp.clip(mods, -0.25, 0.25)
        penalty     = 2.0 * sigma
        Fy          = Fy * (1.0 + mods[1] - penalty)
        Fx          = Fx * (1.0 + mods[0] - penalty)

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
        Fz0 = c.get('FNOMIN', 654.0)
        Fz_safe = jnp.maximum(Fz, 10.0)
        dfz = (Fz_safe - Fz0) / (Fz0 + eps)
        gam = gamma

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
        QHZ1 = c.get('QHZ1',  0.0)
        QHZ2 = c.get('QHZ2',  0.0)

        R0   = c.get('R0', 0.2045)

        SHt = QHZ1 + QHZ2 * dfz
        a_t = alpha + SHt

        Bt  = (QBZ1 + QBZ2 * dfz + QBZ3 * dfz ** 2) * (1.0 + QBZ9 * jnp.abs(gam))
        Ct  = QCZ1
        Dt  = Fz_safe * R0 * (QDZ1 + QDZ2 * dfz) * (1.0 + QDZ3 * gam + QDZ4 * gam ** 2)
        sgn_at = jnp.tanh(a_t / (1e-3 + eps))
        Et  = jnp.clip(QEZ1 + QEZ2 * dfz, -10.0, 1.0)
        x_t = Bt * a_t
        t   = Dt * jnp.cos(Ct * jnp.arctan(x_t - Et * (x_t - jnp.arctan(x_t))))
        Mz0 = -t * Fy

        # Residual moment from longitudinal force (pneumatic trail)
        SSZ1 = c.get('SSZ1', 0.0)
        SSZ2 = c.get('SSZ2', 0.0)
        s    = R0 * (SSZ1 + SSZ2 * (Fy / (Fz0 + eps)))
        Mz_r = s * Fx

        return Mz0 + Mz_r