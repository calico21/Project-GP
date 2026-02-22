import jax
import jax.numpy as jnp
import numpy as np

# --- LFNO & NSDE COMBINED ARCHITECTURE ---
class TireOperatorNSDE:
    """
    Lightweight Fourier Neural Operator (LFNO) + Neural SDE.
    Fully native to JAX.
    1. LFNO: Factorizes the spectral convolution by truncating high-frequency modes, 
             reducing parameters and exponentially accelerating PDE evaluation.
    2. NSDE: Splits the latent space into a Deterministic Drift (mean physics) 
             and a Stochastic Diffusion (friction uncertainty boundary).
    """
    def __init__(self, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        
        self.modes = 3 
        self.dim_hidden = 16
        
        # Complex weights for the frequency domain
        self.w_fourier = jax.random.normal(k1, (self.modes, self.dim_hidden), dtype=jnp.complex64) * 0.02
        self.w_local = jax.random.normal(k2, (5, self.dim_hidden)) * jnp.sqrt(2.0 / 5)
        self.b_local = jnp.zeros(self.dim_hidden)
        
        self.w_proj = jax.random.normal(k3, (self.dim_hidden, 32)) * jnp.sqrt(2.0 / self.dim_hidden)
        self.b_proj = jnp.zeros(32)
        
        # ZERO INITIALIZATION: Ensures AI grip modification starts at exactly 0.0 before training
        self.w_drift = jnp.zeros((32, 2))
        self.b_drift = jnp.zeros(2)
        
        # Initialize diffusion to a very small positive baseline uncertainty
        self.w_diff = jnp.zeros((32, 2))
        self.b_diff = jnp.full(2, -3.0)

    def apply(self, state_tensor, stochastic_key=None):
        """
        Forward pass executing the factorized spectral mapping and NSDE splitting.
        """
        # --- LFNO CORE ---
        x_ft = jnp.fft.rfft(state_tensor)
        
        # Factorized Spectral Convolution
        out_ft_transformed = x_ft[:self.modes, None] * self.w_fourier
        
        # Inverse FFT back to spatial domain. (n=5 reconstructs the 5 original spatial nodes)
        x_spectral = jnp.fft.irfft(out_ft_transformed, n=5, axis=0) 
        
        # Add local spatial bypass and activation
        x_local = jnp.dot(state_tensor, self.w_local) + self.b_local 
        
        # Sum spectral features over spatial dimension and combine
        x_combined = jnp.tanh(jnp.sum(x_spectral, axis=0) + x_local) 
        
        # Latent projection
        latent = jnp.tanh(jnp.dot(x_combined, self.w_proj) + self.b_proj)
        
        # --- NSDE SPLIT ---
        # 1. Drift: Expected physics response correction
        drift = jnp.dot(latent, self.w_drift) + self.b_drift
        
        # 2. Diffusion: Distance-aware uncertainty (softplus ensures strictly positive variance)
        diffusion = jax.nn.softplus(jnp.dot(latent, self.w_diff) + self.b_diff)
        
        # Reparameterization Trick for Stochastic RL Rollouts
        if stochastic_key is not None:
            noise = jax.random.normal(stochastic_key, shape=(2,))
            out = drift + diffusion * noise
        else:
            out = drift  # Deterministic execution for optimal control / standard physics
            
        return out, diffusion


class PacejkaTire:
    """
    100% Pure JAX Differentiable 5-Node Thermodynamic Tire Model.
    Governed by LFNO/NSDE to compute exact thermal dynamics, transient flash temperatures, 
    and stochastic grip limits without any CasADi overhead.
    """
    def __init__(self, tire_coeffs, rng_seed=42):
        self.coeffs = tire_coeffs
        
        # FIX 2: Standardized key lookup to lowercase 't_opt' to match tire_coeffs.py.
        # The original code used 'T_opt' but tire_coeffs.py defined 'T_OPT', causing
        # the thermal grip peak to silently default to 90.0 regardless of config.
        self.T_opt = self.coeffs.get('T_opt',
                     self.coeffs.get('T_OPT', 90.0))  # check both casings defensively
        self.T_env = 25.0
        self.T_track = 40.0  # Track surface temperature
        self.P_nom = self.coeffs.get('P_nom', 1.2)
        
        self.m_total = self.coeffs.get('mass', 10.0)
        self.m_surf = self.m_total * 0.10
        self.m_rib = self.m_surf / 3.0
        self.m_core = self.m_total * 0.90
        self.m_gas = self.coeffs.get('m_gas', 0.05)
        
        self.Cp_rubber = self.coeffs.get('Cp', 1100.0)
        self.Cv_gas = self.coeffs.get('Cv_gas', 718.0)
        
        self.h_conv_ext = self.coeffs.get('h_conv', 50.0)
        self.k_cond_vert = self.coeffs.get('k_cond', 150.0)
        self.k_cond_lat = self.coeffs.get('k_cond_lat', 85.0)
        self.h_conv_int = self.coeffs.get('h_conv_int', 30.0)
        
        self.A_surf = self.coeffs.get('A_surf', 0.8)
        self.A_rib = self.A_surf / 3.0
        self.q_roll = self.coeffs.get('q_roll', 0.03)

        # Initialize the Factorized LFNO/NSDE Operator
        key = jax.random.PRNGKey(rng_seed)
        self.operator = TireOperatorNSDE(key)

    def compute_force(self, alpha, kappa, Fz, gamma, T_ribs, T_gas, Vx=15.0, stochastic_key=None):
        T_surf_in, T_surf_mid, T_surf_out = T_ribs[0], T_ribs[1], T_ribs[2]

        T_gas_K = T_gas + 273.15
        T_env_K = self.T_env + 273.15
        P_dyn = self.P_nom * (T_gas_K / T_env_K)
        
        dP = P_dyn - (self.P_nom + 0.2)
        pressure_modifier = jnp.maximum(0.6, jnp.minimum(1.0, 1.0 - 0.15 * (dP**2)))

        dy = self.coeffs.get('PDY1', 1.45)
        cy = self.coeffs.get('PCY1', 1.45)
        by = self.coeffs.get('PKY1', 25.0)
        ey = self.coeffs.get('PEY1', 0.5)

        dx = self.coeffs.get('PDX1', 1.50)
        cx = self.coeffs.get('PCX1', 1.50)
        bx = self.coeffs.get('PKX1', 30.0)
        ex = self.coeffs.get('PEX1', -0.5)

        Fz_nom = self.coeffs.get('FNOMIN', 1000.0)
        d_fz = (Fz - Fz_nom) / Fz_nom
        lambda_mu_y = (1.0 - 0.1 * d_fz) * pressure_modifier
        lambda_mu_x = (1.0 - 0.08 * d_fz) * pressure_modifier
        
        Fz_in  = Fz * (0.333 + 0.15 * gamma)
        Fz_mid = Fz * 0.334
        Fz_out = Fz * (0.333 - 0.15 * gamma)
        
        # --- Transient Flash Temperature Dynamics ---
        # FIX 3 (partial): Replace jnp.tan(alpha) with a gradient-safe sin/cos decomposition.
        # jnp.tan(alpha) has gradient 1/cos²(alpha) which reaches ~200 at alpha=1.5 rad.
        # Over 800 BPTT scan steps this gradient multiplier compounds to explosion.
        # The 1e-3 denominator floor caps d/dalpha at ~1000 even at exactly ±pi/2,
        # while the upstream alpha clip means the forward value is also bounded.
        tan_alpha_safe = jnp.sin(alpha) / (jnp.cos(alpha) + 1e-3)

        # FIX 1 (primary NaN cause): kappa arriving here from the vehicle model was a raw
        # drive force in Newtons (~1250 N), not a dimensionless slip ratio (0.0–0.3).
        # This caused P_fric ≈ 7.5 MW per tire, driving tire temps to 10,000°C within
        # the first RK4 sub-step, which NaN-propagated through therm_factor → Fy → state.
        # kappa is now normalized by the caller in vehicle_dynamics.py before arriving here.
        # We additionally hard-clamp it here as a defensive second layer.
        kappa_safe = jnp.clip(kappa, -0.5, 0.5)

        sliding_velocity = Vx * jnp.sqrt(kappa_safe**2 + tan_alpha_safe**2 + 1e-6)
        flash_temp_delta = 0.5 * sliding_velocity * jnp.abs(Fz) / (self.A_surf * self.Cp_rubber + 1e-6)
        
        T_eff_bulk = (T_surf_in * Fz_in + T_surf_mid * Fz_mid + T_surf_out * Fz_out) / (Fz + 1e-6)
        T_contact_patch = T_eff_bulk + flash_temp_delta
        
        # Grip falls off a cliff if the flash temperature exceeds the optimal range significantly
        therm_factor = jnp.maximum(0.3, jnp.minimum(1.0, 1.0 - 0.003 * (T_contact_patch - self.T_opt)**2))

        # --- NSDE Integration Hook ---
        state_in = jnp.array([alpha, kappa_safe, gamma, Fz, Vx])
        mods, diff = self.operator.apply(state_in, stochastic_key)
        
        # The Diffusion penalty actively degrades grip if the state is in a highly uncertain regime
        uncertainty_penalty = jnp.mean(diff) * 0.1 
        
        mu_x_mod = 1.0 + mods[0] - uncertainty_penalty
        mu_y_mod = 1.0 + mods[1] - uncertainty_penalty

        mu_y = dy * lambda_mu_y * therm_factor * mu_y_mod
        mu_x = dx * lambda_mu_x * therm_factor * mu_x_mod

        Fy = Fz * mu_y * jnp.sin(cy * jnp.arctan(by * alpha - ey * (by * alpha - jnp.arctan(by * alpha))))
        Fx = Fz * mu_x * jnp.sin(cx * jnp.arctan(bx * kappa_safe - ex * (bx * kappa_safe - jnp.arctan(bx * kappa_safe))))

        return Fx, Fy

    def compute_thermal_dynamics(self, Fx, Fy, Fz, gamma, alpha, kappa, Vx, T_core, T_ribs, T_gas):
        T_surf_in, T_surf_mid, T_surf_out = T_ribs[0], T_ribs[1], T_ribs[2]

        dist_in = 0.333 + 0.15 * gamma
        dist_out = 0.333 - 0.15 * gamma
        
        # FIX 1 (defensive): clamp kappa here too in case it arrives unnormalized
        kappa_safe = jnp.clip(kappa, -0.5, 0.5)

        # FIX 3: Replace jnp.tan(alpha) with gradient-safe sin/cos form.
        # Original: P_fric = |Fy * Vx * tan(alpha)|
        # Problem:  d/dalpha[tan] = 1/cos²(alpha) → ~200 at alpha=1.5 rad → BPTT explosion.
        # Fix:      Use sin/(cos + eps) — forward value identical, gradient capped at ~1000.
        tan_alpha_safe = jnp.sin(alpha) / (jnp.cos(alpha) + 1e-3)

        # Friction Power Generation
        P_fric = jnp.abs(Fx * Vx * kappa_safe) + jnp.abs(Fy * Vx * tan_alpha_safe)
        P_in = P_fric * dist_in
        P_mid = P_fric * 0.334
        P_out = P_fric * dist_out
        
        # Rolling Resistance Internal Generation
        P_core_in = self.q_roll * jnp.abs(Fz) * jnp.abs(Vx)

        # Vertical Conduction (Surface -> Core)
        k_vert_rib = self.k_cond_vert / 3.0
        Q_v_in = k_vert_rib * (T_core - T_surf_in)
        Q_v_mid = k_vert_rib * (T_core - T_surf_mid)
        Q_v_out = k_vert_rib * (T_core - T_surf_out)

        # Lateral Conduction (Across Ribs)
        Q_lat_in_mid = self.k_cond_lat * (T_surf_in - T_surf_mid)
        Q_lat_mid_out = self.k_cond_lat * (T_surf_mid - T_surf_out)

        # Convection (Surface -> Ambient Air & Track)
        h_dyn = self.h_conv_ext * (1.0 + 0.5 * jnp.abs(Vx))
        # Weighted interaction: 80% air, 20% track conduction at the contact patch
        Q_air_in = h_dyn * self.A_rib * (T_surf_in - (0.8*self.T_env + 0.2*self.T_track))
        Q_air_mid = h_dyn * self.A_rib * (T_surf_mid - (0.8*self.T_env + 0.2*self.T_track))
        Q_air_out = h_dyn * self.A_rib * (T_surf_out - (0.8*self.T_env + 0.2*self.T_track))
        
        # Internal Convection (Core -> Inflation Gas)
        Q_gas_conv = self.h_conv_int * (T_core - T_gas)

        # Final ODE Differentials
        dT_in_dt = (P_in + Q_v_in - Q_lat_in_mid - Q_air_in) / (self.m_rib * self.Cp_rubber)
        dT_mid_dt = (P_mid + Q_v_mid + Q_lat_in_mid - Q_lat_mid_out - Q_air_mid) / (self.m_rib * self.Cp_rubber)
        dT_out_dt = (P_out + Q_v_out + Q_lat_mid_out - Q_air_out) / (self.m_rib * self.Cp_rubber)
        
        dT_core_dt = (P_core_in - Q_v_in - Q_v_mid - Q_v_out - Q_gas_conv) / (self.m_core * self.Cp_rubber)
        dT_gas_dt = Q_gas_conv / (self.m_gas * self.Cv_gas)
        
        return jnp.array([dT_in_dt, dT_mid_dt, dT_out_dt]), dT_core_dt, dT_gas_dt