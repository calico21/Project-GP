import jax
import jax.numpy as jnp
import numpy as np
import casadi as ca

# --- LFNO & NSDE COMBINED ARCHITECTURE ---
class TireOperatorNSDE:
    """
    Lightweight Fourier Neural Operator (LFNO) + Neural SDE.
    1. LFNO: Factorizes the spectral convolution by truncating high-frequency modes, 
       reducing parameters and exponentially accelerating PDE evaluation.
    2. NSDE: Splits the latent space into a Deterministic Drift (mean physics) 
       and a Stochastic Diffusion (friction uncertainty boundary).
    """
    def __init__(self, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        
        self.modes = 3 
        self.dim_hidden = 16
        self.w_fourier = jax.random.normal(k1, (self.modes, self.dim_hidden), dtype=jnp.complex64) * 0.02
        self.w_local = jax.random.normal(k2, (5, self.dim_hidden)) * jnp.sqrt(2.0 / 5)
        self.b_local = jnp.zeros(self.dim_hidden)
        
        self.w_proj = jax.random.normal(k3, (self.dim_hidden, 32)) * jnp.sqrt(2.0 / self.dim_hidden)
        self.b_proj = jnp.zeros(32)
        
        # ZERO INITIALIZATION: Ensures grip modification starts at 0.0
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
        # Transform the state into the frequency domain -> shape: (3,)
        x_ft = jnp.fft.rfft(state_tensor)
        
        # Factorized Spectral Convolution
        # Expand dims to multiply each mode by its 16 hidden channel weights -> shape: (3, 16)
        out_ft_transformed = x_ft[:self.modes, None] * self.w_fourier
        
        # Inverse FFT back to spatial domain. (n=5 reconstructs the 5 original spatial nodes)
        # shape: (5, 16)
        x_spectral = jnp.fft.irfft(out_ft_transformed, n=5, axis=0) 
        
        # Add local spatial bypass and activation
        x_local = jnp.dot(state_tensor, self.w_local) + self.b_local # shape: (16,)
        
        # Sum spectral features over spatial dimension and combine
        x_combined = jnp.tanh(jnp.sum(x_spectral, axis=0) + x_local) # shape: (16,)
        
        # Latent projection
        latent = jnp.tanh(jnp.dot(x_combined, self.w_proj) + self.b_proj)
        
        # --- NSDE SPLIT ---
        # 1. Drift: Expected physics response
        drift = jnp.dot(latent, self.w_drift) + self.b_drift
        
        # 2. Diffusion: Distance-aware uncertainty (softplus ensures strictly positive variance)
        diffusion = jax.nn.softplus(jnp.dot(latent, self.w_diff) + self.b_diff)
        
        # Reparameterization Trick for Stochastic RL Rollouts
        if stochastic_key is not None:
            noise = jax.random.normal(stochastic_key, shape=(2,))
            out = drift + diffusion * noise
        else:
            out = drift  # Deterministic execution for TuRBO / standard physics
            
        return out, diffusion


class PacejkaTire:
    """
    Differentiable 5-Node Thermodynamic Tire Model.
    Governed by LFNO/NSDE to compute exact thermal dynamics and stochastic grip limits.
    """
    def __init__(self, tire_coeffs, rng_seed=42):
        self.coeffs = tire_coeffs
        
        self.T_opt = self.coeffs.get('T_opt', 90.0)
        self.T_env = 25.0
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

        # Initialize the Factorized LFNO/NSDE
        key = jax.random.PRNGKey(rng_seed)
        self.operator = TireOperatorNSDE(key)

    def _is_symbolic(self, var):
        return isinstance(var, (ca.SX, ca.MX))

    def compute_force(self, alpha, kappa, Fz, gamma, T_ribs, T_gas, Vx=15.0, stochastic_key=None):
        is_sym = self._is_symbolic(alpha) or self._is_symbolic(Fz)
        
        if is_sym:
            _sin, _arctan, _max, _min = ca.sin, ca.arctan, ca.fmax, ca.fmin
            T_surf_in, T_surf_mid, T_surf_out = T_ribs[0], T_ribs[1], T_ribs[2]
        else:
            _sin, _arctan, _max, _min = jnp.sin, jnp.arctan, jnp.maximum, jnp.minimum
            T_surf_in, T_surf_mid, T_surf_out = T_ribs

        T_gas_K = T_gas + 273.15
        T_env_K = self.T_env + 273.15
        P_dyn = self.P_nom * (T_gas_K / T_env_K)
        
        dP = P_dyn - (self.P_nom + 0.2)
        pressure_modifier = _max(0.6, _min(1.0, 1.0 - 0.15 * (dP**2)))

        dy, cy, by, ey = 1.3, 1.5, 10.0, -1.0
        dx, cx, bx, ex = 1.35, 1.6, 12.0, -0.5

        Fz_nom = 4000.0
        d_fz = (Fz - Fz_nom) / Fz_nom
        lambda_mu_y = (1.0 - 0.1 * d_fz) * pressure_modifier
        lambda_mu_x = (1.0 - 0.08 * d_fz) * pressure_modifier
        
        Fz_in  = Fz * (0.333 + 0.15 * gamma)
        Fz_mid = Fz * 0.334
        Fz_out = Fz * (0.333 - 0.15 * gamma)
        
        T_eff = (T_surf_in * Fz_in + T_surf_mid * Fz_mid + T_surf_out * Fz_out) / (Fz + 1e-6)
        therm_factor = _max(0.5, _min(1.0, 1.0 - 0.002 * (T_eff - self.T_opt)**2))

        # --- NSDE Integration Hook ---
        if is_sym:
            mu_x_mod, mu_y_mod = 1.0, 1.0
        else:
            state_in = jnp.array([alpha, kappa, gamma, Fz, Vx])
            # The operator returns the transient modification and the diffusion penalty
            mods, diff = self.operator.apply(state_in, stochastic_key)
            
            # The Diffusion penalty actively degrades grip if the state is highly uncertain
            uncertainty_penalty = jnp.mean(diff) * 0.1 
            
            mu_x_mod = 1.0 + mods[0] - uncertainty_penalty
            mu_y_mod = 1.0 + mods[1] - uncertainty_penalty

        mu_y = dy * lambda_mu_y * therm_factor * mu_y_mod
        mu_x = dx * lambda_mu_x * therm_factor * mu_x_mod

        Fy = Fz * mu_y * _sin(cy * _arctan(by * alpha - ey * (by * alpha - _arctan(by * alpha))))
        Fx = Fz * mu_x * _sin(cx * _arctan(bx * kappa - ex * (bx * kappa - _arctan(bx * kappa))))

        return Fx, Fy

    def compute_thermal_dynamics(self, Fx, Fy, Fz, gamma, alpha, kappa, Vx, T_core, T_ribs, T_gas):
        is_sym = self._is_symbolic(Fx) or self._is_symbolic(alpha)
        
        if is_sym:
            _abs = lambda x: ca.sqrt(x**2 + 1e-6)
            _tan = ca.tan
            T_surf_in, T_surf_mid, T_surf_out = T_ribs[0], T_ribs[1], T_ribs[2]
        else:
            _abs, _tan = jnp.abs, jnp.tan
            T_surf_in, T_surf_mid, T_surf_out = T_ribs

        dist_in = 0.333 + 0.15 * gamma
        dist_out = 0.333 - 0.15 * gamma
        
        P_fric = _abs(Fx * Vx * kappa) + _abs(Fy * Vx * _tan(alpha))
        P_in = P_fric * dist_in
        P_mid = P_fric * 0.334
        P_out = P_fric * dist_out
        
        P_core_in = self.q_roll * _abs(Fz) * _abs(Vx)

        k_vert_rib = self.k_cond_vert / 3.0
        Q_v_in = k_vert_rib * (T_core - T_surf_in)
        Q_v_mid = k_vert_rib * (T_core - T_surf_mid)
        Q_v_out = k_vert_rib * (T_core - T_surf_out)

        Q_lat_in_mid = self.k_cond_lat * (T_surf_in - T_surf_mid)
        Q_lat_mid_out = self.k_cond_lat * (T_surf_mid - T_surf_out)

        h_dyn = self.h_conv_ext * (1.0 + 0.5 * _abs(Vx))
        Q_air_in = h_dyn * self.A_rib * (T_surf_in - self.T_env)
        Q_air_mid = h_dyn * self.A_rib * (T_surf_mid - self.T_env)
        Q_air_out = h_dyn * self.A_rib * (T_surf_out - self.T_env)
        
        Q_gas_conv = self.h_conv_int * (T_core - T_gas)

        dT_in_dt = (P_in + Q_v_in - Q_lat_in_mid - Q_air_in) / (self.m_rib * self.Cp_rubber)
        dT_mid_dt = (P_mid + Q_v_mid + Q_lat_in_mid - Q_lat_mid_out - Q_air_mid) / (self.m_rib * self.Cp_rubber)
        dT_out_dt = (P_out + Q_v_out + Q_lat_mid_out - Q_air_out) / (self.m_rib * self.Cp_rubber)
        
        dT_core_dt = (P_core_in - Q_v_in - Q_v_mid - Q_v_out - Q_gas_conv) / (self.m_core * self.Cp_rubber)
        dT_gas_dt = Q_gas_conv / (self.m_gas * self.Cv_gas)
        
        if is_sym:
            return [dT_in_dt, dT_mid_dt, dT_out_dt], dT_core_dt, dT_gas_dt
        else:
            return jnp.array([dT_in_dt, dT_mid_dt, dT_out_dt]), dT_core_dt, dT_gas_dt