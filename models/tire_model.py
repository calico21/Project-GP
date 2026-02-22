import jax
import jax.numpy as jnp
import numpy as np


class SparseGPMatern52:
    """
    Sparse Gaussian Process using a Matérn 5/2 kernel.
    Provides mathematically calibrated uncertainty bounds based on the 
    distance from known nominal operating regimes (inducing points).
    """
    def __init__(self, key, num_inducing=50):
        # Lengthscales dictate how quickly confidence decays for each state variable
        # [alpha, kappa, gamma, Fz, Vx]
        self.lengthscale = jnp.array([0.2, 0.15, 0.1, 400.0, 15.0]) 
        self.prior_variance = 0.08  # Maximum grip uncertainty (approx 8% loss of mu)
        
        # Generate synthetic inducing points representing "safe/known" track data
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        alpha_ip = jax.random.uniform(k1, (num_inducing,), minval=-0.15, maxval=0.15)
        kappa_ip = jax.random.uniform(k2, (num_inducing,), minval=-0.1, maxval=0.1)
        gamma_ip = jax.random.uniform(k3, (num_inducing,), minval=-0.05, maxval=0.05)
        Fz_ip = jax.random.uniform(k4, (num_inducing,), minval=600.0, maxval=1400.0)
        Vx_ip = jax.random.uniform(k5, (num_inducing,), minval=5.0, maxval=25.0)
        
        self.Z = jnp.stack([alpha_ip, kappa_ip, gamma_ip, Fz_ip, Vx_ip], axis=1)
        
        # Precompute the inverse covariance matrix of inducing points: (K_ZZ + noise*I)^-1
        K_ZZ = self._matern52_matrix(self.Z, self.Z)
        self.K_ZZ_inv = jnp.linalg.inv(K_ZZ + 1e-4 * jnp.eye(num_inducing))

    def _matern52_kernel(self, x1, x2):
        # Scaled Euclidean distance
        d = jnp.sqrt(jnp.sum(((x1 - x2) / self.lengthscale)**2) + 1e-8)
        sqrt5_d = jnp.sqrt(5.0) * d
        return self.prior_variance * (1.0 + sqrt5_d + (5.0 / 3.0) * d**2) * jnp.exp(-sqrt5_d)

    def _matern52_matrix(self, X1, X2):
        # Vectorized kernel matrix computation
        return jax.vmap(lambda x1: jax.vmap(lambda x2: self._matern52_kernel(x1, x2))(X2))(X1)

    def compute_std_dev(self, x_star):
        """Calculates the predictive standard deviation (sigma) at state x_star"""
        k_xx = self.prior_variance
        k_xZ = jax.vmap(lambda z: self._matern52_kernel(x_star, z))(self.Z)
        
        # Variance reduction due to proximity to known inducing points
        reduction = jnp.dot(k_xZ, jnp.dot(self.K_ZZ_inv, k_xZ))
        
        variance = jnp.maximum(k_xx - reduction, 1e-6)
        return jnp.sqrt(variance)
    

# --- PINN & GP COMBINED ARCHITECTURE ---
class TireOperatorPINN:
    """
    Symmetry-Respecting Physics-Informed Neural Network + Sparse GP.
    PINN computes the deterministic drift (grip modifications).
    Sparse GP computes the calibrated thermodynamic uncertainty bounds.
    """
    def __init__(self, key):
        k1, k2, k_gp = jax.random.split(key, 3)
        self.dim_hidden = 16
        
        self.w1 = jax.random.normal(k1, (7, self.dim_hidden)) * jnp.sqrt(2.0 / 7)
        self.b1 = jnp.zeros(self.dim_hidden)
        
        self.w2 = jax.random.normal(k2, (self.dim_hidden, 32)) * jnp.sqrt(2.0 / self.dim_hidden)
        self.b2 = jnp.zeros(32)
        
        self.w_drift = jnp.zeros((32, 2))
        self.b_drift = jnp.zeros(2)
        
        # Replace neural diffusion with the Gaussian Process
        self.gp = SparseGPMatern52(k_gp)

    def apply(self, state_tensor, stochastic_key=None):
        alpha, kappa, gamma, Fz, Vx = state_tensor
        
        features = jnp.array([
            jnp.sin(alpha), jnp.sin(2.0 * alpha),
            kappa, kappa**3,
            gamma, Fz, Vx
        ])
        
        x = jnp.tanh(jnp.dot(features, self.w1) + self.b1)
        latent = jnp.tanh(jnp.dot(x, self.w2) + self.b2)
        
        drift = jnp.dot(latent, self.w_drift) + self.b_drift
        
        # GP outputs exactly 1 standard deviation (sigma)
        sigma = self.gp.compute_std_dev(state_tensor)
        
        if stochastic_key is not None:
            noise = jax.random.normal(stochastic_key, shape=(2,))
            return drift + sigma * noise, sigma
            
        return drift, sigma

class PacejkaTire:
    """
    100% Pure JAX Differentiable 5-Node Thermodynamic Tire Model.
    Governed by PINN/GP to compute exact thermal dynamics, transient flash temperatures, 
    and stochastic grip limits without any CasADi overhead.
    """
    def __init__(self, tire_coeffs, rng_seed=42):
        self.coeffs = tire_coeffs
        
        # Standardized key lookup to lowercase 't_opt' to match tire_coeffs.py.
        self.T_opt = self.coeffs.get('T_opt',
                     self.coeffs.get('T_OPT', 90.0))  
        self.T_env = 25.0
        self.T_track = 40.0  
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

        # Initialize the Symmetry-Respecting PINN Operator
        key = jax.random.PRNGKey(rng_seed)
        self.operator = TireOperatorPINN(key)

    def compute_flash_temperature(self, mu_actual, Fz, V_slide):
        """
        Calculates the contact patch flash temperature using the Jaeger solution 
        for a sliding semi-infinite solid. Dimensionally correct heat flux.
        """
        k_rubber = 0.25              # Thermal conductivity (W/m*K)
        rho_c = 2.0e6                # Volumetric heat capacity (J/m^3*K)
        contact_patch_length = 0.15  # Estimated footprint length (m)
        a = contact_patch_length / 2.0 
        
        thermal_diffusivity = k_rubber / rho_c
        V_slide_safe = jnp.maximum(jnp.abs(V_slide), 1e-3)
        
        # Friction Power (q) spread over the contact patch area (length * width)
        q = (mu_actual * jnp.abs(Fz) * V_slide_safe) / (contact_patch_length * 0.205)
        
        # High-speed Jaeger flash temperature equation
        T_flash = (q * a) / (k_rubber * jnp.sqrt(jnp.pi * (V_slide_safe * a) / thermal_diffusivity))
        
        return jnp.clip(T_flash, a_min=0.0, a_max=350.0)

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
        tan_alpha_safe = jnp.sin(alpha) / (jnp.cos(alpha) + 1e-3)
        kappa_safe = jnp.clip(kappa, -0.5, 0.5)

        sliding_velocity = Vx * jnp.sqrt(kappa_safe**2 + tan_alpha_safe**2 + 1e-6)
        
        # Estimate base friction coefficient to break the thermal algebraic loop
        mu_base_estimate = jnp.sqrt((dy * lambda_mu_y)**2 + (dx * lambda_mu_x)**2)
        
        # Implement the Jaeger thermal solution
        flash_temp_delta = self.compute_flash_temperature(mu_base_estimate, Fz, sliding_velocity)
        
        T_eff_bulk = (T_surf_in * Fz_in + T_surf_mid * Fz_mid + T_surf_out * Fz_out) / (Fz + 1e-6)
        T_contact_patch = T_eff_bulk + flash_temp_delta
        
        therm_factor = jnp.maximum(0.3, jnp.minimum(1.0, 1.0 - 0.003 * (T_contact_patch - self.T_opt)**2))

        # --- GP Uncertainty & PINN Integration Hook ---
        state_in = jnp.array([alpha, kappa_safe, gamma, Fz, Vx])
        mods, sigma = self.operator.apply(state_in, stochastic_key)
        
        # Lower Confidence Bound (LCB) for robust Tube-MPC constraints.
        # We actively degrade the available friction by 2 standard deviations.
        # If the car enters high-slip (unknown) territory, sigma spikes, and grip plummets.
        uncertainty_penalty = 2.0 * sigma 
        
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
        
        kappa_safe = jnp.clip(kappa, -0.5, 0.5)
        tan_alpha_safe = jnp.sin(alpha) / (jnp.cos(alpha) + 1e-3)

        P_fric = jnp.abs(Fx * Vx * kappa_safe) + jnp.abs(Fy * Vx * tan_alpha_safe)
        P_in = P_fric * dist_in
        P_mid = P_fric * 0.334
        P_out = P_fric * dist_out
        
        P_core_in = self.q_roll * jnp.abs(Fz) * jnp.abs(Vx)

        k_vert_rib = self.k_cond_vert / 3.0
        Q_v_in = k_vert_rib * (T_core - T_surf_in)
        Q_v_mid = k_vert_rib * (T_core - T_surf_mid)
        Q_v_out = k_vert_rib * (T_core - T_surf_out)

        Q_lat_in_mid = self.k_cond_lat * (T_surf_in - T_surf_mid)
        Q_lat_mid_out = self.k_cond_lat * (T_surf_mid - T_surf_out)

        h_dyn = self.h_conv_ext * (1.0 + 0.5 * jnp.abs(Vx))
        Q_air_in = h_dyn * self.A_rib * (T_surf_in - (0.8*self.T_env + 0.2*self.T_track))
        Q_air_mid = h_dyn * self.A_rib * (T_surf_mid - (0.8*self.T_env + 0.2*self.T_track))
        Q_air_out = h_dyn * self.A_rib * (T_surf_out - (0.8*self.T_env + 0.2*self.T_track))
        
        Q_gas_conv = self.h_conv_int * (T_core - T_gas)

        dT_in_dt = (P_in + Q_v_in - Q_lat_in_mid - Q_air_in) / (self.m_rib * self.Cp_rubber)
        dT_mid_dt = (P_mid + Q_v_mid + Q_lat_in_mid - Q_lat_mid_out - Q_air_mid) / (self.m_rib * self.Cp_rubber)
        dT_out_dt = (P_out + Q_v_out + Q_lat_mid_out - Q_air_out) / (self.m_rib * self.Cp_rubber)
        
        dT_core_dt = (P_core_in - Q_v_in - Q_v_mid - Q_v_out - Q_gas_conv) / (self.m_core * self.Cp_rubber)
        dT_gas_dt = Q_gas_conv / (self.m_gas * self.Cv_gas)
        
        return jnp.array([dT_in_dt, dT_mid_dt, dT_out_dt]), dT_core_dt, dT_gas_dt
    
    def compute_transient_slip_derivatives(self, alpha_ss, kappa_ss, alpha_transient, kappa_transient, Fz, Vx):
        """
        Calculates the derivatives for the transient slip states based on the 
        contact patch relaxation length.
        
        alpha_ss, kappa_ss: The steady-state kinematic slip (input from suspension).
        alpha_transient, kappa_transient: The current delayed slip state of the tire carcass.
        """
        # Relaxation lengths (meters). Distance the tire must roll to build 63% of steady-state force.
        # Typically, longitudinal relaxation is shorter than lateral.
        Lx = self.coeffs.get('relaxation_length_x', 0.1)
        Ly = self.coeffs.get('relaxation_length_y', 0.25)
        
        # Prevent division by zero at low speeds by capping the relaxation time constant
        Vx_safe = jnp.maximum(jnp.abs(Vx), 0.1)
        
        # At very low speeds, the relaxation length effectively drops to zero 
        # (force builds instantly with static friction). We use a sigmoid transition.
        low_speed_attenuation = jnp.maximum(0.1, jnp.tanh(Vx_safe / 1.0))
        Lx_eff = Lx * low_speed_attenuation
        Ly_eff = Ly * low_speed_attenuation
        
        # First-order lag dynamics: dx/dt = (x_steady - x_current) * (V / RelaxationLength)
        d_alpha_dt = (Vx_safe / Ly_eff) * (alpha_ss - alpha_transient)
        d_kappa_dt = (Vx_safe / Lx_eff) * (kappa_ss - kappa_transient)
        
        # If the tire is not loaded, it cannot hold a transient twist state, so it snaps 
        # instantly back to the kinematic state.
        is_loaded = jax.nn.sigmoid(Fz - 100.0) # Soft switch around 100N
        d_alpha_dt = jnp.where(is_loaded > 0.5, d_alpha_dt, (alpha_ss - alpha_transient) / 0.01)
        d_kappa_dt = jnp.where(is_loaded > 0.5, d_kappa_dt, (kappa_ss - kappa_transient) / 0.01)
        
        return d_alpha_dt, d_kappa_dt