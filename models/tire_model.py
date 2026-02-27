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
    and stochastic grip limits without any CasADi overhead. Upgraded to full MF6.2.

    Part 8 change: PINN residual corrections (mods) are now clipped to ±0.25
    before being applied to Fx/Fy, preventing blow-up from an untrained network.
    """
    def __init__(self, tire_coeffs, rng_seed=42):
        self.coeffs = tire_coeffs
        
        self.T_opt = self.coeffs.get('T_opt',
                     self.coeffs.get('T_OPT', 90.0))  
        self.T_env = 25.0
        self.T_track = 40.0  
        self.P_nom = self.coeffs.get('P_nom', 1.2)
        
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
        
        return jnp.clip(T_flash, 0.0, 350.0)

    def compute_force(self, alpha, kappa, Fz, gamma, T_ribs, T_gas, Vx=15.0, stochastic_key=None):
        """
        Full Pacejka MF6.2 lateral and longitudinal force with:
        - Complete coefficient set including plysteer/conicity (PHY1/PHY2, PVY1/PVY2)
        - Camber sensitivity on cornering stiffness (PKY3) and peak friction (PDY3)
        - Thermal grip factor from 5-node model
        - Pressure correction from gas temperature (Gay-Lussac)
        - Combined slip reductions via Gyk and Gxa
        - PINN/GP residual corrections clipped to ±0.25 (Part 8)
        Returns (Fx, Fy) in Newtons. Sign convention: positive Fy = left force.
        """
        eps = 1e-6

        # ── Thermal and pressure corrections ────────────────────────────────────
        T_eff        = T_ribs[0] * 0.333 + T_ribs[1] * 0.334 + T_ribs[2] * 0.333
        kappa_c      = jnp.clip(kappa, -0.5, 0.5)
        tan_a        = jnp.sin(alpha) / (jnp.cos(alpha) + eps)
        V_slide      = Vx * jnp.sqrt(kappa_c**2 + tan_a**2 + eps)
        
        # Estimate base friction roughly for flash temp
        mu_base_est  = 1.5 
        flash_dt     = self.compute_flash_temperature(mu_base_est, Fz, V_slide)
        T_contact    = T_eff + flash_dt
        T_opt        = self.coeffs.get('T_opt', 90.0)
        therm_factor = jnp.maximum(0.30,
                       jnp.minimum(1.0, 1.0 - 0.003 * (T_contact - T_opt)**2))

        # Pressure from gas temperature via Gay-Lussac's law: P ∝ T (constant volume)
        T_gas_K  = T_gas + 273.15
        T_env_K  = self.T_env + 273.15
        P_dyn    = self.P_nom * (T_gas_K / (T_env_K + eps))
        dP       = P_dyn - (self.P_nom + 0.2)
        lam_p    = jnp.clip(1.0 - 0.15 * dP**2, 0.6, 1.0)

        lam_muy  = therm_factor * lam_p
        lam_mux  = therm_factor * lam_p

        # ── Reference load ───────────────────────────────────────────────────────
        Fz0  = self.coeffs.get('FNOMIN', 1000.0)
        dfz  = (Fz - Fz0) / jnp.maximum(Fz0, eps)
        gam  = jnp.clip(gamma, -0.35, 0.35)     # radians — clip for safety

        # ════════════════════════════════════════════════════════════════════════
        # PURE LATERAL FORCE  (MF6.2)
        # Plysteer: SHy from PHY1/PHY2 shifts the slip angle origin (conicity effect)
        # Conicity: SVy from PVY1/PVY2 adds a load-dependent vertical force offset
        # Camber sensitivity: PDY3 on peak friction, PKY3 on cornering stiffness
        # ════════════════════════════════════════════════════════════════════════
        c  = self.coeffs
        PCY1 = c.get('PCY1',  1.338)
        PDY1 = c.get('PDY1',  1.0  )
        PDY2 = c.get('PDY2', -0.084)
        PDY3 = c.get('PDY3',  0.265)
        PEY1 = c.get('PEY1', -0.342)
        PEY2 = c.get('PEY2', -0.122)
        PEY3 = c.get('PEY3',  0.0  )
        PEY4 = c.get('PEY4',  0.0  )
        PKY1 = c.get('PKY1', 15.324)
        PKY2 = c.get('PKY2',  1.715)
        PKY3 = c.get('PKY3',  0.370)
        PKY4 = c.get('PKY4',  2.0  )
        PHY1 = c.get('PHY1', -0.0009)
        PHY2 = c.get('PHY2', -0.00082)
        PVY1 = c.get('PVY1',  0.045)
        PVY2 = c.get('PVY2', -0.024)

        SHy    = PHY1 + PHY2 * dfz                           # plysteer horizontal shift
        SVy    = Fz * (PVY1 + PVY2 * dfz) * lam_muy          # conicity vertical shift

        Ky     = (PKY1 * Fz0
                  * jnp.sin(PKY4 * jnp.arctan(Fz / jnp.maximum(PKY2 * Fz0, eps)))
                  * (1.0 - PKY3 * jnp.abs(gam)))              # camber sensitivity on Ky
        Dy     = PDY1 * (1.0 + PDY2 * dfz) * (1.0 - PDY3 * gam**2) * Fz * lam_muy  # camber on peak
        Cy     = PCY1
        By     = Ky / jnp.maximum(Cy * Dy, eps)

        a_s    = alpha + SHy
        sgn_as = jnp.where(a_s >= 0.0, 1.0, -1.0)
        Ey     = jnp.clip(
                     (PEY1 + PEY2 * dfz)
                     * (1.0 - (PEY3 + PEY4 * gam) * sgn_as),
                     -10.0, 1.0)
        x_y    = By * a_s
        Fy0    = Dy * jnp.sin(Cy * jnp.arctan(x_y - Ey * (x_y - jnp.arctan(x_y)))) + SVy

        # ════════════════════════════════════════════════════════════════════════
        # PURE LONGITUDINAL FORCE  (MF6.2)
        # ════════════════════════════════════════════════════════════════════════
        PCX1 = c.get('PCX1',  1.579)
        PDX1 = c.get('PDX1',  1.0  )
        PDX2 = c.get('PDX2', -0.041)
        PEX1 = c.get('PEX1',  0.312)
        PEX2 = c.get('PEX2', -0.261)
        PEX3 = c.get('PEX3',  0.0  )
        PKX1 = c.get('PKX1', 21.687)
        PKX2 = c.get('PKX2', 13.728)
        PKX3 = c.get('PKX3', -0.466)
        PHX1 = c.get('PHX1',  0.0  )
        PHX2 = c.get('PHX2',  0.0  )
        PVX1 = c.get('PVX1',  0.0  )
        PVX2 = c.get('PVX2',  0.0  )

        SHx    = PHX1 + PHX2 * dfz
        SVx    = Fz * (PVX1 + PVX2 * dfz) * lam_mux
        Kx     = Fz * (PKX1 + PKX2 * dfz) * jnp.exp(PKX3 * dfz)
        Dx     = PDX1 * (1.0 + PDX2 * dfz) * Fz * lam_mux
        Cx     = PCX1
        Bx     = Kx / jnp.maximum(Cx * Dx, eps)
        Ex     = jnp.clip(PEX1 + PEX2 * dfz + PEX3 * dfz**2, -10.0, 1.0)
        k_s    = kappa_c + SHx
        x_x    = Bx * k_s
        Fx0    = Dx * jnp.sin(Cx * jnp.arctan(x_x - Ex * (x_x - jnp.arctan(x_x)))) + SVx

        # ════════════════════════════════════════════════════════════════════════
        # COMBINED SLIP  (MF6.2 Gyk and Gxa)
        # Gyk: lateral force reduction factor under longitudinal slip
        # Gxa: longitudinal force reduction factor under lateral slip
        # ════════════════════════════════════════════════════════════════════════
        RBY1 = c.get('RBY1',  7.143); RBY2 = c.get('RBY2', 9.192)
        RBY3 = c.get('RBY3',  0.0  ); RCY1 = c.get('RCY1', 1.059)
        REY1 = c.get('REY1', -0.496); REY2 = c.get('REY2', 0.0  )
        RHY1 = c.get('RHY1',  0.0095); RHY2 = c.get('RHY2', 0.0098)
        RVY1 = c.get('RVY1',  0.052); RVY2 = c.get('RVY2',  0.0455)
        RVY3 = c.get('RVY3', -0.025); RVY4 = c.get('RVY4', 12.12)
        RVY5 = c.get('RVY5',  1.9  ); RVY6 = c.get('RVY6', 22.21)

        Byk   = RBY1 * jnp.cos(jnp.arctan(RBY2 * (alpha - RBY3)))
        Cyk   = RCY1
        Eyk   = jnp.clip(REY1 + REY2 * dfz, -10.0, 1.0)
        SHyk  = RHY1 + RHY2 * dfz
        DVyk  = (Dy * (RVY1 + RVY2 * dfz + RVY3 * gam)
                 * jnp.cos(jnp.arctan(RVY4 * alpha)))
        SVyk  = DVyk * jnp.sin(RVY5 * jnp.arctan(RVY6 * kappa_c))
        k_cs  = kappa_c + SHyk

        def _g(B, C, E, x):
            return jnp.cos(C * jnp.arctan(x - E * (x - jnp.arctan(x))))

        Gyk0  = _g(Byk, Cyk, Eyk, Byk * SHyk)
        Gyk   = jnp.clip(_g(Byk, Cyk, Eyk, Byk * k_cs)
                         / jnp.maximum(jnp.abs(Gyk0), eps), 0.0, 1.0)
        Fy    = Gyk * Fy0 + SVyk

        RBX1 = c.get('RBX1', 13.046); RBX2 = c.get('RBX2',  9.718)
        RCX1 = c.get('RCX1',  0.9995); REX1 = c.get('REX1', 0.0  )
        REX2 = c.get('REX2',  0.0  ); RHX1 = c.get('RHX1', 0.0  )

        Bxa   = RBX1 * jnp.cos(jnp.arctan(RBX2 * kappa_c))
        Cxa   = RCX1
        Exa   = jnp.clip(REX1 + REX2 * dfz, -10.0, 1.0)
        SHxa  = RHX1
        a_cs  = alpha + SHxa
        Gxa0  = _g(Bxa, Cxa, Exa, Bxa * SHxa)
        Gxa   = jnp.clip(_g(Bxa, Cxa, Exa, Bxa * a_cs)
                         / jnp.maximum(jnp.abs(Gxa0), eps), 0.0, 1.0)
        Fx    = Gxa * Fx0

        # ── PINN/GP residual hook ─────────────────────────────────────────────
        # Part 8: Clip mods to ±0.25 before applying to dynamics.
        # This prevents an untrained or diverging network from producing
        # unphysical force corrections (e.g. mods = ±3 → Fy flips sign).
        state_in = jnp.array([alpha, kappa_c, gam, Fz, Vx])
        mods, sigma = self.operator.apply(state_in, stochastic_key)
        mods     = jnp.clip(mods, -0.25, 0.25)   # ← Part 8 gradient clipping
        penalty  = 2.0 * sigma
        Fy = Fy * (1.0 + mods[1] - penalty)
        Fx = Fx * (1.0 + mods[0] - penalty)

        return Fx, Fy

    def compute_aligning_torque(self, alpha, kappa, Fz, gamma, Fy, Fx=0.0):
        """
        Pacejka MF6.2 aligning torque Mz.
        Call after compute_force — pass the Fy returned from that call.
        Returns Mz in Newton-metres. Positive = nose-right restoring moment.
        """
        eps  = 1e-6
        c    = self.coeffs
        Fz0  = c.get('FNOMIN', 1000.0)
        R0   = c.get('R0', 0.2032)
        dfz  = (Fz - Fz0) / jnp.maximum(Fz0, eps)
        gam  = jnp.clip(gamma, -0.35, 0.35)

        QBZ1 = c.get('QBZ1', 10.904); QBZ2 = c.get('QBZ2', -1.896)
        QBZ3 = c.get('QBZ3', -0.937); QBZ4 = c.get('QBZ4',  0.100)
        QBZ5 = c.get('QBZ5', -0.100); QCZ1 = c.get('QCZ1',  1.180)
        QDZ1 = c.get('QDZ1',  0.092); QDZ2 = c.get('QDZ2', -0.006)
        QDZ3 = c.get('QDZ3',  0.0  ); QDZ4 = c.get('QDZ4',  0.0  )
        QEZ1 = c.get('QEZ1', -8.865); QEZ2 = c.get('QEZ2',  0.0  )
        QEZ3 = c.get('QEZ3',  0.0  ); QEZ4 = c.get('QEZ4',  0.254)
        QEZ5 = c.get('QEZ5',  0.0  )
        QHZ1 = c.get('QHZ1',  0.0065); QHZ2 = c.get('QHZ2', 0.0056)

        SHt   = QHZ1 + QHZ2 * dfz
        a_t   = alpha + SHt

        Bt    = ((QBZ1 + QBZ2 * dfz + QBZ3 * dfz**2)
                 * (1.0 + QBZ4 * gam + QBZ5 * jnp.abs(gam)))
        Ct    = QCZ1
        Dt    = (Fz * (QDZ1 + QDZ2 * dfz)
                 * (1.0 + QDZ3 * gam + QDZ4 * gam**2)
                 * (R0 / jnp.maximum(Fz0, eps)))
        x_ez  = Bt * Ct * a_t
        Et    = jnp.clip(
                    (QEZ1 + QEZ2 * dfz + QEZ3 * dfz**2)
                    * (1.0 + (QEZ4 + QEZ5 * gam) * (2.0 / jnp.pi) * jnp.arctan(x_ez)),
                    -10.0, 1.0)

        x_t   = Bt * a_t
        t     = Dt * jnp.cos(Ct * jnp.arctan(x_t - Et * (x_t - jnp.arctan(x_t)))) \
                * jnp.cos(alpha)
        Mz    = -t * Fy
        return Mz

    def compute_transient_slip_derivatives(self, alpha_ss, kappa_ss, alpha_t, kappa_t, Fz, Vx):
        """
        First-order carcass lag with load-dependent relaxation lengths.
        Lx(Fz) = Lx0 * sqrt(Fz / Fz0)   — matches Part 1 requirement.
        """
        eps   = 1e-6
        Fz0   = self.coeffs.get('FNOMIN',           1000.0)
        Lx0   = self.coeffs.get('relaxation_length_x', 0.10)
        Ly0   = self.coeffs.get('relaxation_length_y', 0.25)

        Fz_s  = jnp.maximum(Fz, 50.0)                    # avoid sqrt(0)
        Lx    = Lx0 * jnp.sqrt(Fz_s / Fz0)
        Ly    = Ly0 * jnp.sqrt(Fz_s / Fz0)

        Vx_s  = jnp.maximum(jnp.abs(Vx), 0.1)
        # Smooth low-speed attenuation — prevents instability below ~1 m/s
        atten = jnp.tanh(Vx_s / 1.0)
        Lx_e  = jnp.maximum(Lx * atten, 0.01)
        Ly_e  = jnp.maximum(Ly * atten, 0.01)

        d_alpha = (Vx_s / Ly_e) * (alpha_ss - alpha_t)
        d_kappa = (Vx_s / Lx_e) * (kappa_ss - kappa_t)

        # When tyre is unloaded, snap to steady-state instantly
        loaded  = jax.nn.sigmoid((Fz - 50.0) / 50.0)
        d_alpha = jnp.where(loaded > 0.5, d_alpha, (alpha_ss - alpha_t) / 0.005)
        d_kappa = jnp.where(loaded > 0.5, d_kappa, (kappa_ss - kappa_t) / 0.005)

        return d_alpha, d_kappa

    def compute_thermal_derivatives(self, T_ribs, T_gas, Fz, V_slide, T_env=20.0):
        """
        5-node thermal model: surface_in, surface_mid, surface_out, core → gas.
        Returns d/dt of [T_rib_in, T_rib_mid, T_rib_out, T_core, T_gas].
        """
        c    = self.coeffs
        m    = c.get('mass',       10.0)
        m_g  = c.get('m_gas',      0.05)
        Cp   = c.get('Cp',       1100.0)
        Cv_g = c.get('Cv_gas',    718.0)
        h    = c.get('h_conv',     50.0)
        h_i  = c.get('h_conv_int', 30.0)
        k    = c.get('k_cond',    150.0)
        k_l  = c.get('k_cond_lat', 85.0)
        A    = c.get('A_surf',      0.08)
        q_r  = c.get('q_roll',     0.03)

        T_in, T_mid, T_out = T_ribs[0], T_ribs[1], T_ribs[2]
        T_c  = T_ribs[3] if T_ribs.shape[0] > 3 else (T_in + T_mid + T_out) / 3.0

        # Frictional heat generated at contact patch
        mu_avg = 1.5      # approximate friction coefficient
        Q_fric = mu_avg * Fz * V_slide * (1.0 - q_r)   # heat into rubber
        Q_roll = mu_avg * Fz * V_slide * q_r           # heat into gas via rolling

        # Lateral conduction between ribs
        Q_lat_in_mid  = k_l * A / 3.0 * (T_in  - T_mid)
        Q_lat_mid_out = k_l * A / 3.0 * (T_mid - T_out)

        # Radial conduction to core
        Q_rad_in  = k * A / 3.0 * (T_in  - T_c)
        Q_rad_mid = k * A / 3.0 * (T_mid - T_c)
        Q_rad_out = k * A / 3.0 * (T_out - T_c)

        # Convection to ambient
        Q_conv_in  = h * A / 3.0 * (T_in  - T_env)
        Q_conv_mid = h * A / 3.0 * (T_mid - T_env)
        Q_conv_out = h * A / 3.0 * (T_out - T_env)

        # Core to gas
        Q_core_gas = h_i * A * (T_c - T_gas)

        node_mass  = m / 4.0
        dT_in      = (Q_fric/3.0 - Q_lat_in_mid  - Q_rad_in  - Q_conv_in ) / (node_mass * Cp)
        dT_mid     = (Q_fric/3.0 - Q_lat_mid_out + Q_lat_in_mid - Q_rad_mid - Q_conv_mid) / (node_mass * Cp)
        dT_out     = (Q_fric/3.0 + Q_lat_mid_out - Q_rad_out - Q_conv_out) / (node_mass * Cp)
        dT_core    = (Q_rad_in + Q_rad_mid + Q_rad_out - Q_core_gas) / (node_mass * Cp)
        dT_gas     = (Q_roll + Q_core_gas) / (m_g * Cv_g)

        return jnp.array([dT_in, dT_mid, dT_out, dT_core, dT_gas])