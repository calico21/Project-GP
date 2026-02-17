import numpy as np
import casadi as ca

class PacejkaTire:
    def __init__(self, params):
        """
        Initialize the Advanced Pacejka 5.2 Tire Model with Thermal & Transient features.
        
        Args:
            params (dict): Dictionary containing the MF coefficients.
                           Expects standard Pacejka keys (PCX1, PDX1, etc.).
                           Optional Thermal keys: 'T_OPT', 'C_HEAT', 'K_COOL'.
        """
        self.p = params
        
        # --- Thermal Parameters (Defaults if not in yaml) ---
        self.T_opt = self.p.get('T_OPT', 60.0)      # Optimal Temp [C]
        self.T_range = self.p.get('T_WIDTH', 20.0)  # Width of grip window
        self.Cp = self.p.get('C_HEAT', 800.0)       # Specific Heat Capacity [J/kgK] * Mass [kg]
        self.hA = self.p.get('K_COOL', 12.0)        # Convective Cooling Coeff [W/K]

    def compute_force(self, alpha, kappa, Fz, Vx=None, T_tire=None, mu_scale=1.0, use_combined=True):
        """
        Master function to calculate tire forces with optional advanced physics.
        
        Args:
            alpha: Slip angle [rad]
            kappa: Slip ratio [-]
            Fz: Vertical load [N]
            Vx: Longitudinal Velocity [m/s] (Required for Thermal/Relaxation)
            T_tire: Current Tire Temp [C] (If None, ignores thermal degradation)
            mu_scale: External friction scaler (e.g. wet track)
            use_combined: If True, applies Combined Slip interactions (Friction Circle)
            
        Returns:
            Fx, Fy: Final forces [N]
        """
        # 1. Detect Input Type (Symbolic vs Numeric)
        if isinstance(alpha, (ca.SX, ca.MX)) or isinstance(Fz, (ca.SX, ca.MX)):
            sin, cos, atan, sqrt, exp = ca.sin, ca.cos, ca.atan, ca.sqrt, ca.exp
            fmax, fmin = ca.fmax, ca.fmin
        else:
            sin, cos, atan, sqrt, exp = np.sin, np.cos, np.arctan, np.sqrt, np.exp
            fmax, fmin = np.maximum, np.minimum

        # 2. Safety Clamps (Prevent Divide by Zero)
        Fz = fmax(10.0, Fz)  # Minimum 10N load
        
        # 3. Thermal Scaling
        # If T_tire is provided, degrade mu based on deviation from T_opt
        mu_thermal = 1.0
        if T_tire is not None:
            mu_thermal = self._get_thermal_scaling(T_tire, exp)
        
        total_mu_scale = mu_scale * mu_thermal

        # 4. Pure Slip Calculation (The "Ideal" forces)
        Fx0, Fy0, mux, muy = self._compute_pure_slip(alpha, kappa, Fz, total_mu_scale, sin, cos, atan, fmin)

        # 5. Combined Slip Calculation (Interaction)
        if use_combined:
            Fx, Fy = self._apply_combined_slip(Fx0, Fy0, alpha, kappa, mux, muy, Fz, sin, cos, atan)
        else:
            Fx, Fy = Fx0, Fy0

        return Fx, Fy

    def compute_thermal_dynamics(self, Fx, Fy, alpha, kappa, Vx, T_curr, T_env=25.0):
        """
        Calculates the derivative of Tire Temperature (dT/dt).
        Intended for use in the OCP State Equations.
        
        Model:
            dT/dt = (Power_Heat - Power_Cool) / Cp
            Power_Heat = F_sliding * V_sliding
            Power_Cool = hA * (T - T_env)
        """
        # Sliding Velocities
        # V_sx = Vx * kappa
        # V_sy = Vx * tan(alpha)  (approx for small angles)
        
        # Note: Using absolute values for heating power
        if isinstance(Fx, (ca.SX, ca.MX)):
            fabs, tan = ca.fabs, ca.tan
        else:
            fabs, tan = np.abs, np.tan

        P_heat_x = fabs(Fx * Vx * kappa)
        P_heat_y = fabs(Fy * Vx * tan(alpha))
        P_heat = P_heat_x + P_heat_y
        
        P_cool = self.hA * (T_curr - T_env)
        
        dTdt = (P_heat - P_cool) / self.Cp
        return dTdt

    def get_relaxation_length(self, Fz):
        """
        Returns the Relaxation Length (Sigma) based on load.
        Used for Transient Dynamics (lagged slip).
        
        Sigma = sigma_0 * (1 + k * dFz)
        """
        # Default Parameters
        sigma0 = self.p.get('PTX1', 0.15)  # Base relaxation length [m] (approx contact patch length)
        
        # Simple load dependence
        # As load increases, contact patch grows -> relaxation length increases
        Fz0 = self.p.get('FNOMIN', 1000)
        return sigma0 * (Fz / Fz0)

    # ================= INTERNAL PHYSICS METHODS =================

    def _get_thermal_scaling(self, T, exp_func):
        """Gaussian-like drop-off for friction vs Temperature."""
        # mu = 1 - k * (T - T_opt)^2
        # Using Gaussian: exp( - (T-Topt)^2 / (2*width^2) )
        delta_T = T - self.T_opt
        sensitivity = 1.0 / (2 * self.T_range**2)
        return exp_func(-sensitivity * delta_T**2)

    def _compute_pure_slip(self, alpha, kappa, Fz, mu_scale, sin, cos, atan, fmin):
        """Standard Pacejka 5.2 Pure Slip Formulas."""
        Fz0 = self.p.get('FNOMIN', 1000)
        dfz = (Fz - Fz0) / Fz0

        # --- Longitudinal (Fx) ---
        Cx = self.p.get('PCX1', 1.6)
        mux = (self.p.get('PDX1', 1.2) + self.p.get('PDX2', -0.1) * dfz) * mu_scale
        Dx = mux * Fz
        Ex = (self.p.get('PEX1', 0.5) + self.p.get('PEX2', 0.0) * dfz + self.p.get('PEX3', 0.0) * dfz**2)
        Ex = fmin(1.0, Ex)
        Kx = Fz * (self.p.get('PKX1', 25) + self.p.get('PKX2', 0.0) * dfz)
        Bx = Kx / (Cx * Dx + 1e-6)
        
        Shx = self.p.get('PHX1', 0.0)
        Svx = self.p.get('PVX1', 0.0)
        kappa_x = kappa + Shx
        Bx_kappa = Bx * kappa_x
        Fx0 = Dx * sin(Cx * atan(Bx_kappa - Ex * (Bx_kappa - atan(Bx_kappa)))) + Svx

        # --- Lateral (Fy) ---
        Cy = self.p.get('PCY1', 1.3)
        muy = (self.p.get('PDY1', 1.6) + self.p.get('PDY2', -0.15) * dfz) * mu_scale
        Dy = muy * Fz
        Ey = (self.p.get('PEY1', -0.5) + self.p.get('PEY2', 0.0) * dfz)
        Ey = fmin(1.0, Ey)
        Ky = Fz * (self.p.get('PKY1', 15) + self.p.get('PKY2', 0.0) * dfz)
        By = Ky / (Cy * Dy + 1e-6)
        
        Shy = self.p.get('PHY1', 0.0)
        Svy = self.p.get('PVY1', 0.0)
        alpha_y = alpha + Shy
        By_alpha = By * alpha_y
        Fy0 = Dy * sin(Cy * atan(By_alpha - Ey * (By_alpha - atan(By_alpha)))) + Svy

        return Fx0, Fy0, mux, muy

    def _apply_combined_slip(self, Fx0, Fy0, alpha, kappa, mux, muy, Fz, sin, cos, atan):
        """
        Applies Cosine Weighting Factors for Combined Slip.
        (Based on Pacejka simplified interaction)
        """
        # --- Combined Fx (effect of alpha on Fx) ---
        # Gxa = cos( Cxa * atan( Bxa * alpha ) )
        rBx1 = self.p.get('RBX1', 12.0)
        rCx1 = self.p.get('RCX1', 1.0)
        alpha_s = alpha # In complex models, this is shifted
        Gxa = cos(rCx1 * atan(rBx1 * alpha_s))
        Fx = Fx0 * Gxa

        # --- Combined Fy (effect of kappa on Fy) ---
        # Gyk = cos( Cyk * atan( Byk * kappa ) )
        rBy1 = self.p.get('RBY1', 10.0)
        rCy1 = self.p.get('RCY1', 1.0)
        kappa_s = kappa
        Gyk = cos(rCy1 * atan(rBy1 * kappa_s))
        Fy = Fy0 * Gyk
        
        return Fx, Fy