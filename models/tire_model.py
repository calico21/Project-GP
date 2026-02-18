import numpy as np
import casadi as ca

class PacejkaTire:
    """
    Advanced Pacejka Magic Formula 5.2 Tire Model.
    Includes:
    - Combined Slip (Interactions between steering and braking)
    - Load Sensitivity (Grip is not linear with load)
    - Thermal Dynamics (Tire heats up with sliding)
    - Thermal Degradation (Grip loss when hot/cold)
    
    COMPATIBILITY NOTE:
    Handles both Float (Simulation) and CasADi Symbol (OCP) inputs.
    Uses Smooth Approximations for OCP stability.
    """
    def __init__(self, tire_coeffs):
        self.coeffs = tire_coeffs
        
        # --- THERMAL DEFAULTS ---
        self.T_opt = self.coeffs.get('T_opt', 90.0)      
        self.C_therm = self.coeffs.get('C_therm', 0.002) 
        self.C_heat = self.coeffs.get('C_heat', 1.5)     
        self.C_cool = self.coeffs.get('C_cool', 0.45)    
        self.mass = self.coeffs.get('mass', 10.0)        
        self.Cp = self.coeffs.get('Cp', 1100.0)          
        self.T_env = 25.0                                

    def _is_symbolic(self, var):
        """Check if a variable is a CasADi symbol."""
        return isinstance(var, (ca.SX, ca.MX))

    def _smooth_abs(self, x):
        """
        Differentiable approximation of abs(x).
        sqrt(x^2 + eps) - prevents 'Restoration Failed' errors.
        """
        return ca.sqrt(x**2 + 1e-6)

    def _smooth_max(self, a, b):
        return 0.5 * (a + b + ca.sqrt((a - b)**2 + 1e-6))

    def _smooth_min(self, a, b):
        return 0.5 * (a + b - ca.sqrt((a - b)**2 + 1e-6))

    def compute_force(self, alpha, kappa, Fz, Vx, T_tire=60.0):
        """
        Calculates tire forces (Fx, Fy) using smooth math for the OCP.
        """
        # Detect if we need Symbolic Math (OCP) or Standard Math (Sim)
        if self._is_symbolic(alpha) or self._is_symbolic(Fz) or self._is_symbolic(T_tire):
            _sin = ca.sin
            _arctan = ca.arctan
            _sqrt = ca.sqrt
            # Use Smooth Approx for OCP
            _abs = self._smooth_abs 
            _max = self._smooth_max
            _min = self._smooth_min
        else:
            _sin = np.sin
            _arctan = np.arctan
            _sqrt = np.sqrt
            _abs = abs
            _max = max
            _min = min

        # 1. Parameter Extraction
        dy = self.coeffs.get('Dy', 1.3)
        cy = self.coeffs.get('Cy', 1.5)
        by = self.coeffs.get('By', 10.0)
        ey = self.coeffs.get('Ey', -1.0)
        
        dx = self.coeffs.get('Dx', 1.35)
        cx = self.coeffs.get('Cx', 1.6)
        bx = self.coeffs.get('Bx', 12.0)
        ex = self.coeffs.get('Ex', -0.5)

        # 2. Load Sensitivity
        Fz_nom = 4000.0
        d_fz = (Fz - Fz_nom) / Fz_nom
        lambda_mu_y = 1.0 - 0.1 * d_fz 
        lambda_mu_x = 1.0 - 0.08 * d_fz
        
        # 3. Thermal Sensitivity
        therm_factor = 1.0 - self.C_therm * (T_tire - self.T_opt)**2
        therm_factor = _max(0.5, _min(1.0, therm_factor))

        mu_y = dy * lambda_mu_y * therm_factor
        mu_x = dx * lambda_mu_x * therm_factor

        # 4. Pure Slip Magic Formula
        Shy = 0.0 
        Svy = 0.0 
        alpha_y = alpha + Shy
        Fy_pure = Fz * mu_y * _sin(cy * _arctan(by * alpha_y - ey * (by * alpha_y - _arctan(by * alpha_y)))) + Svy
        
        kappa_x = kappa
        Fx_pure = Fz * mu_x * _sin(cx * _arctan(bx * kappa_x - ex * (bx * kappa_x - _arctan(bx * kappa_x))))

        # 5. Combined Slip (Friction Circle)
        # Use _smooth_abs() to handle the "zero crossing" safely
        safe_fx = _abs(Fx_pure) + 1e-6
        safe_fy = _abs(Fy_pure) + 1e-6
        
        rho = _sqrt((Fx_pure / safe_fx)**2 + (Fy_pure / safe_fy)**2)
        
        # Symbolic branching logic for rho > 1.0
        if self._is_symbolic(rho):
            Fx = ca.if_else(rho > 1.0, Fx_pure / rho, Fx_pure)
            Fy = ca.if_else(rho > 1.0, Fy_pure / rho, Fy_pure)
        else:
            if rho > 1.0:
                Fx = Fx_pure / rho
                Fy = Fy_pure / rho
            else:
                Fx = Fx_pure
                Fy = Fy_pure

        return Fx, Fy

    def compute_thermal_dynamics(self, Fx, Fy, alpha, kappa, Vx, T_curr):
        """
        Calculates dT/dt [K/s] using smooth math.
        """
        # Detect mode
        if self._is_symbolic(Fx) or self._is_symbolic(alpha):
            _abs = self._smooth_abs
            _tan = ca.tan
        else:
            _abs = abs
            _tan = np.tan
            
        # 1. Sliding Velocities
        V_sy = Vx * _tan(alpha)
        V_sx = Vx * kappa
        
        # 2. Heat Generation
        P_gen_lat = _abs(Fy * V_sy)
        P_gen_long = _abs(Fx * V_sx)
        P_total = (P_gen_lat + P_gen_long) * self.C_heat

        # 3. Cooling
        cooling_rate = self.C_cool * (_abs(Vx) + 1.0) * (T_curr - self.T_env)
        
        # 4. Net Flux
        Q_net = P_total - cooling_rate
        
        # 5. Temperature Rate
        dT_dt = Q_net / (self.mass * self.Cp)
        
        return dT_dt

    def get_peak_slip(self, Fz):
        return 0.12