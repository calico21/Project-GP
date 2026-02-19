import numpy as np
import casadi as ca

class PacejkaTire:
    """
    State-of-the-Art 3-Node (Surface/Core/Gas) Pacejka Tire Model.
    
    Thermal Physics:
    1. Surface: Heated by friction (sliding), cooled by air (convection), stabilized by Core.
       -> Dictates instantaneous GRIP (mu).
    2. Core: Heated by deflection (rolling), exchanges heat with Surface and Gas.
       -> Dictates Structural Stiffness.
    3. Gas (Inflation): Heated by Core (convection).
       -> Dictates internal Pressure, which modulates vertical load sensitivity and footprint.
    """
    def __init__(self, tire_coeffs):
        self.coeffs = tire_coeffs
        
        # --- THERMAL PARAMETERS ---
        self.T_opt = self.coeffs.get('T_opt', 90.0)       # Optimum Grip Temp [C]
        self.T_env = 25.0                                 # Ambient Temp [C]
        self.P_nom = self.coeffs.get('P_nom', 1.2)        # Nominal Cold Pressure [bar]
        
        # Masses & Capacities
        self.m_total = self.coeffs.get('mass', 10.0)
        self.m_surf = self.m_total * 0.10
        self.m_core = self.m_total * 0.90
        self.m_gas = self.coeffs.get('m_gas', 0.05)       # Approx mass of inflation air/nitrogen [kg]
        
        self.Cp_rubber = self.coeffs.get('Cp', 1100.0)    # Heat Capacity Rubber [J/kgK]
        self.Cv_gas = self.coeffs.get('Cv_gas', 718.0)    # Heat Capacity Air (constant vol) [J/kgK]
        
        # Heat Transfer Coefficients
        self.h_conv_ext = self.coeffs.get('h_conv', 50.0) # Convection (Air-Surface) [W/m2K]
        self.k_cond = self.coeffs.get('k_cond', 150.0)    # Conduction (Core-Surface) [W/K]
        self.h_conv_int = self.coeffs.get('h_conv_int', 30.0) # Convection (Core-Gas) [W/K]
        
        self.A_surf = self.coeffs.get('A_surf', 0.8)      # Surface Area [m2]
        self.q_roll = self.coeffs.get('q_roll', 0.03)     # Rolling Resistance Factor (Hysteresis)

    def _is_symbolic(self, var):
        return isinstance(var, (ca.SX, ca.MX))

    def _smooth_abs(self, x):
        return ca.sqrt(x**2 + 1e-6)

    def _smooth_max(self, a, b):
        return 0.5 * (a + b + ca.sqrt((a - b)**2 + 1e-6))

    def _smooth_min(self, a, b):
        return 0.5 * (a + b - ca.sqrt((a - b)**2 + 1e-6))

    def compute_force(self, alpha, kappa, Fz, T_surf, T_gas):
        """
        Calculates tire forces (Fx, Fy) with Pressure-Coupling.
        Grip relies on SURFACE temperature. Load sensitivity relies on GAS pressure.
        """
        if self._is_symbolic(alpha) or self._is_symbolic(Fz) or self._is_symbolic(T_surf):
            _sin, _arctan, _sqrt = ca.sin, ca.arctan, ca.sqrt
            _abs, _max, _min = self._smooth_abs, self._smooth_max, self._smooth_min
        else:
            _sin, _arctan, _sqrt = np.sin, np.arctan, np.sqrt
            _abs, _max, _min = abs, max, min

        # 1. Calculate Dynamic Pressure (Ideal Gas Law Approx)
        # P_dyn / P_nom = (T_gas_K) / (T_env_K)
        T_gas_K = T_gas + 273.15
        T_env_K = self.T_env + 273.15
        P_dyn = self.P_nom * (T_gas_K / T_env_K)
        
        # Pressure Variance Penalty (Optimal pressure is usually slightly above cold)
        dP = P_dyn - (self.P_nom + 0.2) 
        pressure_modifier = 1.0 - 0.15 * (dP**2) # Parabolic drop-off for over/under inflation
        pressure_modifier = _max(0.6, _min(1.0, pressure_modifier))

        # 2. Extract Pacejka Coefficients
        dy = self.coeffs.get('Dy', 1.3)
        cy = self.coeffs.get('Cy', 1.5)
        by = self.coeffs.get('By', 10.0)
        ey = self.coeffs.get('Ey', -1.0)
        
        dx = self.coeffs.get('Dx', 1.35)
        cx = self.coeffs.get('Cx', 1.6)
        bx = self.coeffs.get('Bx', 12.0)
        ex = self.coeffs.get('Ex', -0.5)

        # 3. Load & Pressure Sensitivity
        Fz_nom = 4000.0
        d_fz = (Fz - Fz_nom) / Fz_nom
        # Pressure stiffens the tire, modifying how load affects the contact patch
        lambda_mu_y = (1.0 - 0.1 * d_fz) * pressure_modifier
        lambda_mu_x = (1.0 - 0.08 * d_fz) * pressure_modifier
        
        # 4. Thermal Sensitivity (Surface)
        therm_factor = 1.0 - 0.002 * (T_surf - self.T_opt)**2
        therm_factor = _max(0.5, _min(1.0, therm_factor))

        mu_y = dy * lambda_mu_y * therm_factor
        mu_x = dx * lambda_mu_x * therm_factor

        # 5. Pure Slip Magic Formula
        Fy_pure = Fz * mu_y * _sin(cy * _arctan(by * alpha - ey * (by * alpha - _arctan(by * alpha))))
        Fx_pure = Fz * mu_x * _sin(cx * _arctan(bx * kappa - ex * (bx * kappa - _arctan(bx * kappa))))

        # 6. Combined Slip (Friction Circle)
        safe_fx = _abs(Fx_pure) + 1e-6
        safe_fy = _abs(Fy_pure) + 1e-6
        rho = _sqrt((Fx_pure / safe_fx)**2 + (Fy_pure / safe_fy)**2)
        
        scale_factor = ca.if_else(rho > 1.0, 1.0 / rho, 1.0) if self._is_symbolic(rho) else (1.0 / rho if rho > 1.0 else 1.0)
        
        Fx = Fx_pure * scale_factor
        Fy = Fy_pure * scale_factor

        return Fx, Fy

    def compute_thermal_dynamics(self, Fx, Fy, Fz, alpha, kappa, Vx, T_core, T_surf, T_gas):
        """
        3-Node Thermal ODEs.
        Returns: [dT_core/dt, dT_surf/dt, dT_gas/dt]
        """
        if self._is_symbolic(Fx) or self._is_symbolic(alpha):
            _abs, _tan = self._smooth_abs, ca.tan
        else:
            _abs, _tan = abs, np.tan
            
        # Velocities
        V_sy = Vx * _tan(alpha)
        V_sx = Vx * kappa
        
        # Power Generation
        P_surf_in = _abs(Fx * V_sx) + _abs(Fy * V_sy) # Friction -> Surface
        P_core_in = self.q_roll * _abs(Fz) * _abs(Vx) # Deflection -> Core

        # Heat Transfer Fluxes
        Q_cond_core_surf = self.k_cond * (T_surf - T_core)
        Q_conv_core_gas = self.h_conv_int * (T_core - T_gas)
        
        h_dynamic = self.h_conv_ext * (1.0 + 0.5 * _abs(Vx))
        Q_conv_surf_air = h_dynamic * self.A_surf * (T_surf - self.T_env)
        
        # Differential Equations (Energy Balance)
        dT_core_dt = (P_core_in + Q_cond_core_surf - Q_conv_core_gas) / (self.m_core * self.Cp_rubber)
        dT_surf_dt = (P_surf_in - Q_cond_core_surf - Q_conv_surf_air) / (self.m_surf * self.Cp_rubber)
        dT_gas_dt = Q_conv_core_gas / (self.m_gas * self.Cv_gas)
        
        return dT_core_dt, dT_surf_dt, dT_gas_dt