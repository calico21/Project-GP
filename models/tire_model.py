import numpy as np

class PacejkaTire:
    """
    Advanced Pacejka Magic Formula 5.2 Tire Model.
    Includes:
    - Combined Slip (Interactions between steering and braking)
    - Load Sensitivity (Grip is not linear with load)
    - Thermal Dynamics (Tire heats up with sliding)
    - Thermal Degradation (Grip loss when hot/cold)
    """
    def __init__(self, tire_coeffs):
        self.coeffs = tire_coeffs
        
        # --- THERMAL DEFAULTS ---
        # If coefficients are missing from config, use these GT3-style defaults
        self.T_opt = self.coeffs.get('T_opt', 90.0)      # Optimal Temp [C]
        self.C_therm = self.coeffs.get('C_therm', 0.002) # Grip loss per degree^2
        self.C_heat = self.coeffs.get('C_heat', 1.5)     # Heating efficiency
        self.C_cool = self.coeffs.get('C_cool', 0.45)    # Cooling coefficient
        self.mass = self.coeffs.get('mass', 10.0)        # Tire thermal mass [kg]
        self.Cp = self.coeffs.get('Cp', 1100.0)          # Heat capacity [J/kgK]
        self.T_env = 25.0                                # Air Temp [C]

    def compute_force(self, alpha, kappa, Fz, Vx, T_tire=60.0):
        """
        Calculates tire forces (Fx, Fy) based on slip and CURRENT TEMPERATURE.
        """
        # 1. Parameter Extraction (Simplified MF5.2)
        # Lateral
        dy = self.coeffs.get('Dy', 1.3)  # Peak friction
        cy = self.coeffs.get('Cy', 1.5)  # Shape factor
        by = self.coeffs.get('By', 10.0) # Stiffness
        ey = self.coeffs.get('Ey', -1.0) # Curvature
        
        # Longitudinal
        dx = self.coeffs.get('Dx', 1.35)
        cx = self.coeffs.get('Cx', 1.6)
        bx = self.coeffs.get('Bx', 12.0)
        ex = self.coeffs.get('Ex', -0.5)

        # 2. Load Sensitivity
        # Grip decreases as load increases (The reason downforce cars need big tires)
        Fz_nom = 4000.0
        d_fz = (Fz - Fz_nom) / Fz_nom
        lambda_mu_y = 1.0 - 0.1 * d_fz 
        lambda_mu_x = 1.0 - 0.08 * d_fz
        
        # 3. THERMAL SENSITIVITY (The "Smart" Upgrade)
        # Parabolic drop-off around T_opt
        # Mu_factor = 1.0 at T_opt, drops to ~0.9 at T_opt +/- 20C
        therm_factor = 1.0 - self.C_therm * (T_tire - self.T_opt)**2
        
        # Clamp thermal factor to avoid nonsensical physics (e.g. negative friction)
        therm_factor = max(0.5, min(1.0, therm_factor))

        # Apply sensitivities
        mu_y = dy * lambda_mu_y * therm_factor
        mu_x = dx * lambda_mu_x * therm_factor

        # 4. Pure Slip Magic Formula
        # Lateral
        # alpha_eq: equivalent slip angle
        Shy = 0.0 # Curvature shift (simplified)
        Svy = 0.0 # Vertical shift
        alpha_y = alpha + Shy
        Fy_pure = Fz * mu_y * np.sin(cy * np.arctan(by * alpha_y - ey * (by * alpha_y - np.arctan(by * alpha_y)))) + Svy
        
        # Longitudinal
        kappa_x = kappa
        Fx_pure = Fz * mu_x * np.sin(cx * np.arctan(bx * kappa_x - ex * (bx * kappa_x - np.arctan(bx * kappa_x))))

        # 5. Combined Slip (Friction Circle)
        # Simple "Elliptical" combination method
        # If we are using 90% of grip for turning, we only have 10% left for braking
        # Avoid divide by zero
        safe_fx = abs(Fx_pure) + 1e-6
        safe_fy = abs(Fy_pure) + 1e-6
        
        rho = np.sqrt((Fx_pure / safe_fx)**2 + (Fy_pure / safe_fy)**2)
        
        if rho > 1.0:
            Fx = Fx_pure / rho
            Fy = Fy_pure / rho
        else:
            Fx = Fx_pure
            Fy = Fy_pure

        return Fx, Fy

    def compute_thermal_dynamics(self, Fx, Fy, alpha, kappa, Vx, T_curr):
        """
        Calculates the Rate of Change of Temperature (dT/dt) in [K/s].
        Used by the OCP solver to integrate temperature.
        """
        # 1. Sliding Velocities
        # Lateral Slide Velocity ~= Vx * tan(alpha)
        # Longitudinal Slide Velocity = Vx * kappa
        # (Simplified for high speed approximation)
        V_sy = Vx * np.tan(alpha)
        V_sx = Vx * kappa
        
        # 2. Heat Generation (Power = Force * Velocity)
        # We take absolute power because friction always heats, never cools
        P_gen_lat = abs(Fy * V_sy)
        P_gen_long = abs(Fx * V_sx)
        P_total = (P_gen_lat + P_gen_long) * self.C_heat

        # 3. Cooling (Convection)
        # Cooling increases with speed (Nusselt number ~ Re^0.5 or Re^0.8)
        # Q_cool = h * A * (T - T_env)
        # We approximate h * A as C_cool * Vx^0.5
        cooling_rate = self.C_cool * (abs(Vx) + 1.0) * (T_curr - self.T_env)
        
        # 4. Net Flux
        Q_net = P_total - cooling_rate
        
        # 5. Temperature Rate (dT/dt = Q / (m * Cp))
        dT_dt = Q_net / (self.mass * self.Cp)
        
        return dT_dt

    def get_peak_slip(self, Fz):
        """
        Helper: Returns the optimal slip angle for a given load.
        Useful for controllers/ABS.
        """
        # For this model, peak is roughly constant around 6-8 degrees
        # Ideally this would be solved numerically
        return 0.12 # radians (~7 degrees)