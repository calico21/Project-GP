import numpy as np
import casadi as ca
import torch
import torch.nn as nn

# --- PINN FOR TRANSIENT LATENT STATES (The SOTA Upgrade) ---
class TransientTirePINN(nn.Module):
    """
    Physics-Informed Neural Network mapping transient relaxation lengths
    and latent grip modifications during extreme combined slip (trail-braking/drifting).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2) # Outputs: [mu_x_transient_mod, mu_y_transient_mod]
        )
        
        # Initialize with neutral weights to prevent chaotic warm-starts
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state_tensor):
        return self.net(state_tensor)


class PacejkaTire:
    """
    State-of-the-Art 5-Node (Inner/Center/Outer/Core/Gas) Thermodynamic Tire Model.
    
    Thermal Physics:
    1. Surface Ribs (3 Nodes): Heated asymmetrically by friction, laterally connected via Fourier's Law.
    2. Core (1 Node): Heated by rolling resistance, conducts vertically to all 3 surface ribs.
    3. Gas (1 Node): Convection from the core, modulates dynamic inflation pressure.
    """
    def __init__(self, tire_coeffs):
        self.coeffs = tire_coeffs
        
        # --- THERMAL PARAMETERS ---
        self.T_opt = self.coeffs.get('T_opt', 90.0)
        self.T_env = 25.0
        self.P_nom = self.coeffs.get('P_nom', 1.2)
        
        # Masses & Capacities (5-Node Discretization)
        self.m_total = self.coeffs.get('mass', 10.0)
        self.m_surf = self.m_total * 0.10
        self.m_rib = self.m_surf / 3.0    # Discretized Surface Mass
        self.m_core = self.m_total * 0.90
        self.m_gas = self.coeffs.get('m_gas', 0.05)
        
        self.Cp_rubber = self.coeffs.get('Cp', 1100.0)
        self.Cv_gas = self.coeffs.get('Cv_gas', 718.0)
        
        # Heat Transfer Coefficients
        self.h_conv_ext = self.coeffs.get('h_conv', 50.0)
        self.k_cond_vert = self.coeffs.get('k_cond', 150.0) # Core <-> Ribs
        self.k_cond_lat = self.coeffs.get('k_cond_lat', 85.0) # Rib <-> Rib (Fourier)
        self.h_conv_int = self.coeffs.get('h_conv_int', 30.0) # Core <-> Gas
        
        self.A_surf = self.coeffs.get('A_surf', 0.8)
        self.A_rib = self.A_surf / 3.0
        self.q_roll = self.coeffs.get('q_roll', 0.03)

        # Initialize PINN
        self.pinn = TransientTirePINN()
        self.pinn.eval() # Running in inference mode during simulation

    def _is_symbolic(self, var):
        return isinstance(var, (ca.SX, ca.MX))

    def compute_force(self, alpha, kappa, Fz, gamma, T_ribs, T_gas, Vx=15.0):
        """
        Calculates tire forces using the average contact patch temperature,
        pressure-coupling, and PINN-driven transient modulation.
        """
        if self._is_symbolic(alpha) or self._is_symbolic(Fz):
            _sin, _arctan, _max, _min = ca.sin, ca.arctan, ca.fmax, ca.fmin
        else:
            _sin, _arctan, _max, _min = np.sin, np.arctan, max, min

        T_surf_in, T_surf_mid, T_surf_out = T_ribs
        
        # 1. Dynamic Pressure Coupling
        T_gas_K = T_gas + 273.15
        T_env_K = self.T_env + 273.15
        P_dyn = self.P_nom * (T_gas_K / T_env_K)
        
        dP = P_dyn - (self.P_nom + 0.2)
        pressure_modifier = _max(0.6, _min(1.0, 1.0 - 0.15 * (dP**2)))

        # 2. Extract Pacejka Coefficients
        dy, cy, by, ey = 1.3, 1.5, 10.0, -1.0
        dx, cx, bx, ex = 1.35, 1.6, 12.0, -0.5

        # 3. Load & Pressure Sensitivity
        Fz_nom = 4000.0
        d_fz = (Fz - Fz_nom) / Fz_nom
        lambda_mu_y = (1.0 - 0.1 * d_fz) * pressure_modifier
        lambda_mu_x = (1.0 - 0.08 * d_fz) * pressure_modifier
        
        # 4. Asymmetric Thermal Sensitivity (Camber Driven)
        Fz_in  = Fz * (0.333 + 0.15 * gamma)
        Fz_mid = Fz * 0.334
        Fz_out = Fz * (0.333 - 0.15 * gamma)
        
        # Added +1e-6 to prevent division by zero during acados initialization
        T_eff = (T_surf_in * Fz_in + T_surf_mid * Fz_mid + T_surf_out * Fz_out) / (Fz + 1e-6)
        therm_factor = _max(0.5, _min(1.0, 1.0 - 0.002 * (T_eff - self.T_opt)**2))

        # 5. PINN Transient Modulation
        if self._is_symbolic(alpha):
            mu_x_mod, mu_y_mod = 1.0, 1.0
        else:
            state_in = torch.tensor([alpha, kappa, gamma, Fz, Vx], dtype=torch.float32)
            with torch.no_grad():
                mods = self.pinn(state_in).numpy()
            mu_x_mod = 1.0 + mods[0]
            mu_y_mod = 1.0 + mods[1]

        mu_y = dy * lambda_mu_y * therm_factor * mu_y_mod
        mu_x = dx * lambda_mu_x * therm_factor * mu_x_mod

        # 6. Combined Slip Magic Formula
        Fy = Fz * mu_y * _sin(cy * _arctan(by * alpha - ey * (by * alpha - _arctan(by * alpha))))
        Fx = Fz * mu_x * _sin(cx * _arctan(bx * kappa - ex * (bx * kappa - _arctan(bx * kappa))))

        return Fx, Fy

    def compute_thermal_dynamics(self, Fx, Fy, Fz, gamma, alpha, kappa, Vx, T_core, T_ribs, T_gas):
        """
        5-Node Thermal ODEs with Asymmetric Heat Generation and Lateral Conduction.
        """
        if self._is_symbolic(Fx) or self._is_symbolic(alpha):
            # FIX: Properly defined lambda function for CasADi symbolic math
            _abs = lambda x: ca.sqrt(x**2 + 1e-6)
            _tan = ca.tan
        else:
            _abs, _tan = abs, np.tan
            
        T_surf_in, T_surf_mid, T_surf_out = T_ribs

        # Asymmetric Vertical Load Distribution (due to Camber)
        dist_in = 0.333 + 0.15 * gamma
        dist_out = 0.333 - 0.15 * gamma
        
        # Power Generation
        P_fric = _abs(Fx * Vx * kappa) + _abs(Fy * Vx * _tan(alpha))
        P_in = P_fric * dist_in
        P_mid = P_fric * 0.334
        P_out = P_fric * dist_out
        
        P_core_in = self.q_roll * _abs(Fz) * _abs(Vx)

        # 1. VERTICAL Conduction (Core to 3 Ribs)
        k_vert_rib = self.k_cond_vert / 3.0
        Q_v_in = k_vert_rib * (T_core - T_surf_in)
        Q_v_mid = k_vert_rib * (T_core - T_surf_mid)
        Q_v_out = k_vert_rib * (T_core - T_surf_out)

        # 2. LATERAL Conduction (Fourier's Law between Ribs)
        Q_lat_in_mid = self.k_cond_lat * (T_surf_in - T_surf_mid)
        Q_lat_mid_out = self.k_cond_lat * (T_surf_mid - T_surf_out)

        # 3. CONVECTION (Air & Gas)
        h_dyn = self.h_conv_ext * (1.0 + 0.5 * _abs(Vx))
        Q_air_in = h_dyn * self.A_rib * (T_surf_in - self.T_env)
        Q_air_mid = h_dyn * self.A_rib * (T_surf_mid - self.T_env)
        Q_air_out = h_dyn * self.A_rib * (T_surf_out - self.T_env)
        
        Q_gas_conv = self.h_conv_int * (T_core - T_gas)

        # 4. DIFFERENTIAL EQUATIONS
        dT_in_dt = (P_in + Q_v_in - Q_lat_in_mid - Q_air_in) / (self.m_rib * self.Cp_rubber)
        dT_mid_dt = (P_mid + Q_v_mid + Q_lat_in_mid - Q_lat_mid_out - Q_air_mid) / (self.m_rib * self.Cp_rubber)
        dT_out_dt = (P_out + Q_v_out + Q_lat_mid_out - Q_air_out) / (self.m_rib * self.Cp_rubber)
        
        dT_core_dt = (P_core_in - Q_v_in - Q_v_mid - Q_v_out - Q_gas_conv) / (self.m_core * self.Cp_rubber)
        dT_gas_dt = Q_gas_conv / (self.m_gas * self.Cv_gas)
        
        return [dT_in_dt, dT_mid_dt, dT_out_dt], dT_core_dt, dT_gas_dt