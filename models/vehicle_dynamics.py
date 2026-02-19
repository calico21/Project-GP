import numpy as np
import casadi as ca
from models.tire_model import PacejkaTire

class MultiBodyVehicle:
    """
    14-DOF Vehicle Dynamics Model (Time Domain).
    Used by: Surrogate Optimizer (Ground Truth Generation).
    Upgraded to support 3-Node (Surface/Core/Gas) Thermal Physics.
    """
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
    
    def simulate_step(self, x, u, setup_params, dt=0.005):
        """
        Forward integration for one time step.
        State x (14): 
          [X, Y, psi, vx, vy, r, delta, 
           T_core_f, T_surf_f, T_gas_f, 
           T_core_r, T_surf_r, T_gas_r, unused]
        """
        # 1. Unpack Kinematic State
        X, Y, psi = x[0], x[1], x[2]
        vx, vy, r = x[3], x[4], x[5]
        delta     = x[6] 
        
        # 2. Unpack 3-Node Thermal States (Initialize if zero/invalid)
        # Assuming nominal cold temp is 25C, core/surf start slightly warm
        T_core_f = x[7] if x[7] > 10 else 60.0
        T_surf_f = x[8] if x[8] > 10 else 60.0
        T_gas_f  = x[9] if x[9] > 10 else 25.0
        
        T_core_r = x[10] if x[10] > 10 else 60.0
        T_surf_r = x[11] if x[11] > 10 else 60.0
        T_gas_r  = x[12] if x[12] > 10 else 25.0

        # 3. Unpack Setup (Optimization Variables)
        k_f, k_r = setup_params[0], setup_params[1]
        
        # 4. Inputs
        steer_req = u[0]
        d_delta = (steer_req - delta) * 15.0 # Simple actuator lag
        
        # 5. Vertical Loads (Static + Aero + Weight Transfer)
        m = self.vp['m']
        lf, lr = self.vp['lf'], self.vp['lr']
        h_cg = self.vp.get('h_cg', 0.3)
        track_w = 1.2 
        
        # Aero
        rho, A = 1.225, self.vp['A']
        Cl = self.vp.get('Cl', 3.0) 
        Fz_aero = 0.5 * rho * Cl * A * vx**2
        
        # Static
        Fz_f_static = (m * 9.81 * lr) / (lf + lr) + Fz_aero * 0.4
        Fz_r_static = (m * 9.81 * lf) / (lf + lr) + Fz_aero * 0.6
        
        # Lateral Weight Transfer (Elastic)
        roll_stiff_f = k_f 
        roll_stiff_r = k_r
        total_stiff = roll_stiff_f + roll_stiff_r + 1.0
        
        ay = (vx * r + vy * 0) 
        W_transfer = (m * ay * h_cg) / track_w
        
        dFz_f = W_transfer * (roll_stiff_f / total_stiff)
        dFz_r = W_transfer * (roll_stiff_r / total_stiff)
        
        Fz_f_l = max(0, Fz_f_static - dFz_f)
        Fz_f_r = max(0, Fz_f_static + dFz_f)
        Fz_r_l = max(0, Fz_r_static - dFz_r)
        Fz_r_r = max(0, Fz_r_static + dFz_r)

        # 6. Tire Kinematics (Slip Angles)
        beta = np.arctan2(vy, vx) if vx > 1.0 else 0.0
        
        alpha_f = delta - (beta + lf*r/vx)
        alpha_r = - (beta - lr*r/vx)
        
        # 7. Tire Forces (Pressure-Coupled)
        # Front
        Fx_fl, Fy_fl = self.tire.compute_force(alpha_f, 0, Fz_f_l, T_surf_f, T_gas_f)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_f, 0, Fz_f_r, T_surf_f, T_gas_f)
        Fy_f_tot = Fy_fl + Fy_fr
        
        # Rear
        Fx_rl, Fy_rl = self.tire.compute_force(alpha_r, 0, Fz_r_l, T_surf_r, T_gas_r)
        Fx_rr, Fy_rr = self.tire.compute_force(alpha_r, 0, Fz_r_r, T_surf_r, T_gas_r)
        Fy_r_tot = Fy_rl + Fy_rr
        
        # 8. Thermal Dynamics (3-Node ODEs)
        dt_core_f, dt_surf_f, dt_gas_f = self.tire.compute_thermal_dynamics(
            0, Fy_f_tot/2, Fz_f_static, alpha_f, 0, vx, T_core_f, T_surf_f, T_gas_f
        )
        
        dt_core_r, dt_surf_r, dt_gas_r = self.tire.compute_thermal_dynamics(
            0, Fy_r_tot/2, Fz_r_static, alpha_r, 0, vx, T_core_r, T_surf_r, T_gas_r
        )

        # 9. Equations of Motion (Body Frame - Bicycle approx)
        vy_dot = (Fy_f_tot * np.cos(delta) + Fy_r_tot) / m - vx * r
        r_dot = (lf * Fy_f_tot * np.cos(delta) - lr * Fy_r_tot) / self.vp['Iz']
        vx_dot = 0 
        
        # 10. Integration
        x_next = np.zeros_like(x)
        x_next[0] = X + dt * (vx * np.cos(psi) - vy * np.sin(psi))
        x_next[1] = Y + dt * (vx * np.sin(psi) + vy * np.cos(psi))
        x_next[2] = psi + dt * r
        x_next[3] = vx + dt * vx_dot
        x_next[4] = vy + dt * vy_dot
        x_next[5] = r + dt * r_dot
        x_next[6] = delta + dt * d_delta
        
        # Thermal Integration
        x_next[7] = T_core_f + dt * dt_core_f
        x_next[8] = T_surf_f + dt * dt_surf_f
        x_next[9] = T_gas_f + dt * dt_gas_f
        
        x_next[10] = T_core_r + dt * dt_core_r
        x_next[11] = T_surf_r + dt * dt_surf_r
        x_next[12] = T_gas_r + dt * dt_gas_r
        
        return x_next


class DynamicBicycleModel:
    """
    Path-Coordinate Model for OCP.
    """
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
        
    def get_equations(self):
        """
        Returns a CasADi function: f_dyn(state, control, curvature) -> derivatives
        """
        # Symbols for mechanical states
        n = ca.MX.sym('n')
        alpha = ca.MX.sym('alpha') 
        v = ca.MX.sym('v')
        delta = ca.MX.sym('delta')
        r = ca.MX.sym('r')
        
        x_mech = ca.vertcat(n, alpha, v, delta, r)
        
        # Control
        u_d_delta = ca.MX.sym('u_d_delta')
        u_fx = ca.MX.sym('u_fx')
        u = ca.vertcat(u_d_delta, u_fx)
        
        # Curvature
        k_c = ca.MX.sym('k_c') 
        
        # Parameters
        m, Iz = self.vp['m'], self.vp['Iz']
        lf, lr = self.vp['lf'], self.vp['lr']
        
        # Vertical Loads
        rho, A, Cl = 1.225, self.vp['A'], self.vp['Cl']
        Fz_aero = 0.5 * rho * Cl * A * v**2
        Fz_f = m * 9.81 * lr / (lf + lr) + Fz_aero/2
        Fz_r = m * 9.81 * lf / (lf + lr) + Fz_aero/2
        
        # Slip Angles
        alpha_f = delta - (alpha + lf*r/v)
        alpha_r = - (alpha - lr*r/v)
        
        # Decoupled Thermal Assumptions for the Mechanical OCP Phase
        # We assume optimal grip temp (90.0) and optimal hot inflation pressure (60.0)
        Fx_f, Fy_f = self.tire.compute_force(alpha_f, 0, Fz_f, T_surf=90.0, T_gas=60.0)
        Fx_r, Fy_r = self.tire.compute_force(alpha_r, 0, Fz_r, T_surf=90.0, T_gas=60.0)
        
        # Derivatives (Spatial -> Time mapping)
        n_dot = v * ca.sin(alpha)
        alpha_dot = ((Fy_f * ca.cos(delta) + Fy_r) / m - v * r) / v
        v_dot = u_fx / m
        r_dot = (lf * Fy_f * ca.cos(delta) - lr * Fy_r) / Iz
        delta_dot = u_d_delta
        
        rhs = ca.vertcat(n_dot, alpha_dot, v_dot, delta_dot, r_dot)
        
        return ca.Function('f_dyn', [x_mech, u, k_c], [rhs])