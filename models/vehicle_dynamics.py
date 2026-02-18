import numpy as np
import casadi as ca
from models.tire_model import PacejkaTire

class MultiBodyVehicle:
    """
    14-DOF Vehicle Dynamics Model (Time Domain).
    Used by: Evolutionary Optimizer (NSGA-II) for forward simulation.
    """
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
    
    def simulate_step(self, x, u, setup_params, dt=0.005):
        """
        Forward Euler integration for one time step.
        State x (10): [X, Y, psi, vx, vy, r, delta, T_f, T_r, unused]
        Input u (2):  [steer_cmd, throttle_cmd]
        """
        # 1. Unpack State
        X, Y, psi = x[0], x[1], x[2]
        vx, vy, r = x[3], x[4], x[5]
        delta     = x[6] # Current steering angle
        T_f       = x[7] if x[7] > 0 else 60.0 # Front Temp
        T_r       = x[8] if x[8] > 0 else 60.0 # Rear Temp

        # 2. Unpack Setup (Optimization Variables)
        # [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, ...]
        k_f, k_r = setup_params[0], setup_params[1]
        arb_f, arb_r = setup_params[2], setup_params[3]
        
        # 3. Inputs
        steer_req = u[0]
        # Simple steering actuator lag
        d_delta = (steer_req - delta) * 15.0 
        
        # 4. Vertical Loads (Static + Aero + Weight Transfer)
        m = self.vp['m']
        lf, lr = self.vp['lf'], self.vp['lr']
        h_cg = self.vp.get('h_cg', 0.3)
        
        # Aero
        rho, A = 1.225, self.vp['A']
        Cl = self.vp.get('Cl', 3.0) 
        Fz_aero = 0.5 * rho * Cl * A * vx**2
        
        # Static
        Fz_f_static = (m * 9.81 * lr) / (lf + lr) + Fz_aero * 0.4
        Fz_r_static = (m * 9.81 * lf) / (lf + lr) + Fz_aero * 0.6
        
        # Lateral Weight Transfer (Elastic)
        roll_stiff_f = k_f * 0.5**2 + arb_f
        roll_stiff_r = k_r * 0.5**2 + arb_r
        total_stiff = roll_stiff_f + roll_stiff_r + 1.0
        
        ay = (vx * r + vy * r) # approx lat accel
        W_transfer = (m * ay * h_cg) / 1.6 # 1.6m track width
        
        dFz_f = W_transfer * (roll_stiff_f / total_stiff)
        dFz_r = W_transfer * (roll_stiff_r / total_stiff)
        
        Fz_f_l = Fz_f_static - dFz_f
        Fz_f_r = Fz_f_static + dFz_f
        Fz_r_l = Fz_r_static - dFz_r
        Fz_r_r = Fz_r_static + dFz_r
        
        # Clamp loads > 0
        Fz_f_l, Fz_f_r = max(0, Fz_f_l), max(0, Fz_f_r)
        Fz_r_l, Fz_r_r = max(0, Fz_r_l), max(0, Fz_r_r)

        # 5. Tire Kinematics (Slip Angles)
        # Front
        alpha_f_l = delta - np.arctan2(vy + lf*r, vx - 0.6) # 0.6 = half track width (1.2m)
        alpha_f_r = delta - np.arctan2(vy + lf*r, vx + 0.6)
        # Rear
        alpha_r_l = -np.arctan2(vy - lr*r, vx - 0.6)
        alpha_r_r = -np.arctan2(vy - lr*r, vx + 0.6)
        
        # 6. Tire Forces (Include Thermal State!)
        # FIX: Unpack order matches Tire Model (Fx, Fy)
        _, Fy_f_l = self.tire.compute_force(alpha_f_l, 0, Fz_f_l, vx, T_f)
        _, Fy_f_r = self.tire.compute_force(alpha_f_r, 0, Fz_f_r, vx, T_f)
        _, Fy_r_l = self.tire.compute_force(alpha_r_l, 0, Fz_r_l, vx, T_r)
        _, Fy_r_r = self.tire.compute_force(alpha_r_r, 0, Fz_r_r, vx, T_r)
        
        Fy_f_tot = Fy_f_l + Fy_f_r
        Fy_r_tot = Fy_r_l + Fy_r_r
        
        # 7. Thermal Dynamics
        # Calculate heat generation for next step
        dT_f = self.tire.compute_thermal_dynamics(0, Fy_f_tot/2, np.mean([alpha_f_l, alpha_f_r]), 0, vx, T_f)
        dT_r = self.tire.compute_thermal_dynamics(0, Fy_r_tot/2, np.mean([alpha_r_l, alpha_r_r]), 0, vx, T_r)

        # 8. Equations of Motion (Body Frame)
        m_dot_vx = Fy_f_tot * np.sin(delta) + m * vy * r 
        m_dot_vy = Fy_f_tot * np.cos(delta) + Fy_r_tot - m * vx * r
        Iz_dot_r = lf * (Fy_f_tot * np.cos(delta)) - lr * Fy_r_tot

        # 9. Integration
        x_next = np.zeros_like(x)
        x_next[0] = X + dt * (vx * np.cos(psi) - vy * np.sin(psi))
        x_next[1] = Y + dt * (vx * np.sin(psi) + vy * np.cos(psi))
        x_next[2] = psi + dt * r
        x_next[3] = vx + dt * (m_dot_vx / m)
        x_next[4] = vy + dt * (m_dot_vy / m)
        x_next[5] = r + dt * (Iz_dot_r / self.vp['Iz'])
        x_next[6] = delta + dt * d_delta
        # Thermal Integration
        x_next[7] = T_f + dt * dT_f
        x_next[8] = T_r + dt * dT_r
        
        return x_next


class DynamicBicycleModel:
    """
    Path-Coordinate Model.
    Used by: OCP Solver (CasADi) for lap time minimization.
    """
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
        
    def get_equations(self):
        """
        Returns a CasADi function: f_dyn(state, control, curvature) -> derivatives
        State: [s, n, alpha, v, delta, r]
        """
        # Symbols
        s = ca.MX.sym('s')
        n = ca.MX.sym('n')
        alpha = ca.MX.sym('alpha') 
        v = ca.MX.sym('v')
        delta = ca.MX.sym('delta')
        r = ca.MX.sym('r')
        
        x = ca.vertcat(s, n, alpha, v, delta, r)
        
        u_d_delta = ca.MX.sym('u_d_delta')
        u_fx = ca.MX.sym('u_fx')
        u = ca.vertcat(u_d_delta, u_fx)
        
        k_c = ca.MX.sym('k_c') 
        
        # Parameters
        m, Iz = self.vp['m'], self.vp['Iz']
        lf, lr = self.vp['lf'], self.vp['lr']
        
        # Slip Angles (Kinematic)
        alpha_f = delta - (lf * r) / v 
        alpha_r = (lr * r) / v - alpha 
        
        Fz_f = m * 9.81 * lr / (lf + lr)
        Fz_r = m * 9.81 * lf / (lf + lr)
        
        # FIX: Unpack order matches Tire Model (Fx, Fy)
        # Note: We still force T=90.0 for OCP simplicity
        _, Fy_f = self.tire.compute_force(alpha_f, 0, Fz_f, v, T_tire=90.0)
        _, Fy_r = self.tire.compute_force(alpha_r, 0, Fz_r, v, T_tire=90.0)
        
        # Derivatives
        s_dot = (v * ca.cos(alpha)) / (1 - n * k_c)
        n_dot = v * ca.sin(alpha)
        alpha_dot = r - k_c * s_dot
        
        v_dot = u_fx / m 
        r_dot = (lf * Fy_f * ca.cos(delta) - lr * Fy_r) / Iz
        delta_dot = u_d_delta
        
        rhs = ca.vertcat(s_dot, n_dot, alpha_dot, v_dot, delta_dot, r_dot)
        
        return ca.Function('f_dyn', [x, u, k_c], [rhs])