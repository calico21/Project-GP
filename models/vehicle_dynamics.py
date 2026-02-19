import numpy as np
import casadi as ca
from models.tire_model import PacejkaTire

class MultiBodyVehicle:
    """
    14-DOF Vehicle Dynamics Model (Time Domain).
    Used by: Surrogate Optimizer (Ground Truth Generation).
    Upgraded to support 5-Node (Inner/Mid/Outer/Core/Gas) Thermal Physics & Dynamic Camber.
    """
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
    
    def simulate_step(self, x, u, setup_params, dt=0.005):
        """
        Forward integration for one time step.
        State x (17): 
          [X, Y, psi, vx, vy, r, delta, 
           T_core_f, T_in_f, T_mid_f, T_out_f, T_gas_f, 
           T_core_r, T_in_r, T_mid_r, T_out_r, T_gas_r]
        """
        X, Y, psi = x[0], x[1], x[2]
        vx, vy, r = x[3], x[4], x[5]
        delta     = x[6] 
        
        T_core_f = x[7]  if x[7]  > 10 else 60.0
        T_in_f   = x[8]  if x[8]  > 10 else 60.0
        T_mid_f  = x[9]  if x[9]  > 10 else 60.0
        T_out_f  = x[10] if x[10] > 10 else 60.0
        T_gas_f  = x[11] if x[11] > 10 else 25.0
        
        T_core_r = x[12] if x[12] > 10 else 60.0
        T_in_r   = x[13] if x[13] > 10 else 60.0
        T_mid_r  = x[14] if x[14] > 10 else 60.0
        T_out_r  = x[15] if x[15] > 10 else 60.0
        T_gas_r  = x[16] if x[16] > 10 else 25.0

        k_f, k_r = setup_params[0], setup_params[1]
        
        steer_req = u[0]
        d_delta = (steer_req - delta) * 15.0 
        
        m = self.vp['m']
        lf, lr = self.vp['lf'], self.vp['lr']
        h_cg = self.vp.get('h_cg', 0.3)
        track_w = 1.2 
        
        rho, A = 1.225, self.vp['A']
        Cl = self.vp.get('Cl', 3.0) 
        Fz_aero = 0.5 * rho * Cl * A * vx**2
        
        Fz_f_static = (m * 9.81 * lr) / (lf + lr) + Fz_aero * 0.4
        Fz_r_static = (m * 9.81 * lf) / (lf + lr) + Fz_aero * 0.6
        
        roll_stiff_f, roll_stiff_r = k_f, k_r
        total_stiff = roll_stiff_f + roll_stiff_r + 1.0
        
        ay = (vx * r + vy * 0) 
        W_transfer = (m * ay * h_cg) / track_w
        
        phi = (m * ay * h_cg) / total_stiff 
        phi_deg = np.rad2deg(phi)
        
        gamma_f = -2.0 + (phi_deg * 0.6) 
        gamma_r = -1.5 + (phi_deg * 0.4)
        
        dFz_f = W_transfer * (roll_stiff_f / total_stiff)
        dFz_r = W_transfer * (roll_stiff_r / total_stiff)
        
        Fz_f_l = max(0, Fz_f_static - dFz_f)
        Fz_f_r = max(0, Fz_f_static + dFz_f)
        Fz_r_l = max(0, Fz_r_static - dFz_r)
        Fz_r_r = max(0, Fz_r_static + dFz_r)

        beta = np.arctan2(vy, vx) if vx > 1.0 else 0.0
        alpha_f = delta - (beta + lf*r/vx)
        alpha_r = - (beta - lr*r/vx)
        
        T_ribs_f = [T_in_f, T_mid_f, T_out_f]
        Fx_fl, Fy_fl = self.tire.compute_force(alpha_f, 0, Fz_f_l, gamma_f, T_ribs_f, T_gas_f, vx)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_f, 0, Fz_f_r, gamma_f, T_ribs_f, T_gas_f, vx)
        Fy_f_tot = Fy_fl + Fy_fr
        
        T_ribs_r = [T_in_r, T_mid_r, T_out_r]
        Fx_rl, Fy_rl = self.tire.compute_force(alpha_r, 0, Fz_r_l, gamma_r, T_ribs_r, T_gas_r, vx)
        Fx_rr, Fy_rr = self.tire.compute_force(alpha_r, 0, Fz_r_r, gamma_r, T_ribs_r, T_gas_r, vx)
        Fy_r_tot = Fy_rl + Fy_rr
        
        dT_ribs_f, dt_core_f, dt_gas_f = self.tire.compute_thermal_dynamics(
            0, Fy_f_tot/2, Fz_f_static, gamma_f, alpha_f, 0, vx, T_core_f, T_ribs_f, T_gas_f
        )
        
        dT_ribs_r, dt_core_r, dt_gas_r = self.tire.compute_thermal_dynamics(
            0, Fy_r_tot/2, Fz_r_static, gamma_r, alpha_r, 0, vx, T_core_r, T_ribs_r, T_gas_r
        )

        vy_dot = (Fy_f_tot * np.cos(delta) + Fy_r_tot) / m - vx * r
        r_dot = (lf * Fy_f_tot * np.cos(delta) - lr * Fy_r_tot) / self.vp['Iz']
        vx_dot = 0 
        
        x_next = np.zeros_like(x)
        x_next[0:7] = [
            X + dt * (vx * np.cos(psi) - vy * np.sin(psi)),
            Y + dt * (vx * np.sin(psi) + vy * np.cos(psi)),
            psi + dt * r,
            vx + dt * vx_dot,
            vy + dt * vy_dot,
            r + dt * r_dot,
            delta + dt * d_delta
        ]
        
        x_next[7:12] = [
            T_core_f + dt * dt_core_f,
            T_in_f   + dt * dT_ribs_f[0],
            T_mid_f  + dt * dT_ribs_f[1],
            T_out_f  + dt * dT_ribs_f[2],
            T_gas_f  + dt * dt_gas_f
        ]
        
        x_next[12:17] = [
            T_core_r + dt * dt_core_r,
            T_in_r   + dt * dT_ribs_r[0],
            T_mid_r  + dt * dT_ribs_r[1],
            T_out_r  + dt * dT_ribs_r[2],
            T_gas_r  + dt * dt_gas_r
        ]
        
        return x_next


class DynamicBicycleModel:
    """
    Path-Coordinate Model for OCP.
    """
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
        
    def get_equations(self):
        n, alpha, v, delta, r = [ca.MX.sym(name) for name in ['n', 'alpha', 'v', 'delta', 'r']]
        x_mech = ca.vertcat(n, alpha, v, delta, r)
        
        u_d_delta, u_fx = ca.MX.sym('u_d_delta'), ca.MX.sym('u_fx')
        u = ca.vertcat(u_d_delta, u_fx)
        k_c = ca.MX.sym('k_c') 
        
        m, Iz = self.vp['m'], self.vp['Iz']
        lf, lr = self.vp['lf'], self.vp['lr']
        
        rho, A, Cl = 1.225, self.vp['A'], self.vp['Cl']
        Fz_aero = 0.5 * rho * Cl * A * v**2
        Fz_f = m * 9.81 * lr / (lf + lr) + Fz_aero/2
        Fz_r = m * 9.81 * lf / (lf + lr) + Fz_aero/2
        
        alpha_f = delta - (alpha + lf*r/v)
        alpha_r = - (alpha - lr*r/v)
        
        # FIX: The OCP now accurately provides all 7 required arguments to the 5-node tire model
        T_ribs_dummy = [90.0, 90.0, 90.0]
        Fx_f, Fy_f = self.tire.compute_force(alpha_f, 0, Fz_f, -1.5, T_ribs_dummy, 60.0, v)
        Fx_r, Fy_r = self.tire.compute_force(alpha_r, 0, Fz_r, -1.0, T_ribs_dummy, 60.0, v)
        
        n_dot = v * ca.sin(alpha)
        alpha_dot = ((Fy_f * ca.cos(delta) + Fy_r) / m - v * r) / v
        v_dot = u_fx / m
        r_dot = (lf * Fy_f * ca.cos(delta) - lr * Fy_r) / Iz
        delta_dot = u_d_delta
        
        rhs = ca.vertcat(n_dot, alpha_dot, v_dot, delta_dot, r_dot)
        return ca.Function('f_dyn', [x_mech, u, k_c], [rhs])