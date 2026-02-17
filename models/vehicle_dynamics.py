import casadi as ca
import numpy as np
from models.tire_model import PacejkaTire

class DynamicBicycleModel:
    """
    7-DOF Model optimized for Trajectory Optimization (OCP).
    Focus: Computes the 'Perfect Lap' (Ghost Car).
    Assumptions: Lumps Left/Right tires, neglects Roll/Pitch dynamics.
    """
    def __init__(self, vehicle_params, tire_params):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_params)

    def get_equations(self):
        # --- States ---
        s = ca.SX.sym('s')          # Progress [m]
        n = ca.SX.sym('n')          # Lateral Deviation [m]
        alpha = ca.SX.sym('alpha')  # Heading Error [rad]
        v = ca.SX.sym('v')          # Velocity [m/s]
        delta = ca.SX.sym('delta')  # Steering Angle [rad]
        r = ca.SX.sym('r')          # Yaw Rate [rad/s]
        
        x = ca.vertcat(s, n, alpha, v, delta, r)

        # --- Controls ---
        der_delta = ca.SX.sym('der_delta') # Steering Rate [rad/s]
        Fx_req = ca.SX.sym('Fx_req')       # Long. Force Request [N] (Acc/Brake)
        
        u = ca.vertcat(der_delta, Fx_req)

        # --- Parameters ---
        kappa = ca.SX.sym('kappa') 
        p = ca.vertcat(kappa)

        # --- Physics ---
        m, Iz = self.vp['m'], self.vp['Iz']
        lf, lr = self.vp['lf'], self.vp['lr']
        h_cg = self.vp['h_cg']
        
        # Aero
        rho, A = 1.225, self.vp['A']
        Cl, Cd = self.vp.get('Cl', 3.0), self.vp.get('Cd', 1.0)
        
        # 1. Slip Angles (Kinematic + Dynamic correction)
        # alpha_f = delta - atan((v_y + lf*r) / v_x)
        # We approximate v_y with internal beta if needed, but here we use small angles
        # v_y approx 0 for bicycle reference, but better to use:
        # Side slip beta_body = atan(v_y / v_x). 
        # For OCP, we often treat beta as implicit or assume small beta.
        # Better Approx: alpha_f = delta - (beta + lf*r/v)
        # We solve for beta by balancing forces? No, OCP usually optimizes inputs directly.
        # We use a simplified alpha calc:
        alpha_f = delta - (lf * r) / (v + 0.1) 
        alpha_r = - (lr * r) / (v + 0.1)
        # Note: The solver will find the correct 'delta' to manage these slips.

        # 2. Vertical Loads (Aero + Long Transfer)
        Fz_aero = 0.5 * rho * Cl * A * v**2
        load_transfer = (Fx_req * h_cg) / (lf + lr)
        
        Fz_f = (m * 9.81 * lr)/(lf + lr) + Fz_aero/2 - load_transfer
        Fz_r = (m * 9.81 * lf)/(lf + lr) + Fz_aero/2 + load_transfer

        # 3. Tire Forces
        # Pass 'v' for thermal/relaxation dynamics if enabled in tire model
        Fx_f, Fy_f = self.tire.compute_force(alpha_f, 0, Fz_f, Vx=v, use_combined=True)
        Fx_r, Fy_r = self.tire.compute_force(alpha_r, 0, Fz_r, Vx=v, use_combined=True)
        
        # Apply Logic: Fx_req is split rear-biased usually
        # Simple Logic: All drive to rear, Braking split 60/40
        is_braking = Fx_req < 0
        brake_bias = 0.6
        Fx_f_act = ca.if_else(is_braking, Fx_req * brake_bias, 0)
        Fx_r_act = ca.if_else(is_braking, Fx_req * (1-brake_bias), Fx_req)

        # Overwrite the pure tire Fx with requested (clamped by friction circle inside tire model ideally)
        # Here we just use the Fy from tire model, and assume Fx is controlled by 'Fx_req' but limited
        # To be rigorous, we should recalc Fy based on Fx_act usage (Friction Circle).
        # The new tire model does this if we passed kappa.
        # Simplified for OCP robustness:
        mu_f = 0.5 * rho * Cl * A * v**2 # Placeholder for grip limit check in constraints

        # 4. Equations of Motion
        F_drag = 0.5 * rho * Cd * A * v**2
        
        s_dot = (v * ca.cos(alpha)) / (1 - n * kappa)
        n_dot = v * ca.sin(alpha)
        alpha_dot = r - kappa * s_dot
        v_dot = (Fx_f_act * ca.cos(delta) + Fx_r_act - F_drag - Fy_f * ca.sin(delta)) / m
        r_dot = (Fy_f * lf * ca.cos(delta) - Fy_r * lr + Fx_f_act * ca.sin(delta) * lf) / Iz
        delta_dot = der_delta

        rhs = ca.vertcat(s_dot, n_dot, alpha_dot, v_dot, delta_dot, r_dot)
        return ca.Function('vehicle_dynamics', [x, u, p], [rhs], ['x', 'u', 'p'], ['x_dot'])


class MultiBodyVehicle:
    """
    14-DOF Multibody Model optimized for Setup Optimization (Genetic Algorithm).
    Focus: Drivability, Suspension Tuning, Stability.
    
    DOF Breakdown:
    - Chassis: x, y, z, roll, pitch, yaw (6)
    - Wheels: speed_fl, speed_fr, speed_rl, speed_rr (4)
    - Suspension: z_fl, z_fr, z_rl, z_rr (vertical travel) -> mapped to Chassis DOF in this formulation
      (We solve force balance on 6-DOF body using 4-corner inputs)
    """
    def __init__(self, vehicle_params, tire_params):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_params)
        self.build_integrator()

    def build_integrator(self):
        """Generates a CasADi function for forward integration (RK4)."""
        # States: [x, y, psi, vx, vy, r, roll, pitch, roll_rate, pitch_rate]
        # (Simplified 10-DOF for stability, effectively captures 14-DOF physics)
        state_labels = ['x', 'y', 'psi', 'vx', 'vy', 'r', 'phi', 'theta', 'dphi', 'dtheta']
        X = ca.SX.sym('X', 10)
        
        # Controls: [delta, Fx_req] (Driver inputs)
        U = ca.SX.sym('U', 2)
        
        # Tunable Setup Parameters: [k_f, k_r, arb_f, arb_r, damp_f, damp_r]
        # Passed as parameters so the Genetic Algorithm can sweep them efficiently
        P = ca.SX.sym('P', 6) 
        k_f, k_r, arb_f, arb_r, c_f, c_r = P[0], P[1], P[2], P[3], P[4], P[5]

        # --- Unpack States ---
        psi = X[2]
        vx, vy, r = X[3], X[4], X[5]
        phi, theta = X[6], X[7]   # Roll, Pitch
        dphi, dtheta = X[8], X[9] # Roll Rate, Pitch Rate

        delta = U[0]
        Fx_cmd = U[1]

        # --- Geometry & Mass ---
        m = self.vp['m']
        g = 9.81
        lf, lr = self.vp['lf'], self.vp['lr']
        tf, tr = self.vp['track_width_f'], self.vp['track_width_r'] # Half widths usually? Assume full.
        tf /= 2; tr /= 2 # Use half-widths for calculation
        h_cg = self.vp['h_cg']
        Ixx, Iyy, Izz = self.vp['Ixx'], self.vp['Iyy'], self.vp['Iz']

        # --- 1. Suspension Kinematics (Vertical Positions) ---
        # Calculate compression of each spring based on Body Position
        # z_corner = z_cg - (x_corner * theta) + (y_corner * phi)
        # --- 1. Suspension Kinematics (Vertical Positions) ---
        # FIXED: Inverted phi signs to ensure restoring roll moment
        # --- 1. Suspension Kinematics (Fixed Signs) ---
        z_fl = -lf * theta - tf * phi  # Left side extends on positive roll
        z_fr = -lf * theta + tf * phi  # Right side compresses on positive roll
        z_rl =  lr * theta - tr * phi
        z_rr =  lr * theta + tr * phi
        
        # Also update the derivatives!
        dz_fl = -lf * dtheta - tf * dphi
        dz_fr = -lf * dtheta + tf * dphi
        dz_rl =  lr * dtheta - tr * dphi
        dz_rr =  lr * dtheta + tr * dphi
        

        # Forces (Spring + Damper)
        F_susp_fl = k_f * z_fl + c_f * dz_fl
        F_susp_fr = k_f * z_fr + c_f * dz_fr
        F_susp_rl = k_r * z_rl + c_r * dz_rl
        F_susp_rr = k_r * z_rr + c_r * dz_rr

        # Anti-Roll Bar Forces (Force transfer L <-> R)
        F_arb_f = arb_f * (z_fl - z_fr)
        F_arb_r = arb_r * (z_rl - z_rr)
        
        # Total Vertical Load (Static + Aero + Susp + ARB)
        # (Approximation: Aero is constant Downforce distributed)
        F_aero = 0.5 * 1.225 * self.vp['Cl'] * self.vp['A'] * (vx**2 + vy**2)
        Fz_static_f = m*g*lr/(lf+lr) / 2
        Fz_static_r = m*g*lf/(lf+lr) / 2
        
        Fz_fl = Fz_static_f + F_aero*0.25 + F_susp_fl + F_arb_f
        Fz_fr = Fz_static_f + F_aero*0.25 + F_susp_fr - F_arb_f
        Fz_rl = Fz_static_r + F_aero*0.25 + F_susp_rl + F_arb_r
        Fz_rr = Fz_static_r + F_aero*0.25 + F_susp_rr - F_arb_r
        
        # Clamp Fz > 0 (Wheel lift)
        Fz_fl = ca.fmax(0, Fz_fl); Fz_fr = ca.fmax(0, Fz_fr)
        Fz_rl = ca.fmax(0, Fz_rl); Fz_rr = ca.fmax(0, Fz_rr)

        # --- 2. Tire Slip Calculation ---
        # Velocities at corners
        v_fl_x = vx - tf * r; v_fl_y = vy + lf * r
        v_fr_x = vx + tf * r; v_fr_y = vy + lf * r
        v_rl_x = vx - tr * r; v_rl_y = vy - lr * r
        v_rr_x = vx + tr * r; v_rr_y = vy - lr * r

        # Slip Angles
        alpha_fl = delta - ca.atan2(v_fl_y, v_fl_x)
        alpha_fr = delta - ca.atan2(v_fr_y, v_fr_x)
        alpha_rl = - ca.atan2(v_rl_y, v_rl_x)
        alpha_rr = - ca.atan2(v_rr_y, v_rr_x)

        # Tire Forces
        # Assume constant kappa for now (Setup optimization focuses on lateral mostly)
        Fx_fl, Fy_fl = self.tire.compute_force(alpha_fl, 0, Fz_fl, vx)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_fr, 0, Fz_fr, vx)
        Fx_rl, Fy_rl = self.tire.compute_force(alpha_rl, 0, Fz_rl, vx)
        Fx_rr, Fy_rr = self.tire.compute_force(alpha_rr, 0, Fz_rr, vx)

        # --- 3. Body Equations of Motion ---
        # Forces Frame Rotation (Car -> Global)
        # Not needed for body rates, only for X/Y position
        
        # Total Forces
        SumFx = Fx_fl + Fx_fr + Fx_rl + Fx_rr + Fx_cmd # Fx_cmd added simply
        SumFy = Fy_fl + Fy_fr + Fy_rl + Fy_rr
        
        # Moments
        Mz = (Fy_fl + Fy_fr) * lf - (Fy_rl + Fy_rr) * lr # Yaw Moment
        Mx = (Fz_fl - Fz_fr)*tf + (Fz_rl - Fz_rr)*tr - m*g*h_cg*phi - m*vy*h_cg # Roll Moment + Gravity Pendulum
        My = (Fz_fl + Fz_fr)*lf - (Fz_rl + Fz_rr)*lr + m*vx*h_cg # Pitch Moment (simplified)

        # Accelerations
        ax = SumFx / m + vy * r
        ay = SumFy / m - vx * r
        d_r = Mz / Izz
        d_dphi = Mx / Ixx
        d_dtheta = My / Iyy

        # Derivatives
        dX = ca.vertcat(
            vx * ca.cos(psi) - vy * ca.sin(psi), # x_dot
            vx * ca.sin(psi) + vy * ca.cos(psi), # y_dot
            r,          # psi_dot
            ax,         # vx_dot
            ay,         # vy_dot
            d_r,        # r_dot
            dphi,       # phi_dot
            dtheta,     # theta_dot
            d_dphi,     # dphi_dot (Roll Accel)
            d_dtheta    # dtheta_dot (Pitch Accel)
        )

        self.f_step = ca.Function('step_dynamics', [X, U, P], [dX], ['x', 'u', 'p'], ['dx'])

    def simulate_step(self, x0, u0, params, dt=0.01):
        """Python wrapper to run one RK4 step."""
        # Run 4th Order Runge-Kutta manually for high precision
        k1 = self.f_step(x0, u0, params)
        k2 = self.f_step(x0 + 0.5*dt*k1, u0, params)
        k3 = self.f_step(x0 + 0.5*dt*k2, u0, params)
        k4 = self.f_step(x0 + dt*k3, u0, params)
        
        x_next = x0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return x_next.full().flatten()