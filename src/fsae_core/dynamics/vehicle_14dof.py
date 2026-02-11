import casadi as ca
import numpy as np

class Vehicle14DOF:
    """
    14-Degrees-of-Freedom Vehicle Dynamics Model.
    
    DOFs:
    1-6:  Chassis Position & Orientation (x, y, z, roll, pitch, yaw)
    7-10: Wheel/Suspension Vertical Positions (z_fl, z_fr, z_rl, z_rr)
    11-14: Wheel Spin Speeds (w_fl, w_fr, w_rl, w_rr) (Integrates to Wheel Angle)
    
    State Vector (28 elements):
    0-13:  Positions (Global XYZ, Euler RPY, Susp Z, Wheel Theta)
    14-27: Velocities (Body uvw, Body pqr, Susp dZ, Wheel Omega)
    """

    def __init__(self, params):
        self.p = params
        
    def get_dynamics(self, states, controls):
        """
        Symbolic equations of motion.
        """
        # --- 1. Unpack States ---
        # Positions / Orientations
        pos_x, pos_y, pos_z = states[0], states[1], states[2]
        roll, pitch, yaw    = states[3], states[4], states[5]
        
        # Suspension compressions (absolute vertical pos of unsprung mass)
        z_w = [states[6], states[7], states[8], states[9]] # fl, fr, rl, rr
        
        # Wheel Angles (10, 11, 12, 13) - Not actively used in force calc, but integrated
        # theta_w = [states[10], states[11], states[12], states[13]]
        
        # Velocities
        u, v, w = states[14], states[15], states[16] # Chassis body velocities
        p, q, r = states[17], states[18], states[19] # Chassis rotational rates
        dz_w    = [states[20], states[21], states[22], states[23]] # Susp vertical vel
        omega   = [states[24], states[25], states[26], states[27]] # Wheel spin speeds
        
        # --- 2. Unpack Controls ---
        delta_steer = controls[0] # Steering angle at wheel
        throttle    = controls[1] # 0 to 1
        brake       = controls[2] # 0 to 1
        
        # --- 3. Coordinate Transformations ---
        # Rotation Matrix (Body to Global) - ZYX Euler sequence
        cz = ca.cos(yaw);   sz = ca.sin(yaw)
        cy = ca.cos(pitch); sy = ca.sin(pitch)
        cx = ca.cos(roll);  sx = ca.sin(roll)
        
        # Direction Cosine Matrix (Body -> Inertial)
        R_b2i = ca.vertcat(
            ca.horzcat(cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz),
            ca.horzcat(cy*sz, cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz),
            ca.horzcat(-sy,   cy*sx,            cx*cy)
        )
        
        # --- 4. Tire & Suspension Geometry ---
        track_f = self.p['track_width_f']
        track_r = self.p['track_width_r']
        wb_f    = self.p['wheelbase_f']
        wb_r    = self.p['wheelbase_r']
        
        # Hardpoint locations (x, y) relative to CG
        wheel_pos_body = [
            [wb_f,  track_f/2], # FL
            [wb_f, -track_f/2], # FR
            [-wb_r, track_r/2], # RL
            [-wb_r, -track_r/2] # RR
        ]
        
        tire_forces_x = []
        tire_forces_y = []
        tire_forces_z = [] 
        
        for i in range(4):
            # A. Calculate Normal Load (Fz)
            wx, wy = wheel_pos_body[i]
            z_chassis_corner = pos_z - wy * roll + wx * pitch
            
            susp_deflection = z_chassis_corner - z_w[i]
            susp_velocity   = (w - wy * p + wx * q) - dz_w[i]
            
            # Spring + Damper force
            F_susp = self.p['k_spring'][i] * susp_deflection + self.p['c_damper'][i] * susp_velocity
            
            # Tire stiffness (vertical)
            tire_compression = self.p['tire_radius'] - z_w[i] 
            Fz_tire = ca.fmax(0, self.p['k_tire'] * tire_compression) 
            
            tire_forces_z.append(Fz_tire) 
            
            # B. Calculate Slips
            vx_tire = u - wy * r
            vy_tire = v + wx * r
            
            delta = delta_steer if i < 2 else 0
            
            # Slip Angle (Alpha) & Ratio (Kappa)
            alpha = ca.arctan2(vy_tire, vx_tire) - delta
            kappa = (omega[i] * self.p['tire_radius'] - vx_tire) / (vx_tire + 1.0) # +1 avoids div/0 at stop
            
            # C. Magic Formula (Simplified)
            mu_y = self.p['Dy'] * ca.sin(self.p['Cy'] * ca.arctan(self.p['By'] * alpha))
            mu_x = self.p['Dx'] * ca.sin(self.p['Cx'] * ca.arctan(self.p['Bx'] * kappa))
            
            # Combined slip (Friction Circle)
            Fx = Fz_tire * mu_x
            Fy = Fz_tire * mu_y
            
            # Transform to Body Frame
            Fx_b = Fx * ca.cos(delta) - Fy * ca.sin(delta)
            Fy_b = Fx * ca.sin(delta) + Fy * ca.cos(delta)
            
            tire_forces_x.append(Fx_b)
            tire_forces_y.append(Fy_b)

        # --- 5. Equations of Motion (Newton-Euler) ---
        mass = self.p['mass']
        
        Sum_Fx = sum(tire_forces_x) - 0.5 * 1.225 * self.p['Cd'] * u**2 
        Sum_Fy = sum(tire_forces_y)
        Sum_Fz = sum([F - mass*9.81/4 for F in tire_forces_z]) # Approx gravity distribution
        
        # Moments
        M_z = 0
        M_y = 0 
        M_x = 0 
        
        for i in range(4):
            wx, wy = wheel_pos_body[i]
            Fx, Fy = tire_forces_x[i], tire_forces_y[i]
            M_z += wx * Fy - wy * Fx
            # Roll/Pitch moments simplified for prototype
            M_x += (tire_forces_z[i] * wy) 
            M_y += (tire_forces_z[i] * -wx)

        # 6. Accelerations
        u_dot = Sum_Fx/mass + v*r - w*q
        v_dot = Sum_Fy/mass - u*r + w*p
        w_dot = Sum_Fz/mass + u*q - v*p 
        
        Ixx, Iyy, Izz = self.p['Ixx'], self.p['Iyy'], self.p['Izz']
        p_dot = M_x / Ixx 
        q_dot = M_y / Iyy
        r_dot = M_z / Izz
        
        # 7. Unsprung Mass Dynamics
        dz_w_dot = []
        for i in range(4):
             accel = (tire_forces_z[i] - 0) / self.p['unsprung_mass'] # Simplified
             dz_w_dot.append(accel)

        # 8. Wheel Spin Dynamics
        omega_dot = []
        for i in range(4):
            T_drive = throttle * self.p['max_torque'] / 2 if i >= 2 else 0 
            T_brake = brake * self.p['max_brake'] / 4
            torque_net = T_drive - T_brake - tire_forces_x[i] * self.p['tire_radius']
            omega_dot.append(torque_net / self.p['wheel_inertia'])

        # --- Pack Derivatives ---
        d_pos = ca.mtimes(R_b2i, ca.vertcat(u, v, w))
        d_ang = ca.vertcat(p, q, r) 
        
        d_states = ca.vertcat(
            d_pos,
            d_ang,
            ca.vertcat(*dz_w),      # d(z_susp)/dt = vel_susp
            ca.vertcat(*omega),     # <--- FIXED: d(theta_wheel)/dt = omega
            ca.vertcat(u_dot, v_dot, w_dot),
            ca.vertcat(p_dot, q_dot, r_dot),
            ca.vertcat(*dz_w_dot),
            ca.vertcat(*omega_dot)
        )
        
        return d_states