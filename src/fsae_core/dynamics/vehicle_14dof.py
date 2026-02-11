import casadi as ca
import numpy as np

class Vehicle14DOF:
    """
    14-Degrees-of-Freedom Vehicle Dynamics Model.
    
    DOFs:
    1-6:  Chassis Position & Orientation (x, y, z, roll, pitch, yaw)
    7-10: Wheel/Suspension Vertical Positions (z_fl, z_fr, z_rl, z_rr)
    11-14: Wheel Spin Speeds (w_fl, w_fr, w_rl, w_rr)
    
    This class calculates the time-derivative of the state vector: x_dot = f(x, u)
    """

    def __init__(self, params):
        self.p = params
        
    def get_dynamics(self, states, controls):
        """
        Symbolic equations of motion.
        
        Args:
            states: CasADi MX/SX vector (28 elements: 14 positions + 14 velocities)
            controls: CasADi MX/SX vector (3 elements: steer_angle, throttle, brake_bias)
        
        Returns:
            d_states: Time derivative of the state vector
        """
        # --- 1. Unpack States ---
        # Positions / Orientations
        pos_x, pos_y, pos_z = states[0], states[1], states[2]
        roll, pitch, yaw    = states[3], states[4], states[5]
        
        # Suspension compressions (relative to chassis hardpoints usually, 
        # but here treating as absolute vertical pos of unsprung mass for rigorous 14-DOF)
        z_w = [states[6], states[7], states[8], states[9]] # fl, fr, rl, rr
        
        # Wheel rotations (radians)
        # theta_w = ... (We mostly care about speeds, so we skip tracking absolute wheel angle)
            
        # Velocities
        u, v, w = states[14], states[15], states[16] # Chassis body velocities
        p, q, r = states[17], states[18], states[19] # Chassis rotational rates
        dz_w    = [states[20], states[21], states[22], states[23]] # Susp vertical vel
        omega   = [states[24], states[25], states[26], states[27]] # Wheel spin speeds
        
        # --- 2. Unpack Controls ---
        delta_steer = controls[0] # Steering angle at wheel (assume simplified Ackermann or direct map)
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
        # Calculate tire velocities and slip angles
        # Positions of wheels relative to CG
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
        tire_forces_z = [] # Normal load
        
        for i in range(4):
            # A. Calculate Normal Load (Fz)
            # Fz = Static + Aero + Suspension Force
            # Simplified Suspension Force: k * (z_chassis_at_wheel - z_wheel) + c * velocity_diff
            
            # Kinematics: Vertical pos of chassis corner
            # z_corner = z_cg - y_arm * sin(roll) + x_arm * sin(pitch) (Small angle approx)
            wx, wy = wheel_pos_body[i]
            z_chassis_corner = pos_z - wy * roll + wx * pitch
            
            susp_deflection = z_chassis_corner - z_w[i]
            susp_velocity   = (w - wy * p + wx * q) - dz_w[i]
            
            # Spring + Damper force
            F_susp = self.p['k_spring'][i] * susp_deflection + self.p['c_damper'][i] * susp_velocity
            
            # Add Downforce (simple map based on velocity squared)
            # Distribute Aero forces 40/60 usually, simplified here
            F_aero = 0.5 * 1.225 * self.p['Cl'] * (u**2 + v**2) * (0.25) 
            
            # Tire stiffness (vertical)
            # F_tire_spring = k_tire * (Radius - z_wheel_ground)
            # Assuming flat ground at z=0
            tire_compression = self.p['tire_radius'] - z_w[i] 
            Fz_tire = ca.fmax(0, self.p['k_tire'] * tire_compression) # Tire vertical spring
            
            # Store forces acting ON THE CHASSIS from suspension
            # For Unsprung mass Eq: m_u * z_dd = F_tire - F_susp - m_u * g
            tire_forces_z.append(Fz_tire) 
            
            # B. Calculate Slips
            # Velocity at the tire contact patch
            vx_tire = u - wy * r
            vy_tire = v + wx * r
            
            # Steering angle for this wheel
            delta = delta_steer if i < 2 else 0
            
            # Slip Angle (Alpha)
            # alpha = atan(vy / vx) - delta
            alpha = ca.arctan2(vy_tire, vx_tire) - delta
            
            # Slip Ratio (Kappa)
            # kappa = (omega * R - vx) / vx
            kappa = (omega[i] * self.p['tire_radius'] - vx_tire) / (vx_tire + 0.1)
            
            # C. Magic Formula (Simplified Call)
            # Need to import/define Pacejka logic here or externally
            # For conciseness, using a placeholder linear/peak model
            mu_y = self.p['Dy'] * ca.sin(self.p['Cy'] * ca.arctan(self.p['By'] * alpha))
            mu_x = self.p['Dx'] * ca.sin(self.p['Cx'] * ca.arctan(self.p['Bx'] * kappa))
            
            # Combined slip reduction (friction circle)
            # Fx = Fz * mu_x; Fy = Fz * mu_y ... (simplified)
            Fx = Fz_tire * mu_x
            Fy = Fz_tire * mu_y
            
            # Transform to Body Frame
            # Fx_body = Fx * cos(delta) - Fy * sin(delta)
            # Fy_body = Fx * sin(delta) + Fy * cos(delta)
            Fx_b = Fx * ca.cos(delta) - Fy * ca.sin(delta)
            Fy_b = Fx * ca.sin(delta) + Fy * ca.cos(delta)
            
            tire_forces_x.append(Fx_b)
            tire_forces_y.append(Fy_b)

        # --- 5. Equations of Motion (Newton-Euler) ---
        
        # Sum of Forces
        Sum_Fx = sum(tire_forces_x) - 0.5 * 1.225 * self.p['Cd'] * u**2 # Drag
        Sum_Fy = sum(tire_forces_y)
        Sum_Fz = sum([F - self.p['mass']*9.81 for F in tire_forces_z]) # Placeholder
        
        # Sum of Moments
        # M = r x F
        M_z = 0
        M_y = 0 # Pitch moment from aero/accel
        M_x = 0 # Roll moment
        
        for i in range(4):
            wx, wy = wheel_pos_body[i]
            Fx, Fy = tire_forces_x[i], tire_forces_y[i]
            
            M_z += wx * Fy - wy * Fx
            # Pitch/Roll moments depend on suspension forces pushing on chassis
            # F_susp calculated earlier acts on chassis
            # ... (Full moment arm calc would go here)

        # 6. Accelerations (solving Newton-Euler)
        # m(u_dot - v*r + w*q) = Sum_Fx
        # I * omega_dot + ... = Sum_M
        
        mass = self.p['mass']
        u_dot = Sum_Fx/mass + v*r - w*q
        v_dot = Sum_Fy/mass - u*r + w*p
        w_dot = Sum_Fz/mass + u*q - v*p # Simplified
        
        Ixx, Iyy, Izz = self.p['Ixx'], self.p['Iyy'], self.p['Izz']
        p_dot = M_x / Ixx # Euler coupling terms omitted for brevity
        q_dot = M_y / Iyy
        r_dot = M_z / Izz
        
        # 7. Unsprung Mass Dynamics
        # m_u * z_dd = F_tire_vertical - F_suspension - m_u * g
        dz_w_dot = []
        for i in range(4):
             # Re-calc F_susp and Fz_tire for clarity or reuse
             accel = (tire_forces_z[i] - 0) / self.p['unsprung_mass'] # Simplified
             dz_w_dot.append(accel)

        # 8. Wheel Spin Dynamics
        # I_w * w_dot = T_drive - T_brake - Fx * R
        omega_dot = []
        for i in range(4):
            T_drive = throttle * self.p['max_torque'] / 2 if i >= 2 else 0 # RWD
            T_brake = brake * self.p['max_brake']
            torque_net = T_drive - T_brake - tire_forces_x[i] * self.p['tire_radius']
            omega_dot.append(torque_net / self.p['wheel_inertia'])

        # --- Pack Derivatives ---
        # [dx, dy, dz, droll, dpitch, dyaw] -> Rotation matrix * [u,v,w]
        # But for small angles/body rates:
        d_pos = ca.mtimes(R_b2i, ca.vertcat(u, v, w))
        d_ang = ca.vertcat(p, q, r) # Euler rates approx body rates (valid for small angles)
        
        d_states = ca.vertcat(
            d_pos,
            d_ang,
            ca.vertcat(*dz_w),      # d(z_susp)/dt = vel_susp
            ca.vertcat(*d_pos[:0]), # Placeholder for ignored pos
            ca.vertcat(u_dot, v_dot, w_dot),
            ca.vertcat(p_dot, q_dot, r_dot),
            ca.vertcat(*dz_w_dot),
            ca.vertcat(*omega_dot)
        )
        
        return d_states