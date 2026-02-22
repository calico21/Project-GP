"""
Vehicle Parameter Configuration (FORMULA STUDENT SPEC - REALISTIC)
----------------------------------------------------------------
Physical properties updated from actual vehicle dimensions.
"""

vehicle_params = {
    # --- MASS & INERTIA ---
    'm': 300.0,           # Total Mass [kg]
    'Iz': 150.0,          # Yaw Inertia [kg*m^2] (Assumed baseline)
    
    # --- GEOMETRY ---
    'lf': 0.8525,         # CG to Front Axle [m] (45% Front Weight)
    'lr': 0.6975,         # CG to Rear Axle [m] (55% Rear Weight)
    'track_width': 1.19,  # Track Width [m] (Averaged 1200mm Fr / 1180mm Rr)
    'h_cg': 0.33,         # Total CG Height [m]
    'h_rc_f': 0.040,      # Front Roll Center [m]
    'h_rc_r': 0.06028,    # Rear Roll Center [m]
    
    # --- SUSPENSION & KINEMATIC COUPLINGS ---
    'motion_ratio_f': 1.14,
    'motion_ratio_r': 1.16,
    'unsprung_mass_f': 7.74, # [kg] per corner
    'unsprung_mass_r': 7.76, # [kg] per corner
    
    # --- AERODYNAMICS ---
    'A': 1.1,             # Frontal Area [m^2]
    'Cl': 4.14,           # Downforce Coeff (Total 2285.70N @ 105km/h)
    'Cd': 0.8,            # Drag Coeff
    'aero_bias_front': 0.45, # 45% Front Downforce
    
    # --- POWERTRAIN LIMITS ---
    'power_max': 80000.0, # [W] 80 kW
    'v_max': 29.2,        # [m/s] 105 km/h
    'drive_wheels': 'RWD',

    # --- SETUP OPTIMIZATION BOUNDS ---
    # Front Spring bounds (Searching around your 35,030 N/m baseline)
    'min_spring_f': 25000.0,
    'max_spring_f': 45000.0,
    
    # Rear Spring bounds (Searching around your 52,540 N/m baseline)
    'min_spring_r': 40000.0,
    'max_spring_r': 65000.0,
    
    'min_damp': 1000.0,
    'max_damp': 4000.0,
    
    # ARB Bounds (Let the AI discover the optimal Rear ARB!)
    'min_arb_f': 0.0,
    'max_arb_f': 800.0,
    'min_arb_r': 0.0,     # 0 = Disconnected
    'max_arb_r': 500.0,   # Realistic maximum stiffness for an FSAE rear ARB
    
    # Add these under your --- SETUP OPTIMIZATION BOUNDS --- section
    'min_camber_f': -3.0,
    'max_camber_f': -0.5,
    'min_camber_r': -2.5,
    'max_camber_r': -0.5,

    # --- BRAKES ---
    'brake_bias': 0.60,     
    'max_brake_torque': 800.0, 
}

# Derived Parameters
vehicle_params['wb'] = vehicle_params['lf'] + vehicle_params['lr']
vehicle_params['mass_dist'] = vehicle_params['lr'] / vehicle_params['wb']