"""
Vehicle Parameter Configuration (FORMULA STUDENT SPEC - REALISTIC)
----------------------------------------------------------------
Physical properties for a typical FSAE vehicle.
"""

vehicle_params = {
    # --- MASS & INERTIA ---
    'm': 300.0,           # Mass [kg] (Car + Driver)
    'Iz': 150.0,          # Yaw Inertia [kg*m^2]
    
    # --- GEOMETRY ---
    'lf': 0.8525,           # CG to Front Axle [m]
    'lr': 0.68868,           # CG to Rear Axle [m]
    'track_width': 1.2,   # Track Width [m]
    'h_cg': 0.33,         # CG Height [m]
    
    # --- AERODYNAMICS ---
    # Adjusted for 1.5G target (Less downforce)
    'A': 1.1,             # Frontal Area [m^2]
    'Cl': 4.14,            # Downforce Coeff 
    'Cd': 0.8,            # Drag Coeff
    
    # --- POWERTRAIN LIMITS ---
    'power_max': 80000.0, # [W] 80 kW Engine Power
    'v_max': 29.2,        # [m/s] 105 km/h (Gearing Limit / Redline)
    'drive_wheels': 'RWD',

    # --- SUSPENSION LIMITS ---
    'min_spring': 20000.0,
    'max_spring': 80000.0,
    'min_damp': 500.0,
    'max_damp': 4000.0,
    'min_arb': 0.0,
    'max_arb': 800.0,
    
    # --- BRAKES ---
    'brake_bias': 0.60,     
    'max_brake_torque': 800.0, 
}

# Derived Parameters
vehicle_params['wb'] = vehicle_params['lf'] + vehicle_params['lr']
vehicle_params['mass_dist'] = vehicle_params['lr'] / vehicle_params['wb']