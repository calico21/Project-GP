"""
Vehicle Parameter Configuration (FORMULA STUDENT SPEC)
----------------------------------------------------
Physical properties for a typical FSAE vehicle.
"""

vehicle_params = {
    # --- MASS & INERTIA ---
    'm': 250.0,           # Mass [kg] (Car + Driver)
    'Iz': 150.0,          # Yaw Inertia [kg*m^2] (Very low for snappy rotation)
    
    # --- GEOMETRY ---
    'lf': 0.82,           # CG to Front Axle [m] (Total wheelbase ~1.55m)
    'lr': 0.73,           # CG to Rear Axle [m]
    'track_width': 1.2,   # Track Width [m]
    'h_cg': 0.28,         # CG Height [m] (Very low)
    
    # --- AERODYNAMICS ---
    # FSAE cars have high Lift Coefficient (Cl) but small Area (A)
    'A': 1.0,             # Frontal Area [m^2]
    'Cl': 3.5,            # Downforce Coefficient (High downforce package)
    'Cd': 1.1,            # Drag Coefficient (High drag due to wings)
    
    # --- SUSPENSION LIMITS (Optimization Bounds) ---
    # Springs are still stiff relative to mass for aero support
    'min_spring': 20000.0,  # [N/m] (~110 lbs/in)
    'max_spring': 80000.0,  # [N/m] (~450 lbs/in)
    
    'min_damp': 500.0,      # [Ns/m]
    'max_damp': 4000.0,     # [Ns/m]
    
    'min_arb': 0.0,         # [Nm/deg]
    'max_arb': 800.0,       # [Nm/deg]
    
    # --- BRAKES ---
    'brake_bias': 0.60,     # % Front Bias (Usually higher in FSAE due to big weight transfer)
    'max_brake_torque': 800.0, # [Nm] Per wheel
    
    # --- POWERTRAIN ---
    # Restricted power (approx 80kW / 100hp limit for Electric/Combustion)
    'power_max': 80000.0,   # [W] 80 kW
    'redline': 12000.0,     # [RPM] (Motor or Bike Engine)
    'drive_wheels': 'RWD',
}

# Derived Parameters
vehicle_params['wb'] = vehicle_params['lf'] + vehicle_params['lr']
vehicle_params['mass_dist'] = vehicle_params['lr'] / vehicle_params['wb']