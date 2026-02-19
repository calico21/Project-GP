# tire_coeffs.py
# Adjusted for Realistic Tarmac Grip (~1.5G peak)

tire_coeffs = {
    # --- Scaling & Limits ---
    'FNOMIN': 1000.0,   # Nominal Load [N]
    
    # --- Lateral (Fy) ---
    'PCY1': 1.45,       # Shape Factor
    'PDY1': 1.45,       # Peak Friction (Mu). Reduced from 2.1 to 1.45
    'PDY2': -0.15,      # Load Sensitivity (Grip drops slightly with load)
    
    'PKY1': 25.0,       # Cornering Stiffness
    'PKY2': -1.0,
    'PEY1': 0.5,        # Curvature
    
    # --- Longitudinal (Fx) ---
    'PCX1': 1.50,
    'PDX1': 1.50,       # Traction matches cornering
    'PDX2': -0.15,
    'PKX1': 30.0,
    
    # --- Combined Slip & Transient ---
    'RHX1': 0.0,        
    'PTX1': 0.20,       # Relaxation Length [m]
    
    # --- Thermal ---
    'T_OPT': 70.0,      
    'T_WIDTH': 20.0,    
    'C_HEAT': 900.0,    
    'K_COOL': 15.0      
}