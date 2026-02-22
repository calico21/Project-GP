# tire_coeffs.py
# Adjusted for Realistic Tarmac Grip (~1.5G peak)

tire_coeffs = {
    # --- Scaling & Limits ---
    'FNOMIN': 1000.0,   # Nominal Load [N]
    
    # --- Lateral (Fy) ---
    'PDY1': 2.218,    # Peak lateral friction - FITTED
    'PDY2': -0.250,   # Load sensitivity - CORRECTED (raw fit gave -0.011, physically wrong)
    'PCY1': 1.45,     # Replace with your mean C_Shape from master_pacejka
    'PEY1': -0.15,    # Replace with your mean E_Curve from master_pacejka
    'PKY1': 25.0,     # Keep generic until properly fitted from stiffness vs load curve
    
    # Longitudinal - no data, use conservative estimates
    'PDX1': 2.44,     # PDY1 * 1.10 (longitudinal typically 5-15% higher)
    'PCX1': 1.60,
    'PEX1': -0.50,
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