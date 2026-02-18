# tire_coeffs.py
# Updated with fits from "RawData_Cornering_Matlab_SI_Round9"

tire_coeffs = {
    # --- Scaling & Limits ---
    'FNOMIN': 1000.0,   # Nominal Load [N] (Reference for all params)
    
    # --- Lateral (Fy) - The Critical Updates ---
    'PCY1': 1.50,       # Shape Factor (C). Data avg: ~1.5 (was 1.35)
    'PDY1': 2.1,       # Peak Friction (Mu). Data avg at 1kN: ~2.8 (was 1.60)
    'PDY2': -0.25,      # Load Sensitivity. Grip drops high loads. (was -0.10)
    
    # Cornering Stiffness = Fz * (PKY1 + PKY2 * dfz)
    # Calculated from B_Stiff * C_Shape * D_Peak in your data
    'PKY1': 35.0,       # Base Stiffness. (was 25.0)
    'PKY2': -1.5,       # Stiffness drops slightly with load
    'PEY1': 0.5,        # Curvature at peak (E). Data avg: ~0.4-0.6
    
    # --- Longitudinal (Fx) - Estimated scaling based on Fy ---
    # Usually Fx is 85-90% of Fy on bias-ply FSAE tires
    'PCX1': 1.50,
    'PDX1': 2.60,       # Slightly less than lateral grip
    'PDX2': -0.20,
    'PKX1': 40.0,       # Long. stiffness is usually higher
    
    # --- Combined Slip & Transient ---
    'RHX1': 0.0,        
    'PTX1': 0.20,       # Relaxation Length [m]
    
    # --- Thermal ---
    'T_OPT': 70.0,      
    'T_WIDTH': 20.0,    
    'C_HEAT': 900.0,    
    'K_COOL': 15.0      
}