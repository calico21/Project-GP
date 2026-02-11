import numpy as np

def calculate_pacejka(slip, Fz, params):
    """
    Standard Pacejka Magic Formula implementation.
    Args:
        slip: Tire slip angle or ratio
        Fz: Vertical load (N)
        params: Dictionary of tire coefficients (B, C, D, E)
    """
    # Load sensitivity: Friction coeff decays as load increases
    # D is peak friction force
    mu = params.get('a1', 1.5) * (1 - params.get('load_sensitivity', 0.0) * Fz)
    D = mu * Fz 
    
    B = params.get('stiffness', 10.0)
    C = params.get('shape', 1.3)
    E = params.get('curvature', 1.0)
    
    # The Core Formula
    # y = D * sin(C * arctan(B*x - E*(B*x - arctan(B*x))))
    y = D * np.sin(C * np.arctan(B * slip - E * (B * slip - np.arctan(B * slip))))
    
    return y
