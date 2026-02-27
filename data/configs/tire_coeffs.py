# data/configs/tire_coeffs.py
# Full Pacejka MF6.2 coefficient set for Hoosier R20A 20x6-13
# Baseline values from TTC Round 8 — UPDATE after skidpad fitting

tire_coeffs = {
    # ── Reference conditions ─────────────────────────────────────────────────
    'FNOMIN':   1000.0,     # Nominal load Fz0 (N)
    'R0':       0.2032,     # Unloaded radius (m) — 8.0 inch
    'T_opt':    90.0,       # Optimal contact temp (°C)
    'P_nom':    1.2,        # Nominal inflation pressure (bar)
    'V0':       16.67,      # Nominal velocity for scaling (m/s) = 60 km/h

    # ── Lateral coefficients (MF6.2) ─────────────────────────────────────────
    'PCY1':   1.338,        # Shape factor
    'PDY1':   2.218,        # Peak friction at Fz0 (fitted from TTC)
    'PDY2':  -0.250,        # Load sensitivity of PDY1
    'PDY3':   0.265,        # Camber sensitivity of peak friction
    'PEY1':  -0.342,        # Curvature factor at Fz0
    'PEY2':  -0.122,        # Load sensitivity of PEY1
    'PEY3':   0.000,        # Asymmetry of curvature
    'PEY4':   0.000,        # Camber sensitivity of asymmetry
    'PKY1':  15.324,        # Maximum cornering stiffness (N/rad / Fz0)
    'PKY2':   1.715,        # Load at maximum cornering stiffness / Fz0
    'PKY3':   0.370,        # Camber sensitivity of cornering stiffness
    'PKY4':   2.000,        # Shape of cornering stiffness vs load
    'PHY1':  -0.0009,       # Plysteer horizontal shift at Fz0
    'PHY2':  -0.00082,      # Load sensitivity of PHY1
    'PVY1':   0.045,        # Conicity vertical shift at Fz0/Fz0
    'PVY2':  -0.024,        # Load sensitivity of PVY1

    # ── Longitudinal coefficients (MF6.2) ────────────────────────────────────
    'PCX1':   1.579,        # Shape factor
    'PDX1':   1.000,        # Peak friction at Fz0
    'PDX2':  -0.041,        # Load sensitivity
    'PDX3':   0.000,        # Camber sensitivity
    'PEX1':   0.312,        # Curvature factor at Fz0
    'PEX2':  -0.261,        # Load sensitivity of PEX1
    'PEX3':   0.000,        # Quadratic load sensitivity
    'PEX4':   0.000,        # Combined slip curvature sensitivity
    'PKX1':  21.687,        # Longitudinal slip stiffness / Fz
    'PKX2':  13.728,        # Load sensitivity of PKX1
    'PKX3':  -0.466,        # Exponential load sensitivity
    'PHX1':   0.000,        # Longitudinal horizontal shift
    'PHX2':   0.000,
    'PVX1':   0.000,
    'PVX2':   0.000,

    # ── Combined slip coefficients ───────────────────────────────────────────
    # Lateral reduction under longitudinal slip
    'RBY1':   7.143,
    'RBY2':   9.192,
    'RBY3':   0.000,
    'RCY1':   1.059,
    'REY1':  -0.496,
    'REY2':   0.000,
    'RHY1':   0.00947,
    'RHY2':   0.00975,
    'RVY1':   0.05187,
    'RVY2':   0.04551,
    'RVY3':  -0.025,
    'RVY4':  12.12,
    'RVY5':   1.9,
    'RVY6':  22.21,
    # Longitudinal reduction under lateral slip
    'RBX1':  13.046,
    'RBX2':   9.718,
    'RCX1':   0.9995,
    'REX1':   0.000,
    'REX2':   0.000,
    'RHX1':   0.000,

    # ── Aligning torque (MF6.2) ──────────────────────────────────────────────
    'QBZ1':  10.904,
    'QBZ2':  -1.896,
    'QBZ3':  -0.937,
    'QBZ4':   0.100,
    'QBZ5':  -0.100,
    'QCZ1':   1.180,
    'QDZ1':   0.092,
    'QDZ2':  -0.006,
    'QDZ3':   0.000,
    'QDZ4':   0.000,
    'QEZ1':  -8.865,
    'QEZ2':   0.000,
    'QEZ3':   0.000,
    'QEZ4':   0.254,
    'QEZ5':   0.000,
    'QHZ1':   0.0065,
    'QHZ2':   0.0056,

    # ── Thermal model ────────────────────────────────────────────────────────
    'mass':       10.0,     # Tyre carcass mass (kg)
    'm_gas':       0.05,    # Gas mass inside tyre (kg)
    'Cp':       1100.0,     # Rubber specific heat (J/kg·K)
    'Cv_gas':    718.0,     # Air Cv (J/kg·K)
    'h_conv':     50.0,     # External convection coefficient (W/m²·K)
    'h_conv_int': 30.0,     # Internal convection to gas (W/m²·K)
    'k_cond':    150.0,     # Radial thermal conductivity (W/m·K)
    'k_cond_lat': 85.0,     # Lateral thermal conductivity (W/m·K)
    'A_surf':      0.08,    # Contact patch area at Fz0 (m²)
    'q_roll':      0.03,    # Rolling resistance heat fraction

    # ── Transient dynamics ───────────────────────────────────────────────────
    'relaxation_length_x': 0.10,   # Longitudinal (m) at Fz0
    'relaxation_length_y': 0.25,   # Lateral (m) at Fz0
}