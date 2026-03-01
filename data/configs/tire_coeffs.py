# data/configs/tire_coeffs.py
# Hoosier R25B 16x7.5 on 10x7 rim
# Lateral/aligning coefficients from TTC PAC2002 .tir (Stackpole, Feb 2019)
# Longitudinal/combined slip: retained from previous fit (USE_MODE=2 in .tir — Fx zeroed)
# Thermal: retained (not in .tir format)

tire_coeffs = {
    # ── Reference conditions — direct from .tir ──────────────────────────────
    'FNOMIN':   654.0,      # was 1000.0  ← CRITICAL: scales all Pacejka output
    'R0':       0.2045,     # was 0.2032  (UNLOADED_RADIUS)
    'T_opt':    90.0,
    'P_nom':    0.834,      # was 1.2 bar  (83441 Pa / 100000)
    'V0':       11.176,     # was 16.67   (LONGVL = 11.176 m/s = 25 mph)

    # ── Lateral coefficients — from .tir ────────────────────────────────────
    # PDY1/PKY1: magnitude only — sign convention differs (PAC2002 right-tyre)
    'PCY1':   1.53041,      # was 1.338
    'PDY1':   2.40275,      # was 2.218   (abs of -2.40275)
    'PDY2':   0.343535,     # was -0.250  (positive in .tir — load softening)
    'PDY3':   3.89743,      # was  0.265  (camber sensitivity, large — R25B characteristic)
    'PEY1':   0.000,        # was -0.342  (zeroed in .tir)
    'PEY2':  -0.280762,     # was -0.122
    'PEY3':   0.70403,      # was  0.000  (new — camber asymmetry of curvature)
    'PEY4':  -0.478297,     # was  0.000  (new — camber sensitivity of asymmetry)
    'PKY1':  53.2421,       # was 15.324  ← CRITICAL: abs of -53.2421, 3.5× stiffer
    'PKY2':   2.38205,      # was  1.715  (load at peak stiffness / Fz0)
    'PKY3':   1.36502,      # was  0.370  (camber sensitivity of Ky, large)
    'PKY4':   2.000,        # unchanged
    'PHY1':  -9.87381e-05,  # was -0.0009 (plysteer shift, very small — good)
    'PHY2':   7.11965e-04,  # was -0.00082
    'PHY3':   0.147449,     # was  0.000  (new — camber-induced horizontal shift)
    'PVY1':   0.0441197,    # was  0.045
    'PVY2':   0.0124743,    # was -0.024
    'PVY3':   1.54004,      # was  0.000  (new — camber vertical shift, significant)
    'PVY4':  -1.71672,      # was  0.000  (new — camber×load vertical shift)

    # ── Longitudinal — RETAINED (USE_MODE=2 in .tir, all Fx coeffs zeroed) ──
    'PCX1':   1.579,
    'PDX1':   1.000,
    'PDX2':  -0.041,
    'PDX3':   0.000,
    'PEX1':   0.312,
    'PEX2':  -0.261,
    'PEX3':   0.000,
    'PEX4':   0.000,
    'PKX1':  21.687,
    'PKX2':  13.728,
    'PKX3':  -0.466,
    'PHX1':   0.000,
    'PHX2':   0.000,
    'PVX1':   0.000,
    'PVX2':   0.000,

    # ── Combined slip — RETAINED (zeroed in .tir) ────────────────────────────
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
    'RBX1':  13.046,
    'RBX2':   9.718,
    'RCX1':   0.9995,
    'REX1':   0.000,
    'REX2':   0.000,
    'RHX1':   0.000,

    # ── Aligning torque — from .tir ──────────────────────────────────────────
    'QBZ1':   8.22843,      # was 10.904
    'QBZ2':   2.98676,      # was -1.896
    'QBZ3':  -3.57739,      # was -0.937
    'QBZ4':  -0.429117,     # was  0.100
    'QBZ5':   0.433125,     # was -0.100
    'QCZ1':   1.41359,      # was  1.180
    'QDZ1':   0.152526,     # was  0.092  (larger trail → more self-aligning)
    'QDZ2':  -0.0381101,    # was -0.006
    'QDZ3':   0.387762,     # was  0.000  (new — camber effect on trail)
    'QDZ4':  -3.95699,      # was  0.000  (new — camber² effect, large)
    'QEZ1':  -0.239731,     # was -8.865  (much less aggressive curvature falloff)
    'QEZ2':   1.29253,      # was  0.000
    'QEZ3':  -1.21298,      # was  0.000
    'QEZ4':   0.197579,     # was  0.254
    'QEZ5':   0.244,        # was  0.000  (new — camber×sign(alpha) curvature term)
    'QHZ1':  -0.00101749,   # was  0.0065
    'QHZ2':   3.78319e-04,  # was  0.0056
    'QHZ3':  -0.0405191,    # was  0.000  (new — camber shift of trail)
    'QHZ4':   0.0185463,    # was  0.000  (new)

    # ── Thermal — RETAINED (not in .tir format) ──────────────────────────────
    'mass':       10.0,
    'm_gas':       0.05,
    'Cp':       1100.0,
    'Cv_gas':    718.0,
    'h_conv':     50.0,
    'h_conv_int': 30.0,
    'k_cond':    150.0,
    'k_cond_lat': 85.0,
    'A_surf':      0.08,
    'q_roll':      0.03,

    # ── Transient — RETAINED (PTY1/PTY2 zeroed in .tir) ─────────────────────
    'relaxation_length_x': 0.10,
    'relaxation_length_y': 0.25,
}