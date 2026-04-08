# data/configs/tire_coeffs.py
# Hoosier R25B 16x7.5 on 10x7 rim
# Lateral/aligning coefficients from TTC PAC2002 .tir (Stackpole, Feb 2019)
# Longitudinal/combined slip: retained from previous fit (USE_MODE=2 in .tir — Fx zeroed)
# Thermal: retained (not in .tir format)
#
# FIX (this revision):
#   PDY2 sign corrected: +0.343535 → -0.343535
#
#   ROOT CAUSE: The Pacejka formula implemented in tire_model.py is:
#       Dy = PDY1 * (1 + PDY2 * dfz) * (1 - PDY3*γ²) * Fz * λ_μy
#   In this form NEGATIVE PDY2 gives degressive (diminishing-returns) grip.
#   The .tir file stores PDY2 with the opposite sign convention
#   (the .tir reader is expected to negate it before use).
#   The previous positive value made mu_y INCREASE with load → Ratio1 = 2.54,
#   which fails the sanity-check gate (1.2, 1.9) and is physically wrong.
#
#   With PDY2 = -0.343535:
#     Fz=500  N:  factor = 1 + (−0.343535)(−0.235) = 1.081  →  Dy ∝ 540
#     Fz=1000 N:  factor = 1 + (−0.343535)( 0.529) = 0.818  →  Dy ∝ 818
#     Fz=2000 N:  factor = 1 + (−0.343535)( 2.059) = 0.293  →  Dy ∝ 586
#     Ratio1 = 818/540 = 1.51  ✓ in (1.2, 1.9)
#     Ratio2 = 586/818 = 0.72  < Ratio1  ✓ (degressive confirmed)

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

    # *** FIX: negated from +0.343535.  The .tir sign convention is opposite
    #     to the tire_model.py formula  Dy = PDY1*(1+PDY2*dfz)*Fz.
    #     Positive PDY2 in that formula = PROGRESSIVE (mu increases with load)
    #     Negative PDY2                 = DEGRESSIVE  (mu decreases with load) ✓
    'PDY2':  -0.343535,     # was +0.343535 ← BUG (produced Ratio1=2.54)

    'PDY3':   3.89743,      # was  0.265  (camber sensitivity, large — R25B characteristic)
    'PEY1':   0.000,        # was -0.342  (zeroed in .tir)
    'PEY2':  -0.280762,     # was -0.122
    'PEY3':   0.70403,      # was  0.000  (new — camber asymmetry of curvature)
    'PEY4':  -0.478297,     # was  0.000  (new — camber sensitivity of asymmetry)
    'PKY1':  53.2421,       # was 15.324  ← CRITICAL: abs of -53.2421, 3.5× stiffer
    'PKY2':   2.38205,      # was  1.715  (load at peak stiffness / Fz0)
    'PKY3':   0.15,
    'PKY4':   2.0,
    'PHY1':  -0.0009,
    'PHY2':  -0.00082,
    'PVY1':   0.045,
    'PVY2':  -0.024,

    # ── Longitudinal coefficients ─────────────────────────────────────────────
    'PCX1':   1.685,
    'PDX1':   1.210,
    'PDX2':  -0.037,
    'PEX1':   0.344,
    'PEX2':   0.095,
    'PEX3':  -0.020,
    'PEX4':   0.0,
    'PKX1':  21.51,
    'PKX2':  13.49,
    'PKX3':  -0.41,
    'PHX1':   0.0,
    'PHX2':   0.0,
    'PVX1':   0.0,
    'PVX2':   0.0,

    # ── Combined slip weighting ───────────────────────────────────────────────
    'RBX1':  12.35,
    'RBX2':  -10.77,
    'RCX1':   1.092,
    'REX1':   0.344,
    'REX2':   0.095,
    'RHX1':   0.0,
    'RBY1':   6.461,
    'RBY2':   4.196,
    'RBY3':  -0.015,
    'RCY1':   1.081,
    'REY1':   0.0,
    'REY2':   0.0,
    'RHY1':   0.0,
    'RHY2':   0.0,
    'RVY1':   0.0,
    'RVY2':   0.0,
    'RVY3':   0.0,
    'RVY4':  14.0,
    'RVY5':   1.9,
    'RVY6':  10.0,

    # ── Aligning moment ───────────────────────────────────────────────────────
    'QBZ1':  10.904,
    'QBZ2':  -1.217,
    'QBZ3':  -0.412,
    'QBZ4':   0.0,
    'QBZ5':   0.0,
    'QCZ1':   1.178,
    'QDZ1':   0.1013,
    'QDZ2':  -0.009,
    'QDZ3':   0.0,
    'QDZ4':   0.0,
    'QEZ1':  -1.609,
    'QEZ2':   0.359,
    'QEZ3':   0.0,
    'QEZ4':   0.0,
    'QEZ5':  -2.097,
    'QHZ1':   0.0046,
    'QHZ2':   0.0026,
    'QHZ3':   0.1088,
    'QHZ4':   0.0,

    # ── Turn slip correction (MF6.2 extension) ────────────────────────────────
    'PDXP1':  0.4,
    'PDXP2':  0.0,
    'PDXP3':  0.0,
    'PKYP1':  1.0,
    'PDYP1':  0.4,
    'PDYP2':  0.0,
    'PDYP3':  0.0,
    'PDYP4':  0.0,
    'PHYP1':  1.0,
    'PHYP2':  0.15,
    'PHYP3':  0.0,
    'PHYP4':  -4.0,
    'PECP1':  0.5,
    'PECP2':  0.0,
    'QDTP1':  10.0,
    'QCRP1':  0.2,
    'QCRP2':  0.1,
    'QBRP1':  0.1,
    'QDRP1':  1.0,

    # ── Thermal parameters (5-node model) ────────────────────────────────────
    'T_OPT':  90.0,
    'T_ENV':  25.0,
    'BETA_T': 0.0008,       # K^-2  Gaussian peak width
}