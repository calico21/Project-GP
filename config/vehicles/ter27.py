# config/vehicles/ter27.py
# Project-GP  —  Ter27 Formula Student 2027  (4WD)
# ═══════════════════════════════════════════════════════════════════════════════
# MASTER VEHICLE PARAMETER FILE — TER27 FOUR-WHEEL DRIVE
#
# PURPOSE: Initial design-space definition for the Ter27 4WD platform.
# Calibrated against validated OptimumKinematics & Alex.xlsx CAD geometry.
#
# KEY DIFFERENCES VS TER26 (RWD):
# ─────────────────────────────────
# · 4 × individual hub motors (~20 kW each, 80 kW total per FSG rules)
# · Front unsprung mass +4.5 kg/corner (motor + halfshaft + inverter bracket)
# · Weight distribution: ~45/55 F/R (calibrated to actual CAD CG at x = -847 mm)
# · Total mass: 280 kg (including 75 kg driver)
# · Lower yaw inertia: distributed motor mass closer to CG
# · Front anti-squat now physically meaningful (front wheels drive)
# · Torque vectoring fundamentally changes stability landscape
# · Drivetrain mode: 'awd' (activates DrivetrainDispatcher 4-motor path)
#
# CONVENTIONS:
#   All SI base units. Angles in degrees where documented.
#   Positive x = forward, positive z = up.
# ═══════════════════════════════════════════════════════════════════════════════

vehicle_params_ter27 = {

    # ════════════════════════════════════════════════════════════════════════
    # IDENTITY
    # ════════════════════════════════════════════════════════════════════════
    'car_id':              'ter27',
    'car_name':            'Ter27 4WD',
    'season':              2027,
    'drivetrain_mode':     'awd',       # Activates 4-motor DrivetrainDispatcher

    # ════════════════════════════════════════════════════════════════════════
    # MASS AND INERTIA
    # ════════════════════════════════════════════════════════════════════════
    # 4WD adds ~20 kg over RWD: 2 front motors (~3.5 kg each), 2 front
    # halfshafts (~1.5 kg each), front inverter bracket (~2 kg), wiring
    # harness delta (~3 kg), additional cooling loop (~5 kg).
    'total_mass':          280.0,   # kg  [incl. 75 kg driver]
    'm':                   280.0,   # kg  [Legacy alias]
    'mass_driver':          75.0,   # kg
    # sprung_mass computed below in derived section

    # Inertias — estimated from CAD mass redistribution study.
    'Ix':   52.0,    # kg·m²  Roll  (+15% vs Ter26: front drivetrain adds roll mass)
    'Iy':   92.0,    # kg·m²  Pitch (+8%: more total mass, similar distribution)
    'Iz':  140.0,    # kg·m²  Yaw   (-7%: distributed motors, no heavy rear diff)
    'Iw':    1.4,    # kg·m²  Wheel rot. inertia (heavier front hubs w/ motor)

    # CG — Calibrated with Alex.xlsx / OptimumKinematics:
    # Batalla = 1.540 m (contact_patch_l.x = -1540 mm)
    # CG real: x = -847 mm -> lf = 0.847 m, lr = 0.693 m (45% F / 55% R)
    'wheelbase':    1.540,  # m   [Calibrado con Alex.xlsx: antes 1.550]
    'lf':           0.847,  # m   [CG -> eje delantero: CG.x = -847 mm en OK]
    'lr':           0.693,  # m   [CG -> eje trasero: 1.540 - 0.847 m]
    'h_cg':         0.290,  # m   [Altura total CG: CG.z = 290 mm en OK]
    'h_cg_sprung':  0.310,  # m   [Altura masa suspendida CG]

    'track_front':  1.230,  # m   [Calibrado con OK: 2 * 0.615 m half track]
    'track_rear':   1.230,  # m   [Calibrado con OK: 2 * 0.615 m half track]

    # Masa no suspendida por esquina
    'unsprung_mass_f': 12.50,  # kg por esquina delantera (motor de cubo + mangueta)
    'unsprung_mass_r':  8.00,  # kg por esquina trasera

    # ════════════════════════════════════════════════════════════════════════
    # SUSPENSION GEOMETRY  (fixed kinematic — NOT in SuspensionSetup)
    # ════════════════════════════════════════════════════════════════════════

    # Alturas de centros de balanceo (Roll centre heights) calibradas con OK
    'h_rc_f':     0.020,    # m   [front.roll_center.z = 19.847 mm en OK]
    'h_rc_r':     0.040,    # m   [rear.roll_center.z = 40.029 mm en OK]
    'dh_rc_dz_f': 0.22,    # m/m  Tasa de migración RC
    'dh_rc_dz_r': 0.28,    # m/m

    # Motion ratios:
    # Delantero: 1.135 progresivo (+14.9% rising rate en compresión para aero)
    # Trasero: 1.150 lineal (desacoplando radios a 78/48 mm o 60/37 mm para tracción)
    'motion_ratio_f_poly': [1.135,  2.8, 0.0],
    'motion_ratio_r_poly': [1.150,  0.2, 0.0],

    # ── Static alignment at design ride height ─────────────────────────────
    'static_camber_f':  -1.50,  # deg  [Calibrado con OK: front.camber_angle_l = -1.50°]
    'static_camber_r':  -1.20,  # deg  [Calibrado con OK: rear.camber_angle_l = -1.20°]
    'static_toe_f':      0.00,  # deg  [front.toe_angle_l = 0.00°]
    'static_toe_r':      0.00,  # deg  [rear.toe_angle_l = 0.00°]
    'castor_f':          5.00,  # deg  [front.caster_angle_l = 5.00°]

    # ── Camber sensitivity ─────────────────────────────────────────────────
    'camber_gain_f': -0.805,  # deg/deg roll  [front.camber_angle_gain_roll_l = 0.805]
    'camber_gain_r': -0.692,  # deg/deg roll  [rear.camber_angle_gain_roll_l = 0.692]

    'camber_per_m_travel_f': -18.0,  # deg/m  [front gain heave: -0.018 deg/mm = -18 deg/m]
    'camber_per_m_travel_r': -29.0,  # deg/m  [rear gain heave: -0.029 deg/mm = -29 deg/m]

    # ── Bump steer ─────────────────────────────────────────────────────────
    'bump_steer_f':       0.000,  # rad/m   (target: zero / 0.001 deg/mm en OK)
    'bump_steer_r':       0.000,  # rad/m
    'bump_steer_quad_f':  0.000,  # rad/m²
    'bump_steer_quad_r':  0.000,  # rad/m²

    # ── Compliance steer ───────────────────────────────────────────────────
    'compliance_steer_f': -0.12,  # deg/kN  (stiffer front bushings for 4WD)
    'compliance_steer_r': -0.10,  # deg/kN

    # ── Ackermann ──────────────────────────────────────────────────────────
    'ackermann_factor': 0.0,   # Paralelo en diseño base

    # ── Anti-pitch geometry ────────────────────────────────────────────────
    'anti_squat':   0.398,  # fraction — rear anti-squat under acceleration
    'anti_squat_f': 0.150,  # fraction — FRONT anti-squat (NEW for 4WD)
    'anti_lift':    0.143,  # fraction — rear anti-lift under deceleration (14.35% en OK)
    'anti_dive_f':  0.565,  # fraction — front anti-dive under braking (56.45% en OK)
    'anti_dive_r':  0.150,  # fraction — rear anti-dive under braking

    # ════════════════════════════════════════════════════════════════════════
    # SPRING AND DAMPER RATES  (baseline — optimizer overrides)
    # ════════════════════════════════════════════════════════════════════════

    # Springs — calibrados según Alex.xlsx
    'spring_rate_f': 44000.0,   # N/m  [44.0 N/mm en Alex.xlsx]
    'spring_rate_r': 53000.0,   # N/m  [53.0 N/mm en Alex.xlsx; nota: bajar a ~22 kN/m con MR=1.15]

    'arb_rate_f':     400.0,    # N/m at wheel  (stiffer: manage front roll)
    'arb_rate_r':     300.0,    # N/m at wheel

    # Dampers — digressive bilinear model
    'damper_c_low_f':    2200.0,   # N·s/m  (higher: heavier front unsprung)
    'damper_c_low_r':    1800.0,   # N·s/m
    'damper_c_high_f':    900.0,   # N·s/m
    'damper_c_high_r':    700.0,   # N·s/m
    'damper_v_knee_f':      0.10,  # m/s
    'damper_v_knee_r':      0.10,  # m/s
    'damper_v_knee':        0.10,  # [Legacy alias]

    'rebound_ratio_f':      1.70,  # Higher rebound: control front weight transfer
    'rebound_ratio_r':      1.55,

    'damper_gas_force_f':  130.0,  # N
    'damper_gas_force_r':  120.0,  # N

    # ── Ride heights ───────────────────────────────────────────────────────
    'h_ride_f':      0.028,   # m  (ground clearance bajo morro)
    'h_ride_r':      0.028,   # m
    'h_ride_design': 0.035,   # m  [Legacy alias]

    # ── Bump stops ─────────────────────────────────────────────────────────
    'bump_stop_rate':   55000.0,  # N/m
    'bump_stop_engage':   0.025,  # m

    'lambda_stiffness': 2.0e-9,

    # ════════════════════════════════════════════════════════════════════════
    # AERODYNAMICS
    # ════════════════════════════════════════════════════════════════════════
    'Cl_ref':     4.50,   # Target: 8% more than Ter26 (4.14)
    'Cl':         4.50,
    'Cd_ref':     2.40,   # Slightly cleaner (improved diffuser design)
    'Cd':         0.78,
    'A_ref':      1.12,   # m²  Slightly larger frontal area (wider track)
    'A':          1.12,
    'h_aero_ref': 0.035,  # m

    'k_ground_f': 0.32,
    'k_ground_r': 0.48,

    # Aero balance
    'aero_split_f': 0.48,   # 48% carga frontal
    'aero_split_r': 0.52,   # 52% carga trasera
    'dCl_f_dtheta': 0.38,

    'rho_air': 1.225,

    # ════════════════════════════════════════════════════════════════════════
    # DRIVETRAIN — 4WD CONFIGURATION
    # ════════════════════════════════════════════════════════════════════════
    # 4 × AMK DD5-14 (o equivalente) motores de cubo.
    # Cada motor: ~20 kW peak, ~22 Nm peak en eje motor. Reducción ~10:1 -> ~220 Nm en rueda.
    'motor_peak_torque':            880.0,    # N·m  total en las 4 ruedas
    'motor_peak_torque_per_wheel':  220.0,    # N·m por rueda
    'motor_peak_power':           80000.0,    # W    (Límite de reglas FS: 80 kW total)
    'motor_peak_power_per_wheel': 20000.0,    # W por motor
    'motor_max_rpm':              20000.0,    # rpm
    'drivetrain_ratio':              10.0,    # Reducción planetaria integrada en mangueta
    'final_drive_ratio':             10.0,    # [Legacy alias]
    'wheel_radius':                 0.2032,   # m  (Hoosier 43075 R20, D = 406.4 mm en Alex.xlsx)
    'drivetrain_efficiency':          0.95,   # Direct drive en mangueta
    'm_drivetrain_eff':               8.0,    # kg  Inercia efectiva reducida

    # Diferencial electrónico (Torque Vectoring)
    'diff_lock_ratio': 0.0,   # 0.0 = cada rueda controlada independientemente

    # ════════════════════════════════════════════════════════════════════════
    # TORQUE VECTORING PARAMETERS (4WD-specific)
    # ════════════════════════════════════════════════════════════════════════
    'tv_yaw_gain':         0.8,    # —   Ganancia de momento de guiñada (0=off, 1=full)
    'tv_slip_limit':       0.12,   # —   Límite de slip ratio por rueda
    'tv_regen_max_frac':   0.30,   # —   Fracción máxima de par motor en regenerativa
    'tv_power_limit':    80000.0,  # W   Límite estricto de potencia eléctrica instantánea

    # ════════════════════════════════════════════════════════════════════════
    # BRAKES
    # ════════════════════════════════════════════════════════════════════════
    # Reparto de frenada hidráulico mecánico de Alex.xlsx (76.59% delantero)
    'brake_bias_f':        0.766,   # [front.brake_bias = 76.59% en Alex.xlsx]
    'ideal_brake_balance': False,
    'max_brake_torque':   900.0,    # N·m
    'brake_mu':            0.40,

    # ════════════════════════════════════════════════════════════════════════
    # TYRE ENVIRONMENT
    # ════════════════════════════════════════════════════════════════════════
    'T_env':   20.0,   # °C
    'P_nom':    1.2,   # bar
    'T_target': 85.0,  # °C

    # ════════════════════════════════════════════════════════════════════════
    # STEERING
    # ════════════════════════════════════════════════════════════════════════
    'max_steer_angle':  0.35,   # rad
    'steer_ratio':      4.37,   # [front.steering_ratio = 4.370 en Alex.xlsx]

    # ════════════════════════════════════════════════════════════════════════
    # SIMULATION / INTEGRATOR
    # ════════════════════════════════════════════════════════════════════════
    'physics_hz':     200,
    'substeps':         5,
    'integrator': 'implicit_midpoint',
}

# ── Derived parameters ──────────────────────────────────────────────────────
vehicle_params_ter27['wb'] = (vehicle_params_ter27['lf']
                              + vehicle_params_ter27['lr'])
vehicle_params_ter27['mass_dist'] = (vehicle_params_ter27['lr']
                                     / vehicle_params_ter27['wb'])
vehicle_params_ter27['m_us_total'] = (
    2 * vehicle_params_ter27['unsprung_mass_f']
    + 2 * vehicle_params_ter27['unsprung_mass_r']
)
vehicle_params_ter27['sprung_mass'] = (
    vehicle_params_ter27['total_mass']
    - vehicle_params_ter27['m_us_total']
)

# ── Validated 3D hardpoints (Ter27 4WD Layout) ───────────────────────────
# Coordenadas 3D exactas extraídas de Alex.xlsx / OptimumKinematics (en metros).
# Origen [0, 0, 0] situado en el parche de contacto delantero en posición estática.
vehicle_params_ter27['hardpoints_f'] = {
    'lca_in_f':    [ 0.1618, 0.1551, 0.1099],   # Lower wishbone chassis front pivot (CHAS_LowFor)
    'lca_in_r':    [-0.1600, 0.1600, 0.1300],   # Lower wishbone chassis rear pivot (CHAS_LowAft)
    'lbj':         [ 0.0023, 0.5834, 0.1227],   # Lower Ball Joint (outboard) (UPRI_LowPnt)
    'uca_in_f':    [ 0.1200, 0.2451, 0.2670],   # Upper wishbone chassis front pivot (CHAS_UppFor)
    'uca_in_r':    [-0.1202, 0.2443, 0.2580],   # Upper wishbone chassis rear pivot (CHAS_UppAft)
    'ubj':         [-0.0115, 0.5556, 0.2800],   # Upper Ball Joint (outboard) (UPRI_UppPnt)
    'pushrod_out': [-0.0051, 0.5135, 0.2930],   # Pushrod anclado a trapecio superior (NSMA_PPAttPnt_L)
    'rocker_piv':  [-0.0257, 0.1937, 0.5438],   # Eje de giro balancín chasis (CHAS_RocPiv_L)
    'spring_in':   [-0.1814, 0.1442, 0.5827],   # Anclaje amortiguador a chasis (CHAS_AttPnt_L)
    'tie_in':      [ 0.0500, 0.1448, 0.1445],   # Barra dirección chasis / cremallera (CHAS_TiePnt)
    'tie_out':     [ 0.0700, 0.5646, 0.1500],   # Barra dirección mangueta (UPRI_TiePnt)
}

vehicle_params_ter27['hardpoints_r'] = {
    'lca_in_f':    [ 0.1500, 0.2400, 0.1182],   # Lower wishbone chassis front pivot (CHAS_LowFor)
    'lca_in_r':    [-0.1500, 0.2400, 0.1130],   # Lower wishbone chassis rear pivot (CHAS_LowAft)
    'lbj':         [ 0.0000, 0.5768, 0.1127],   # Lower Ball Joint (outboard) (UPRI_LowPnt)
    'uca_in_f':    [ 0.1500, 0.2400, 0.2375],   # Upper wishbone chassis front pivot (CHAS_UppFor)
    'uca_in_r':    [-0.1500, 0.2400, 0.2782],   # Upper wishbone chassis rear pivot (CHAS_UppAft)
    'ubj':         [ 0.0000, 0.5200, 0.2800],   # Upper Ball Joint (outboard) (UPRI_UppPnt)
    'pushrod_out': [ 0.0060, 0.4940, 0.2938],   # Pushrod anclado a trapecio superior (NSMA_PPAttPnt_L)
    'rocker_piv':  [ 0.0397, 0.1072, 0.4629],   # Eje de giro balancín chasis (CHAS_RocPiv_L)
    'spring_in':   [-0.1000, 0.0500, 0.3917],   # Anclaje amortiguador a chasis (CHAS_AttPnt_L)
    'tie_in':      [-0.0950, 0.2400, 0.1594],   # Tirante de convergencia chasis (CHAS_TiePnt)
    'tie_out':     [-0.0800, 0.5900, 0.1658],   # Tirante convergencia mangueta (UPRI_TiePnt)
}

def get_design_bounds():
    """Returns the lower and upper bounds for the Ter27 setup space."""
    import jax.numpy as jnp
    from models.vehicle_dynamics import SETUP_LB, SETUP_UB
    
    # Return the bounds as JAX arrays for the optimizer
    return jnp.array(SETUP_LB), jnp.array(SETUP_UB)