# config/vehicles/ter27.py
# Project-GP — Ter27 Formula Student 2027 (4WD)
# ═══════════════════════════════════════════════════════════════════════════════
# MASTER VEHICLE PARAMETER FILE — TER27 FOUR-WHEEL DRIVE
# Calibrado con exportación OptimumKinematics (TeR27_OptimumK.xlsx)
# ═══════════════════════════════════════════════════════════════════════════════

vehicle_params_ter27 = {

    # ════════════════════════════════════════════════════════════════════════
    # IDENTIDAD Y TREN MOTRIZ
    # ════════════════════════════════════════════════════════════════════════
    'car_id':              'ter27',
    'car_name':            'Ter27 4WD',
    'season':              2027,
    'drivetrain_mode':     'awd',       # Activa DrivetrainDispatcher 4 motores

    # ════════════════════════════════════════════════════════════════════════
    # MASAS E INERCIAS
    # ════════════════════════════════════════════════════════════════════════
    'total_mass':          290.0,   # kg (con piloto de 75 kg)
    'm':                   290.0,   # kg (alias legacy)
    'mass_driver':          75.0,   # kg

    'Ix':   52.0,    # kg·m²  Roll
    'Iy':   92.0,    # kg·m²  Pitch
    'Iz':  140.0,    # kg·m²  Yaw
    'Iw':    0.20,   # kg·m²  Inercia rotacional por rueda (llanta + neumático + disco)

    # Posición del Centro de Gravedad (45% F / 55% R sobre batalla de 1.540 m)
    'wheelbase':    1.540,  # m   [Reference Distance en OK: 1540 mm]
    'lf':           0.847,  # m   [CG -> Eje delantero: 0.55 * wb]
    'lr':           0.693,  # m   [CG -> Eje trasero:   0.45 * wb]
    'h_cg':         0.290,  # m   [Center of Gravity Height: 290 mm en OK]
    'h_cg_sprung':  0.310,  # m   [Altura CG masa suspendida]

    'track_front':  1.230,  # m   [2 * 615 mm Half Track en OK]
    'track_rear':   1.230,  # m   [2 * 615 mm Half Track en OK]

    'unsprung_mass_f': 12.50,  # kg por esquina delantera (motor de cubo + mangueta)
    'unsprung_mass_r':  8.00,  # kg por esquina trasera

    # ════════════════════════════════════════════════════════════════════════
    # GEOMETRÍA DE SUSPENSIÓN Y CINEMÁTICA
    # ════════════════════════════════════════════════════════════════════════
    'h_rc_f':     0.020,    # m   Front roll centre height
    'h_rc_r':     0.040,    # m   Rear roll centre height
    'dh_rc_dz_f': 0.22,     # m/m
    'dh_rc_dz_r': 0.28,     # m/m

    # Motion ratios: delantero progresivo (+14.9%), trasero linealizado
    'motion_ratio_f_poly': [1.135,  2.8, 0.0],
    'motion_ratio_r_poly': [1.150,  0.2, 0.0],

    # Alineación estática en altura de diseño
    'static_camber_f':  -1.50,  # deg [Static Camber en OK]
    'static_camber_r':  -1.20,  # deg [Static Camber en OK]
    'static_toe_f':      0.00,  # deg [Static Toe en OK]
    'static_toe_r':      0.00,  # deg [Static Toe en OK]
    'castor_f':          5.00,  # deg

    # Ganancias de caída
    'camber_gain_f': -0.805,  # deg/deg roll
    'camber_gain_r': -0.692,  # deg/deg roll
    'camber_per_m_travel_f': -18.0,  # deg/m heave (-0.018 deg/mm)
    'camber_per_m_travel_r': -29.0,  # deg/m heave (-0.029 deg/mm)

    'bump_steer_f':       0.000,  # rad/m
    'bump_steer_r':       0.000,  # rad/m
    'bump_steer_quad_f':  0.000,  # rad/m²
    'bump_steer_quad_r':  0.000,  # rad/m²

    'compliance_steer_f': -0.12,  # deg/kN
    'compliance_steer_r': -0.10,  # deg/kN

    'ackermann_factor': 0.0,   # Paralelo en diseño base
    'steer_ratio':      4.37,   # Ratio efectivo cremallera (101.6 mm/rev en OK)

    # Anti-geometrías
    'anti_squat':   0.398,  # Trasero en aceleración
    'anti_squat_f': 0.150,  # Delantero en tracción 4WD
    'anti_lift':    0.143,  # Trasero en deceleración
    'anti_dive_f':  0.565,  # Delantero en frenada
    'anti_dive_r':  0.150,

    # ════════════════════════════════════════════════════════════════════════
    # MUELLES, ESTABILIZADORAS Y AMORTIGUADORES
    # ════════════════════════════════════════════════════════════════════════
    'spring_rate_f': 44000.0,   # N/m [44 N/mm en OK]
    'spring_rate_r': 53000.0,   # N/m [53 N/mm en OK]

    'arb_rate_f':     5000.0,   # N/m [U-Bar 5 N/mm en OK]
    'arb_rate_r':     5000.0,   # N/m [U-Bar 5 N/mm en OK]

    'damper_c_low_f':    2200.0,   # N·s/m
    'damper_c_low_r':    1800.0,   # N·s/m
    'damper_c_high_f':    900.0,   # N·s/m
    'damper_c_high_r':    700.0,   # N·s/m
    'damper_v_knee_f':      0.10,  # m/s
    'damper_v_knee_r':      0.10,  # m/s
    'damper_v_knee':        0.10,  # [Legacy alias]

    'rebound_ratio_f':      1.70,
    'rebound_ratio_r':      1.55,
    'damper_gas_force_f':  130.0,  # N
    'damper_gas_force_r':  120.0,  # N

    'h_ride_f':      0.028,   # m
    'h_ride_r':      0.028,   # m
    'h_ride_design': 0.035,   # m  [Legacy alias]

    'bump_stop_rate':   55000.0,  # N/m
    'bump_stop_engage':   0.025,  # m
    'lambda_stiffness': 2.0e-9,

    # ════════════════════════════════════════════════════════════════════════
    # AERODINÁMICA
    # ════════════════════════════════════════════════════════════════════════
    'Cl_ref':     4.50,
    'Cl':         4.50,
    'Cd_ref':     2.40,   # Calibrado drag total
    'Cd':         2.40,   # Sincronizado
    'A_ref':      1.12,   # m²
    'A':          1.12,
    'h_aero_ref': 0.035,  # m

    'k_ground_f': 0.32,
    'k_ground_r': 0.48,
    'aero_split_f': 0.48,
    'aero_split_r': 0.52,
    'dCl_f_dtheta': 0.38,
    'rho_air': 1.225,

    # ════════════════════════════════════════════════════════════════════════
    # PROPULSIÓN 4WD Y REPARTO
    # ════════════════════════════════════════════════════════════════════════
    'motor_peak_torque':            880.0,    # N·m total (4 ruedas)
    'motor_peak_torque_per_wheel':  220.0,    # N·m por rueda
    'motor_peak_power':           80000.0,    # W (Límite FS: 80 kW)
    'motor_peak_power_per_wheel': 20000.0,    # W
    'motor_max_rpm':              20000.0,    # rpm
    'drivetrain_ratio':              10.0,    # Planetaria en mangueta
    'final_drive_ratio':             10.0,    # [Legacy alias]
    'wheel_radius':                 0.2032,   # m (Tire Diameter: 406.4 mm / 2)
    'drivetrain_efficiency':          0.95,
    'm_drivetrain_eff':               8.0,    # kg
    'diff_lock_ratio':                0.0,    # Diferencial electrónico abierto (TV)
    'drive_bias_f':                   0.35,   # [Drive Bias: 35% en OK]

    # Torque Vectoring
    'tv_yaw_gain':         0.8,
    'tv_slip_limit':       0.12,
    'tv_regen_max_frac':   0.30,
    'tv_power_limit':    80000.0,

    # ════════════════════════════════════════════════════════════════════════
    # FRENOS
    # ════════════════════════════════════════════════════════════════════════
    'brake_bias_f':        0.7659,  # [Brake Bias: 76.59% en OK]
    'ideal_brake_balance': False,
    'max_brake_torque':   900.0,    # N·m
    'brake_mu':            0.40,

    # Neumáticos entorno
    'T_env':   20.0,
    'P_nom':    1.2,
    'T_target': 85.0,
    'max_steer_angle': 0.35,

    # Simulación
    'physics_hz':     200,
    'substeps':         5,
    'integrator': 'implicit_midpoint',
}

# ── Parámetros derivados ───────────────────────────────────────────────────
vehicle_params_ter27['wb'] = vehicle_params_ter27['lf'] + vehicle_params_ter27['lr']
vehicle_params_ter27['mass_dist'] = vehicle_params_ter27['lr'] / vehicle_params_ter27['wb']
vehicle_params_ter27['m_us_total'] = (
    2 * vehicle_params_ter27['unsprung_mass_f']
    + 2 * vehicle_params_ter27['unsprung_mass_r']
)
vehicle_params_ter27['sprung_mass'] = (
    vehicle_params_ter27['total_mass']
    - vehicle_params_ter27['m_us_total']
)

# ── HARDPOINTS 3D EXACTOS (OptimumK / TeR27_OptimumK.xlsx) ────────────────
# En metros [m]. Origen local X=0 en el eje de rueda correspondiente.
vehicle_params_ter27['hardpoints_f'] = {
    'lca_in_f':    [ 0.16183,  0.15508,  0.10985],  # CHAS_LowFor
    'lca_in_r':    [-0.16000,  0.16000,  0.13000],  # CHAS_LowAft
    'lbj':         [ 0.00227,  0.58337,  0.12265],  # UPRI_LowPnt
    'uca_in_f':    [ 0.11997,  0.24508,  0.26700],  # CHAS_UppFor
    'uca_in_r':    [-0.12024,  0.24430,  0.25795],  # CHAS_UppAft
    'ubj':         [-0.01150,  0.55563,  0.28000],  # UPRI_UppPnt
    'pushrod_out': [-0.00561,  0.51343,  0.29302],  # NSMA_PPAttPnt_L
    'rocker_piv':  [-0.03148,  0.17988,  0.55391],  # CHAS_RocPiv_L
    'spring_in':   [-0.18718,  0.13026,  0.59272],  # CHAS_AttPnt_L
    'tie_in':      [ 0.05000,  0.14478,  0.14450],  # CHAS_TiePnt
    'tie_out':     [ 0.07000,  0.56460,  0.15000],  # UPRI_TiePnt
    'rocker_axis': [-0.03148,  0.20614,  0.58748],  # CHAS_RocAxi_L
    'rocker_rod':  [ 0.02821,  0.17507,  0.55767],  # ROCK_RodPnt_L
    'rocker_coi':  [-0.03148,  0.13026,  0.59272],  # ROCK_CoiPnt_L
    'antiroll_in': [ 0.06852,  0.15625,  0.61239],  # CHAS_PivPnt_L (U-Bar)
    'antiroll_out':[-0.00148,  0.15625,  0.57239],  # NSMA_UBarAttPnt_L
}

vehicle_params_ter27['hardpoints_r'] = {
    'lca_in_f':    [ 0.15000,  0.24000,  0.11820],  # CHAS_LowFor
    'lca_in_r':    [-0.15000,  0.24000,  0.11300],  # CHAS_LowAft
    'lbj':         [ 0.00000,  0.57678,  0.11265],  # UPRI_LowPnt
    'uca_in_f':    [ 0.15000,  0.24000,  0.23750],  # CHAS_UppFor
    'uca_in_r':    [-0.15000,  0.24000,  0.27820],  # CHAS_UppAft
    'ubj':         [ 0.00000,  0.52000,  0.28000],  # UPRI_UppPnt
    'pushrod_out': [ 0.00596,  0.49404,  0.29380],  # NSMA_PPAttPnt_L
    'rocker_piv':  [ 0.03965,  0.10722,  0.46285],  # CHAS_RocPiv_L
    'spring_in':   [-0.10000,  0.05000,  0.39166],  # CHAS_AttPnt_L
    'tie_in':      [-0.09500,  0.24000,  0.15940],  # CHAS_TiePnt
    'tie_out':     [-0.08000,  0.59000,  0.16580],  # UPRI_TiePnt
    'rocker_axis': [ 0.01817,  0.11946,  0.49515],  # CHAS_RocAxi_L
    'rocker_rod':  [ 0.06912,  0.11600,  0.47912],  # ROCK_RodPnt_L
    'rocker_coi':  [ 0.02964,  0.05000,  0.47788],  # ROCK_CoiPnt_L
    'antiroll_in': [-0.05238,  0.08290,  0.36283],  # CHAS_PivPnt_L (U-Bar)
    'antiroll_out':[ 0.02578,  0.08290,  0.46285],  # NSMA_UBarAttPnt_L
}

def get_design_bounds():
    """Límites de diseño para optimizador MORL."""
    import jax.numpy as jnp
    from models.vehicle_dynamics import SETUP_LB, SETUP_UB
    return jnp.array(SETUP_LB), jnp.array(SETUP_UB)