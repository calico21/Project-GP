# config/vehicles/ter27.py
# Project-GP  —  Ter27 Formula Student 2027  (4WD)
# ═══════════════════════════════════════════════════════════════════════════════
# MASTER VEHICLE PARAMETER FILE — TER27 FOUR-WHEEL DRIVE
#
# PURPOSE: Initial design-space definition for the Ter27 4WD platform.
# All geometry/suspension values are STARTING ESTIMATES — the MORL optimizer
# will explore the full bounded space to recommend optimal targets.
#
# KEY DIFFERENCES VS TER26 (RWD):
# ─────────────────────────────────
# · 4 × individual hub motors (~20 kW each, 80 kW total per FSG rules)
# · Front unsprung mass +4.5 kg/corner (motor + halfshaft + inverter bracket)
# · More centred weight distribution: ~48/52 F/R (was 45/55)
# · Higher total mass: ~320 kg (additional drivetrain components)
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
    'total_mass':          320.0,   # kg  [incl. 75 kg driver]
    'm':                   320.0,   # kg  [Legacy alias]
    'mass_driver':          75.0,   # kg
    # sprung_mass computed below in derived section

    # Inertias — estimated from CAD mass redistribution study.
    # Roll inertia increases (wider mass at front axle).
    # Yaw inertia DECREASES — distributed motors move mass inward vs
    # single rear motor + chain drive + diff assembly.
    'Ix':   52.0,    # kg·m²  Roll  (+15% vs Ter26: front drivetrain adds roll mass)
    'Iy':   92.0,    # kg·m²  Pitch (+8%: more total mass, similar distribution)
    'Iz':  140.0,    # kg·m²  Yaw   (-7%: distributed motors, no heavy rear diff)
    'Iw':    1.4,    # kg·m²  Wheel rot. inertia (heavier front hubs w/ motor)

    # CG — more centred due to front drivetrain mass pulling CG forward.
    # Target: 48% front / 52% rear (vs Ter26's 45/55).
    'lf':   0.806,   # m  CG → front axle  [48% front: lf = 0.52 × wb]
    'lr':   0.744,   # m  CG → rear axle   [52% rear:  lr = 0.48 × wb]
    'h_cg': 0.310,   # m  Total CG height  (lower: front motors below CG plane)
    'h_cg_sprung': 0.330,  # m  Sprung mass CG height

    'track_front':  1.220,  # m  [Slightly wider for front motor packaging]
    'track_rear':   1.200,  # m  [Wider than Ter26 for symmetry]
    'wheelbase':    1.550,  # m  [Unchanged — chassis platform continuity]

    # Front unsprung mass jumps significantly: hub motor (~3.5 kg) +
    # halfshaft (~1.5 kg) + upright reinforcement (~0.5 kg) = +5.5 kg/corner.
    'unsprung_mass_f': 12.50,  # kg per front corner  (was 7.74 on Ter26)
    'unsprung_mass_r':  8.00,  # kg per rear corner   (slight increase: wider hub)

    # ════════════════════════════════════════════════════════════════════════
    # SUSPENSION GEOMETRY  (fixed kinematic — NOT in SuspensionSetup)
    # These are the INITIAL ESTIMATES. The optimizer will recommend the
    # optimal values for the optimizable subset (anti-geometry, alignment).
    # ════════════════════════════════════════════════════════════════════════

    # Roll centre heights — starting slightly higher front for load transfer
    # management with increased front unsprung mass.
    'h_rc_f':     0.045,    # m   (Ter26: 0.040)
    'h_rc_r':     0.055,    # m   (Ter26: 0.060)
    'dh_rc_dz_f': 0.22,    # m/m  RC migration rate
    'dh_rc_dz_r': 0.28,    # m/m

    # Motion ratio — preliminary, bellcrank geometry TBD
    'motion_ratio_f_poly': [1.12,  2.8, 0.0],   # Slightly lower MR: more wheel travel
    'motion_ratio_r_poly': [1.14,  2.2, 0.0],

    # ── Static alignment at design ride height ─────────────────────────────
    # THESE ARE OPTIMIZER STARTING POINTS — will be overridden by MORL output.
    'static_camber_f':  -2.5,   # deg  (more negative: compensate heavier front)
    'static_camber_r':  -1.8,   # deg
    'static_toe_f':     -0.08,  # deg  (slight toe-in for stability)
    'static_toe_r':      0.00,  # deg  (neutral — optimizer will explore)
    'castor_f':          5.5,   # deg  (slightly more caster for self-centering)

    # ── Camber sensitivity ─────────────────────────────────────────────────
    'camber_gain_f': -0.75,   # deg/deg roll  (slightly less: stiffer front)
    'camber_gain_r': -0.60,   # deg/deg roll

    'camber_per_m_travel_f': -22.0,  # deg/m wheel travel
    'camber_per_m_travel_r': -18.0,  # deg/m

    # ── Bump steer ─────────────────────────────────────────────────────────
    'bump_steer_f':       0.000,  # rad/m   (target: zero)
    'bump_steer_r':       0.000,  # rad/m
    'bump_steer_quad_f':  0.000,  # rad/m²
    'bump_steer_quad_r':  0.000,  # rad/m²

    # ── Compliance steer ───────────────────────────────────────────────────
    'compliance_steer_f': -0.12,  # deg/kN  (stiffer front bushings for 4WD)
    'compliance_steer_r': -0.10,  # deg/kN

    # ── Ackermann ──────────────────────────────────────────────────────────
    'ackermann_factor': 0.0,   # OPTIMIZER TARGET — will explore [-0.5, 1.0]

    # ── Anti-pitch geometry ────────────────────────────────────────────────
    # CRITICAL FOR 4WD: front anti-squat now matters under acceleration.
    # Starting values are conservative — optimizer will find optimal.
    'anti_squat':   0.35,   # fraction — rear anti-squat under acceleration
    'anti_squat_f': 0.15,   # fraction — FRONT anti-squat (NEW for 4WD)
    'anti_lift':    0.20,   # fraction — rear anti-lift under deceleration
    'anti_dive_f':  0.35,   # fraction — front anti-dive under braking
    'anti_dive_r':  0.15,   # fraction — rear anti-dive under braking

    # ════════════════════════════════════════════════════════════════════════
    # SPRING AND DAMPER RATES  (baseline — optimizer overrides)
    # ════════════════════════════════════════════════════════════════════════

    # Springs — stiffer front to manage increased unsprung mass & motor torque.
    'spring_rate_f': 40000.0,   # N/m  (Ter26: 35030)
    'spring_rate_r': 48000.0,   # N/m  (Ter26: 52540 — lighter rear now)

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
    'h_ride_f':      0.028,   # m  (slightly lower: more aero ground effect)
    'h_ride_r':      0.028,   # m
    'h_ride_design': 0.035,   # m  [Legacy alias]

    # ── Bump stops ─────────────────────────────────────────────────────────
    'bump_stop_rate':   55000.0,  # N/m  (slightly stiffer: heavier car)
    'bump_stop_engage':   0.025,  # m

    'lambda_stiffness': 2.0e-9,

    # ════════════════════════════════════════════════════════════════════════
    # AERODYNAMICS
    # ════════════════════════════════════════════════════════════════════════
    # Aero package evolution: slightly more downforce, better balance.
    'Cl_ref':     4.50,   # Target: 8% more than Ter26 (4.14)
    'Cl':         4.50,
    'Cd_ref':     2.40,   # Slightly cleaner (improved diffuser design)
    'Cd':         0.78,
    'A_ref':      1.12,   # m²  Slightly larger frontal area (wider track)
    'A':          1.12,
    'h_aero_ref': 0.035,  # m

    'k_ground_f': 0.32,
    'k_ground_r': 0.48,

    # More forward aero balance to complement 4WD traction advantage.
    'aero_split_f': 0.48,   # (Ter26: 0.45) — more front downforce
    'aero_split_r': 0.52,
    'dCl_f_dtheta': 0.38,

    'rho_air': 1.225,

    # ════════════════════════════════════════════════════════════════════════
    # DRIVETRAIN — 4WD CONFIGURATION
    # ════════════════════════════════════════════════════════════════════════
    # 4 × AMK DD5-14 (or equivalent) hub motors.
    # Each motor: ~20 kW peak, ~22 Nm peak torque at motor shaft.
    # Planetary reduction ratio ~10:1 per wheel → ~220 Nm at wheel each.
    # Total at wheels: 4 × 220 = 880 Nm (vs Ter26's 450 Nm single motor).
    'motor_peak_torque':    880.0,    # N·m  total at all 4 wheels
    'motor_peak_torque_per_wheel': 220.0,  # N·m per wheel
    'motor_peak_power':   80000.0,    # W  (FSG limit: 80 kW total)
    'motor_peak_power_per_wheel': 20000.0,  # W per motor
    'motor_max_rpm':      20000.0,    # rpm  (hub motor, higher RPM than chain drive)
    'drivetrain_ratio':      10.0,    # Planetary reduction per wheel
    'final_drive_ratio':     10.0,    # [Legacy alias]
    'wheel_radius':         0.2032,   # m  (same tire: Hoosier 43075 R20)
    'drivetrain_efficiency':  0.95,   # Higher: no chain losses, direct drive
    'm_drivetrain_eff':       8.0,    # kg  Lower effective inertia (no chain/sprocket)

    # No mechanical differential — torque vectoring is purely electronic.
    'diff_lock_ratio': 0.0,   # 0.0 = open (each wheel independently controlled)

    # ════════════════════════════════════════════════════════════════════════
    # TORQUE VECTORING PARAMETERS (4WD-specific)
    # ════════════════════════════════════════════════════════════════════════
    'tv_yaw_gain':         0.8,    # —   Yaw moment gain (0=off, 1=full authority)
    'tv_slip_limit':       0.12,   # —   Max traction slip ratio per wheel
    'tv_regen_max_frac':   0.30,   # —   Max regen fraction of peak torque
    'tv_power_limit':    80000.0,  # W   Total instantaneous power cap

    # ════════════════════════════════════════════════════════════════════════
    # BRAKES
    # ════════════════════════════════════════════════════════════════════════
    'brake_bias_f':        0.55,    # More balanced: 4WD regen on all wheels
    'ideal_brake_balance': False,
    'max_brake_torque':   900.0,    # N·m  (slightly more: heavier car)
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
    'steer_ratio':      4.00,   # Slightly quicker ratio for 4WD agility

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

# ── Placeholder 3D hardpoints (Ter27 — TBD from CAD) ───────────────────────
# Copied from Ter26 with track width adjustments. Replace once CAD is done.
vehicle_params_ter27['hardpoints_f'] = {
    'lca_in_f':   [ 0.180, 0.160, 0.235],
    'lca_in_r':   [-0.180, 0.160, 0.235],
    'lbj':        [ 0.000, 0.120, 0.570],
    'uca_in_f':   [ 0.140, 0.310, 0.188],
    'uca_in_r':   [-0.140, 0.310, 0.188],
    'ubj':        [ 0.000, 0.380, 0.550],
    'pushrod_out': [ 0.000, 0.180, 0.560],
    'rocker_piv':  [ 0.000, 0.280, 0.345],
    'spring_in':   [ 0.150, 0.280, 0.160],
}

vehicle_params_ter27['hardpoints_r'] = {
    'lca_in_f':   [ 0.200, 0.160, 0.225],
    'lca_in_r':   [-0.200, 0.160, 0.225],
    'lbj':        [ 0.000, 0.120, 0.560],
    'uca_in_f':   [ 0.150, 0.310, 0.175],
    'uca_in_r':   [-0.150, 0.310, 0.175],
    'ubj':        [ 0.000, 0.380, 0.540],
    'pushrod_out': [ 0.000, 0.180, 0.550],
    'rocker_piv':  [ 0.000, 0.280, 0.325],
    'spring_in':   [ 0.150, 0.280, 0.155],
}

def get_design_bounds():
    """Returns the lower and upper bounds for the Ter27 setup space."""
    import jax.numpy as jnp
    from models.vehicle_dynamics import SETUP_LB, SETUP_UB
    
    # Return the bounds as JAX arrays for the optimizer
    return jnp.array(SETUP_LB), jnp.array(SETUP_UB)