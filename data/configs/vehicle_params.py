# data/configs/vehicle_params.py
# Project-GP  —  Ter26 Formula Student 2026
# ═══════════════════════════════════════════════════════════════════════════════
# MASTER VEHICLE PARAMETER FILE
# Merged with Team Ter26 CAD specifics (300 kg spec, 80 kW power, empirical aero)
#
# UPGRADE LOG (this revision)
# ────────────────────────────
# · Added all 28 SuspensionSetup parameter counterparts as fallback defaults.
#   vehicle_dynamics.make_setup_from_params(vehicle_params) now produces a
#   fully-specified SuspensionSetup without relying on DEFAULT_SETUP fill-ins.
# · Digressive damper: split c_low / c_high / v_knee / rebound_ratio per axle.
# · Per-axle ride height (h_ride_f / h_ride_r) replaces single h_ride_design.
# · Static camber + toe kept as separate optimizable defaults.
# · Caster angle, bump_steer_quad (nonlinear K&C term) added.
# · All legacy alias keys preserved for backward compat with physics_server,
#   inertia_fitting, scripts/run_analysis, and tests.
# ═══════════════════════════════════════════════════════════════════════════════

vehicle_params = {

    # ════════════════════════════════════════════════════════════════════════
    # MASS AND INERTIA
    # ════════════════════════════════════════════════════════════════════════
    'total_mass':          300.0,   # kg  [Total mass incl. driver]
    'm':                   300.0,   # kg  [Legacy alias]
    'mass_driver':          75.0,   # kg
    'sprung_mass':         268.9,   # kg  [DERIVED — 300 - 2×7.74 - 2×7.76]

    'Ix':   45.0,    # kg·m²  Roll axis  (K&C rig estimate, ±5%)
    'Iy':   85.0,    # kg·m²  Pitch axis
    'Iz':  150.0,    # kg·m²  Yaw axis   (baseline; refine via inertia_fitting)
    'Iw':    1.2,    # kg·m²  Wheel rotational inertia (rim + tire + brake disc)

    # CG locations  (from CAD, validated against corner weight data)
    'lf':   0.8525,  # m  CG → front axle  [45% front weight distribution]
    'lr':   0.6975,  # m  CG → rear axle   [55% rear weight distribution]
    'h_cg': 0.330,   # m  Total CG height  (driver seated, ballast included)
    'h_cg_sprung': 0.350,  # m  Sprung mass CG height

    'track_front':  1.200,  # m  [1200 mm — from CAD hub face to hub face]
    'track_rear':   1.180,  # m  [1180 mm]
    'wheelbase':    1.550,  # m  [lf + lr = 0.8525 + 0.6975]

    'unsprung_mass_f': 7.74,   # kg per front corner  (upright + wheel + brake)
    'unsprung_mass_r': 7.76,   # kg per rear corner

    # ════════════════════════════════════════════════════════════════════════
    # SUSPENSION GEOMETRY  (fixed kinematic parameters — NOT in SuspensionSetup)
    # These are determined by the physical build and cannot be adjusted at the
    # circuit. SuspensionSetup carries the optimizable subset.
    # ════════════════════════════════════════════════════════════════════════

    # Roll centre heights
    'h_rc_f':     0.040,    # m  Front roll centre at design ride height
    'h_rc_r':     0.060,    # m  Rear roll centre
    'dh_rc_dz_f': 0.20,    # m/m  RC rise per metre of front bump travel
    'dh_rc_dz_r': 0.30,    # m/m  RC rise per metre of rear bump travel

    # Motion ratio polynomial  MR(z) = a0 + a1·z + a2·z²
    # Fitted from suspension K&C rig sweep (±50 mm wheel travel)
    'motion_ratio_f_poly': [1.14,  2.5, 0.0],
    'motion_ratio_r_poly': [1.16,  2.0, 0.0],

    # ── Static alignment at design ride height (SuspensionSetup defaults) ──
    'static_camber_f':  -2.0,   # deg   (negative = lean-in, FS convention)
    'static_camber_r':  -1.5,   # deg
    'static_toe_f':     -0.10,  # deg   (negative = toe-in)
    'static_toe_r':     -0.15,  # deg
    'castor_f':          5.0,   # deg   Front caster angle (aligning moment)

    # ── Camber sensitivity (K&C rig data) ──────────────────────────────────
    'camber_gain_f': -0.80,   # deg camber / deg body roll  (front)
    'camber_gain_r': -0.65,   # deg camber / deg body roll  (rear)

    'camber_per_m_travel_f': -25.0,  # deg camber / m wheel travel  (front)
    'camber_per_m_travel_r': -20.0,  # deg camber / m wheel travel  (rear)

    # ── Bump steer ─────────────────────────────────────────────────────────
    'bump_steer_f':       0.000,  # rad/m  linear coefficient   (front)
    'bump_steer_r':       0.000,  # rad/m  linear coefficient   (rear)
    'bump_steer_quad_f':  0.000,  # rad/m² quadratic coefficient (from K&C)
    'bump_steer_quad_r':  0.000,  # rad/m² quadratic coefficient

    # ── Compliance steer (lateral push test) ───────────────────────────────
    'compliance_steer_f': -0.15,  # deg/kN  (negative = toe-in under lateral load)
    'compliance_steer_r': -0.10,  # deg/kN

    # ── Ackermann geometry ──────────────────────────────────────────────────
    'ackermann_factor': 0.0,   # 0=parallel steer, 1=full Ackermann, -1=reverse

    # ── Anti-pitch geometry ─────────────────────────────────────────────────
    # These are build-fixed via pushrod/bellcrank pivot locations.
    # Moved into SuspensionSetup as optimizable fractions to allow
    # the optimizer to find the minimum lap time across FSAE tracks.
    'anti_squat':   0.30,   # fraction  [0=no anti, 1=full anti]
    'anti_lift':    0.20,   # fraction  [rear lift under decel]
    'anti_dive_f':  0.40,   # fraction  [front dive under braking]
    'anti_dive_r':  0.10,   # fraction  [rear anti-dive under braking]

    # ════════════════════════════════════════════════════════════════════════
    # SPRING AND DAMPER RATES
    # All parameters here have direct counterparts in SuspensionSetup.
    # vehicle_dynamics.make_setup_from_params() maps these to the 28-vector.
    # ════════════════════════════════════════════════════════════════════════

    # Spring rates (at the spring, before motion ratio)
    'spring_rate_f': 35030.0,   # N/m  (baseline from corner weight + target freq)
    'spring_rate_r': 52540.0,   # N/m

    # Anti-roll bar rates (at wheel, torsional stiffness / MR²)
    'arb_rate_f':     200.0,    # N/m at wheel per side
    'arb_rate_r':     150.0,    # N/m at wheel per side

    # ── Digressive damper (Horstman bilinear model) ─────────────────────────
    # See vehicle_dynamics.digressive_damper_force() for full model.
    # F(v) = c_low·v / (1 + v/v_knee) + c_high·v  [bump]
    #        ρ·c_low·v / (1 - v/v_knee) + c_high·v  [rebound, ρ=rebound_ratio]
    'damper_c_low_f':    1800.0,   # N·s/m  low-speed damping coefficient — front
    'damper_c_low_r':    1600.0,   # N·s/m  low-speed damping coefficient — rear
    'damper_c_high_f':    720.0,   # N·s/m  high-speed viscous term — front
    'damper_c_high_r':    640.0,   # N·s/m  high-speed viscous term — rear
    'damper_v_knee_f':      0.10,  # m/s    digressive knee velocity — front
    'damper_v_knee_r':      0.10,  # m/s    digressive knee velocity — rear
    # Legacy single-key aliases (used by older modules / sanity_checks)
    'damper_v_knee':        0.10,  # m/s    [Legacy alias — maps to both axles]

    # Rebound damping ratio (rebound force = ratio × bump force at same |v|)
    # Typical Öhlins/Penske FSAE spec: 1.5–1.8 (more rebound than bump)
    'rebound_ratio_f':      1.60,  # —  [dimensionless, must be ≥ 1.0]
    'rebound_ratio_r':      1.60,  # —

    # Damper gas spring preload (nitrogen charge, measured at design length)
    'damper_gas_force_f':  120.0,  # N  (at spring — divide by MR for wheel equiv)
    'damper_gas_force_r':  120.0,  # N

    # ── Ride heights ────────────────────────────────────────────────────────
    # h_ride = distance from ground to hub centreline at design load.
    # Enters aero model as ground-effect reference.
    'h_ride_f':      0.030,   # m  front ride height (optimizable)
    'h_ride_r':      0.030,   # m  rear ride height  (optimizable)
    'h_ride_design': 0.040,   # m  [Legacy alias — nominal design target]

    # ── Bump stops ──────────────────────────────────────────────────────────
    # Soft contact at full suspension compression — NOT in SuspensionSetup
    # (build-fixed, not a circuit-adjustable parameter).
    'bump_stop_rate':   50000.0,  # N/m  stiffness once contact is made
    'bump_stop_engage':   0.025,  # m    compression at which bump stop contacts

    # ── Centering penalty weight (optimizer regulariser) ───────────────────
    'lambda_stiffness': 2.0e-9,  # penalty weight for setup deviating from baseline

    # ════════════════════════════════════════════════════════════════════════
    # AERODYNAMICS
    # ════════════════════════════════════════════════════════════════════════
    'Cl_ref':     4.14,   # Lift coefficient at h_aero_ref (downforce: positive)
    'Cl':         4.14,   # [Legacy alias]
    'Cd_ref':     2.50,   # Drag coefficient at h_aero_ref
    'Cd':         0.80,   # [Legacy alias — low-speed value]
    'A_ref':      1.10,   # m²  Frontal reference area
    'A':          1.10,   # m²  [Legacy alias]
    'h_aero_ref': 0.040,  # m   Reference ride height for Cl_ref / Cd_ref

    'k_ground_f': 0.30,   # Front ground effect sensitivity (ΔCl/ΔRH normalised)
    'k_ground_r': 0.45,   # Rear ground effect sensitivity

    'aero_split_f': 0.45,   # Fraction of total downforce on front axle
    'aero_split_r': 0.55,   # Fraction on rear axle
    'dCl_f_dtheta': 0.35,   # Front Cl sensitivity to pitch angle [per rad]

    'rho_air': 1.225,   # kg/m³  ISA sea level (correct for FSG altitude ~100m)

    # ════════════════════════════════════════════════════════════════════════
    # DRIVETRAIN
    # ════════════════════════════════════════════════════════════════════════
    'motor_peak_torque':    450.0,    # N·m  at wheel (after final drive)
    'motor_peak_power':   80000.0,    # W    (80 kW — EV class)
    'motor_max_rpm':       6000.0,    # rpm
    'drivetrain_ratio':       4.5,    # Final drive ratio (chain + gearbox combined)
    'final_drive_ratio':      4.5,    # [Legacy alias]
    'wheel_radius':         0.2032,   # m    (8-inch rim, loaded radius at 1.2 bar)
    'drivetrain_efficiency':  0.92,   # —    Accounts for chain losses
    'm_drivetrain_eff':      12.0,    # kg   Equivalent translational inertia

    # Differential  (build-fixed spool; 0.0=open, 1.0=locked spool, 0.3–0.7=LSD)
    # Also appears in SuspensionSetup[23] as an optimizable parameter.
    'diff_lock_ratio': 1.0,   # baseline: locked spool

    # ════════════════════════════════════════════════════════════════════════
    # BRAKES
    # ════════════════════════════════════════════════════════════════════════
    'brake_bias_f':        0.60,    # Fraction of total brake force on front axle
    'ideal_brake_balance': False,   # True = dynamic pressure-proportioning
    'max_brake_torque':   800.0,    # N·m  total at wheels
    'brake_mu':            0.40,    # —    Brake pad friction coefficient

    # ════════════════════════════════════════════════════════════════════════
    # TYRE ENVIRONMENT
    # ════════════════════════════════════════════════════════════════════════
    'T_env':   20.0,   # °C  Ambient temperature (FSG June average)
    'P_nom':    1.2,   # bar Cold inflation pressure (Hoosier spec)
    'T_target': 85.0,  # °C  Target operating temperature for thermal objective

    # ════════════════════════════════════════════════════════════════════════
    # STEERING
    # ════════════════════════════════════════════════════════════════════════
    'max_steer_angle':  0.35,   # rad  Physical rack limit
    'steer_ratio':      4.20,   # —    Steering wheel angle / rack angle

    # ════════════════════════════════════════════════════════════════════════
    # SIMULATION / INTEGRATOR SETTINGS  (not vehicle physics — used by server)
    # ════════════════════════════════════════════════════════════════════════
    'physics_hz':     200,     # Hz   Server tick rate
    'substeps':         5,     # —    Picard substeps per tick (dt_sub = 1ms)
    'integrator': 'implicit_midpoint',   # algorithm selector
}

# ── Derived parameters (computed once at import) ────────────────────────────
vehicle_params['wb']          = vehicle_params['lf'] + vehicle_params['lr']
vehicle_params['mass_dist']   = vehicle_params['lr'] / vehicle_params['wb']
vehicle_params['m_us_total']  = (2 * vehicle_params['unsprung_mass_f']
                                  + 2 * vehicle_params['unsprung_mass_r'])
vehicle_params['sprung_mass'] = (vehicle_params['total_mass']
                                  - vehicle_params['m_us_total'])

# ════════════════════════════════════════════════════════════════════════
# 3D KINEMATIC HARDPOINTS (For 3D Visualizer & Advanced Kinematics)
# Coordinate System: +X = Forward, +Y = Up, +Z = Right (Starboard)
# Origin (0,0,0) for Front: Center of the front axle on the ground.
# Origin (0,0,0) for Rear: Center of the rear axle on the ground.
# All units in meters [m].
# ════════════════════════════════════════════════════════════════════════

vehicle_params['hardpoints_f'] = {
    # Lower Wishbone (Inboard Front, Inboard Rear, Outboard/LBJ)
    'lca_in_f':   [ 0.180, 0.160, 0.230],  
    'lca_in_r':   [-0.180, 0.160, 0.230],  
    'lbj':        [ 0.000, 0.120, 0.560],  

    # Upper Wishbone (Inboard Front, Inboard Rear, Outboard/UBJ)
    'uca_in_f':   [ 0.140, 0.310, 0.183],  
    'uca_in_r':   [-0.140, 0.310, 0.183],  
    'ubj':        [ 0.000, 0.380, 0.540],  

    # Pushrod & Inboard Suspension 
    'pushrod_out': [ 0.000, 0.180, 0.550],  # Mount on upright/LCA
    'rocker_piv':  [ 0.000, 0.280, 0.340],  # Bellcrank center pivot
    'spring_in':   [ 0.150, 0.280, 0.155],  # Spring chassis mount
}

vehicle_params['hardpoints_r'] = {
    # Lower Wishbone
    'lca_in_f':   [ 0.200, 0.160, 0.220],  
    'lca_in_r':   [-0.200, 0.160, 0.220],  
    'lbj':        [ 0.000, 0.120, 0.550],  

    # Upper Wishbone
    'uca_in_f':   [ 0.150, 0.310, 0.170],  
    'uca_in_r':   [-0.150, 0.310, 0.170],  
    'ubj':        [ 0.000, 0.380, 0.530],  

    # Pushrod & Inboard Suspension
    'pushrod_out': [ 0.000, 0.180, 0.540],  
    'rocker_piv':  [ 0.000, 0.280, 0.320],  
    'spring_in':   [ 0.150, 0.280, 0.150],  
}