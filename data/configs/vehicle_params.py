# data/configs/vehicle_params.py
# Project-GP master vehicle parameter file
# Merged with Team Ter26 CAD specifics (300kg spec, 80kW power, empirical aero)

vehicle_params = {

    # ════════════════════════════════════════════════════════════════════════
    # MASS AND INERTIA
    # ════════════════════════════════════════════════════════════════════════
    'total_mass':          300.0,   # kg [Total Mass including driver]
    'm':                   300.0,   # kg [Legacy key alias]
    'mass_driver':          75.0,   # kg 
    'sprung_mass':         268.9,   # kg [DERIVED — 300 - 2*7.74 - 2*7.76]

    'Ix':   45.0,    # kg·m²  [Roll axis inertia]
    'Iy':   85.0,    # kg·m²  [Pitch axis inertia]
    'Iz':  150.0,    # kg·m²  [Yaw axis inertia - Assumed baseline]
    'Iw':    1.2,    # kg·m²  [Wheel rotational inertia]

    'lf':   0.8525,  # m [CG to Front Axle — 45% Front Weight]
    'lr':   0.6975,  # m [CG to Rear Axle — 55% Rear Weight]
    'h_cg': 0.330,   # m [Total CG Height]
    'h_cg_sprung': 0.350,   # m [Sprung mass CG]

    'track_front':  1.200,  # m [1200mm Fr]
    'track_rear':   1.180,  # m [1180mm Rr]
    'wheelbase':    1.550,  # m [lf + lr]

    'unsprung_mass_f': 7.74,   # kg [per front corner]
    'unsprung_mass_r': 7.76,   # kg [per rear corner]

    # ════════════════════════════════════════════════════════════════════════
    # SUSPENSION GEOMETRY
    # ════════════════════════════════════════════════════════════════════════
    # Roll centre heights
    'h_rc_f':  0.040,     # m 
    'h_rc_r':  0.06028,   # m 
    'dh_rc_dz_f': 0.20,   # m/m [Roll centre rise per m of bump travel]
    'dh_rc_dz_r': 0.30,

    # Motion ratio polynomial MR(z) = a0 + a1*z + a2*z^2
    'motion_ratio_f_poly': [1.14,  2.5, 0.0],
    'motion_ratio_r_poly': [1.16,  2.0, 0.0],

    # Static alignment at design ride height
    'static_camber_f':  -2.0,   # deg 
    'static_camber_r':  -1.5,   # deg 
    'static_toe_f':     -0.10,  # deg (neg = toe-in) 
    'static_toe_r':     -0.15,  # deg 

    # Camber gain [change in camber per degree of body roll]
    'camber_gain_f': -0.80,   # deg/deg 
    'camber_gain_r': -0.65,

    # Camber per metre of heave
    'camber_per_m_travel_f': -25.0,  # deg/m
    'camber_per_m_travel_r': -20.0,

    # Bump steer [change in toe per m of wheel travel]
    'bump_steer_f':  0.000,  # rad/m 
    'bump_steer_r':  0.000,

    # Compliance steer [lateral push test, deg/kN]
    'compliance_steer_f': -0.15,
    'compliance_steer_r': -0.10,

    # Ackermann
    'ackermann_factor': 0.0,   # 0=parallel, 1=full, -1=reverse 

    # Anti-pitch geometry
    'anti_squat':   0.30,
    'anti_lift':    0.20,
    'anti_dive_f':  0.40,
    'anti_dive_r':  0.10,

    # ════════════════════════════════════════════════════════════════════════
    # SPRING AND DAMPER RATES (Baselines for testing)
    # ════════════════════════════════════════════════════════════════════════
    'spring_rate_f': 35030.0,   # N/m at spring
    'spring_rate_r': 52540.0,
    'arb_rate_f':      200.0,   # N/m at wheel per side
    'arb_rate_r':      150.0,
    'lambda_stiffness': 2.0e-9, # Setup centering penalty weight

    'damper_c_low_f':  1800.0,   # N·s/m — low-speed region
    'damper_c_low_r':  1600.0,
    'damper_c_high_f':  720.0,   # N·s/m — high-speed region (digressive)
    'damper_c_high_r':  640.0,
    'damper_v_knee':     0.10,   # m/s — transition velocity

    'bump_stop_rate':  50000.0,   # N/m
    'bump_stop_engage': 0.025,    # m compression to engage
    'h_ride_design':    0.040,    # m 

    # ════════════════════════════════════════════════════════════════════════
    # AERODYNAMICS
    # ════════════════════════════════════════════════════════════════════════
    'Cl_ref':     4.14,     # downforce coefficient at h_aero_ref 
    'Cl':         4.14,     # [Legacy key alias]
    'Cd_ref':     2.50,     # drag coefficient 
    'Cd':         0.80,     # [Legacy key alias]
    'A_ref':      1.10,     # frontal area m² 
    'A':          1.10,     # [Legacy key alias]
    'h_aero_ref': 0.040,    # reference ride height for Cl_ref [m]
    
    'k_ground_f': 0.30,     # front ground effect sensitivity 
    'k_ground_r': 0.45,     # rear 
    
    'aero_split_f': 0.45,   # fraction of downforce on front axle 
    'aero_split_r': 0.55,
    'dCl_f_dtheta': 0.35,   # front Cl sensitivity to pitch angle (per rad)
    'rho_air':    1.225,    # kg/m³

    # ════════════════════════════════════════════════════════════════════════
    # DRIVETRAIN
    # ════════════════════════════════════════════════════════════════════════
    'motor_peak_torque':    450.0,    # N·m at wheel
    'motor_peak_power':   80000.0,    # W (80 kW)
    'motor_max_rpm':       6000.0,
    'final_drive_ratio':      4.5,
    'wheel_radius':          0.2032,  # m (8 inch rim to loaded radius)
    'drivetrain_efficiency':   0.92,
    'm_drivetrain_eff':       12.0,   # kg equivalent translational inertia
    'diff_lock_ratio':         1.0,   # 1.0=locked spool, 0.0=open, 0.3–0.7=LSD

    # ════════════════════════════════════════════════════════════════════════
    # BRAKES
    # ════════════════════════════════════════════════════════════════════════
    'brake_bias_f':        0.60,      # fraction of total brake force on front
    'ideal_brake_balance': False,
    'max_brake_torque':     800.0, 

    # ════════════════════════════════════════════════════════════════════════
    # TYRE ENVIRONMENT
    # ════════════════════════════════════════════════════════════════════════
    'T_env':          20.0,   # °C ambient temperature
    'P_nom':           1.2,   # bar cold inflation pressure
}

# Derived Parameters ensuring absolute mathematical consistency across modules
vehicle_params['wb'] = vehicle_params['lf'] + vehicle_params['lr']
vehicle_params['mass_dist'] = vehicle_params['lr'] / vehicle_params['wb']