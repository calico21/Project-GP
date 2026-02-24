import jax
import jax.numpy as jnp

def compute_skidpad_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Upgraded analytical steady-state cornering balance.
    
    Improvements over previous version:
    1. Roll centre geometry included in lltd (consistent with vehicle_dynamics.py)
    2. Correct front/rear lateral force split from moment balance
    3. Aero downforce contribution at representative cornering speed
    4. Camber effect on tyre grip capacity
    5. Frequency penalty lower bound fixed to 1.20 Hz to stop lower-bound pinning
    6. Stiffness penalty anchored to realistic FSAE spring rates
    7. G sweep widened to 0.8-2.0 G to capture full envelope
    8. Brake balance added to search space and optimized for 1G braking
    9. Safety constraint tightened for autocross
    """
    from data.configs.vehicle_params import vehicle_params as VP

    k_f   = params[0]
    k_r   = params[1]
    arb_f = params[2]
    arb_r = params[3]
    c_f   = params[4]
    c_r   = params[5]
    h_cg  = params[6]
    
    # --- UPGRADE: Brake bias added to search space (Part 7.2) ---
    brake_bias_f = params[7] 

    mr_f = jnp.array(VP.get('motion_ratio_f_poly', [1.20]))[0]
    mr_r = jnp.array(VP.get('motion_ratio_r_poly', [1.15]))[0]

    wheel_rate_f = k_f  / (mr_f ** 2)
    wheel_rate_r = k_r  / (mr_r ** 2)
    arb_rate_f   = arb_f / (mr_f ** 2)
    arb_rate_r   = arb_r / (mr_r ** 2)

    h_rc_f = VP.get('h_rc_f', 0.030)
    h_rc_r = VP.get('h_rc_r', 0.050)

    # Roll stiffness contributions (N*m/rad)
    t_w = VP.get('track_front', 1.20)
    Kroll_f = (wheel_rate_f + arb_rate_f) * (t_w ** 2) * 0.5
    Kroll_r = (wheel_rate_r + arb_rate_r) * (VP.get('track_rear', 1.18) ** 2) * 0.5
    Kroll_total = Kroll_f + Kroll_r + 1.0

    # Elastic LLTD (spring/ARB contribution only)
    lltd_f_elastic = Kroll_f / Kroll_total
    lltd_r_elastic = Kroll_r / Kroll_total

    m  = VP.get('total_mass', VP.get('m', 230.0))
    lf = VP.get('lf', 0.680)
    lr = VP.get('lr', 0.920)
    L  = lf + lr
    g  = 9.81

    # Tyre coefficients from MF6.2
    PDY1 = 2.218 * 0.6   # Re-scaled to account for aero contribution
    PDY2 = -0.25
    Fz0  = 1000.0

    static_camber = VP.get('static_camber_f', -1.5)

    Fz_f_static = m * g * lr / L
    Fz_r_static = m * g * lf / L

    # Aero downforce at representative cornering speed
    v_corner = 15.0
    rho = VP.get('rho_air', 1.225)
    A   = VP.get('A_ref', 1.1)
    Cl  = VP.get('Cl_ref', 3.0)
    Fz_aero = 0.5 * rho * Cl * A * v_corner**2
    
    aero_split_f = VP.get('aero_split_f', 0.40)
    aero_split_r = VP.get('aero_split_r', 0.60)
    Fz_f_static += Fz_aero * aero_split_f
    Fz_r_static += Fz_aero * aero_split_r

    ay_sweep = jnp.linspace(0.8, 2.0, 300)

    def compute_balance_at_ay(ay_g):
        ay = ay_g * g

        LLT_total = m * ay * h_cg / t_w

        LLT_geo_f = m * ay * h_rc_f / t_w
        LLT_geo_r = m * ay * h_rc_r / VP.get('track_rear', 1.18)

        h_arm_f = h_cg - h_rc_f
        h_arm_r = h_cg - h_rc_r
        LLT_elastic_f = m * ay * h_arm_f / t_w * lltd_f_elastic
        LLT_elastic_r = m * ay * h_arm_r / VP.get('track_rear', 1.18) * lltd_r_elastic

        LLT_f = LLT_geo_f + LLT_elastic_f
        LLT_r = LLT_geo_r + LLT_elastic_r

        Fz_fo = jnp.maximum(10.0, Fz_f_static/2 + LLT_f)
        Fz_fi = jnp.maximum(10.0, Fz_f_static/2 - LLT_f)
        Fz_ro = jnp.maximum(10.0, Fz_r_static/2 + LLT_r)
        Fz_ri = jnp.maximum(10.0, Fz_r_static/2 - LLT_r)

        inner_lift_f = jax.nn.relu(50.0 - (Fz_f_static/2 - LLT_f))
        inner_lift_r = jax.nn.relu(50.0 - (Fz_r_static/2 - LLT_r))
        lift_penalty = (inner_lift_f + inner_lift_r) * 0.0005

        phi_est = (m * ay * h_cg) / (Kroll_total + 1.0)
        phi_deg = jnp.rad2deg(phi_est)
        camber_gain_f = VP.get('camber_gain_f', -0.8)
        effective_camber_outer = static_camber + phi_deg * camber_gain_f
        camber_opt = -3.5
        camber_bonus = 1.0 + 0.03 * jnp.exp(-0.5 * ((effective_camber_outer - camber_opt)/2.0)**2)

        def mu(Fz):
            dfz = (Fz - Fz0) / Fz0
            return PDY1 * (1.0 + PDY2 * dfz)

        Fy_f_max = (mu(Fz_fo) * Fz_fo * camber_bonus + mu(Fz_fi) * Fz_fi)
        Fy_r_max = (mu(Fz_ro) * Fz_ro * camber_bonus + mu(Fz_ri) * Fz_ri)

        Fy_required = m * ay
        Fy_f_req = Fy_required * lr / L
        Fy_r_req = Fy_required * lf / L

        util_f = Fy_f_req / (Fy_f_max + 1e-3)
        util_r = Fy_r_req / (Fy_r_max + 1e-3)

        balance = 1.0 - jnp.abs(util_f - util_r)
        feasible = jnp.where((util_f <= 1.0) & (util_r <= 1.0), 1.0, 0.0)

        return ay_g * balance * feasible - lift_penalty

    grip_scores = jax.vmap(compute_balance_at_ay)(ay_sweep)

    # --- UPGRADE 7.1: Fix Spring Rate Lower-Bound Pinning ---
    bump_rms = 0.007  
    fz_variation_f = wheel_rate_f * bump_rms
    fz_variation_r = wheel_rate_r * bump_rms
    
    # Combine vibration penalty with a quadratic centering to pull springs up to 25k
    lambda_stiff = VP.get('lambda_stiffness', 2.0e-9)
    stiffness_penalty = (jnp.abs(PDY2) * (fz_variation_f / Fz0) + 
                         jnp.abs(PDY2) * (fz_variation_r / Fz0)) * 0.4
    stiffness_penalty += lambda_stiff * ((k_f - 25000.0)**2 + (k_r - 25000.0)**2)

    m_s_total = VP.get('sprung_mass', m * 0.85)
    m_corner = m_s_total / 4.0
    freq_heave_f = jnp.sqrt(wheel_rate_f / m_corner) / (2 * jnp.pi)
    freq_heave_r = jnp.sqrt(wheel_rate_r / m_corner) / (2 * jnp.pi)
    
    freq_lower_bound = 1.20   # Raised from 1.0 to 1.2 to resolve lower-bound pinning
    freq_penalty_f   = jax.nn.relu(freq_lower_bound - freq_heave_f) ** 2 * 500.0
    freq_penalty_r   = jax.nn.relu(freq_lower_bound - freq_heave_r) ** 2 * 500.0
    freq_penalty_high = (jax.nn.relu(freq_heave_f - 3.5)**2 + jax.nn.relu(freq_heave_r - 3.5)**2) * 500.0
    freq_penalty = freq_penalty_f + freq_penalty_r + freq_penalty_high

    # --- UPGRADE 7.2: Brake Balance Penalty ---
    # Ideal bias = Fz_f / (Fz_f + Fz_r) at 1G braking
    Fz_f_brake = (m * g * lr / L) + (m * 1.0 * g * h_cg / L)
    Fz_r_brake = (m * g * lf / L) - (m * 1.0 * g * h_cg / L)
    ideal_bias = Fz_f_brake / (Fz_f_brake + Fz_r_brake)
    brake_balance_penalty = 200.0 * (brake_bias_f - ideal_bias) ** 2

    # Final Grip Objective Assembly
    obj_grip = jnp.max(grip_scores) - stiffness_penalty - freq_penalty - brake_balance_penalty

    # --- UPGRADE 7.3: Tighten Safety Constraint ---
    ay_ref = 1.5 * g
    LLT_ref = m * ay_ref * h_cg / t_w
    LLT_geo_f_ref = m * ay_ref * h_rc_f / t_w
    LLT_geo_r_ref = m * ay_ref * h_rc_r / VP.get('track_rear', 1.18)
    h_arm_ref = h_cg - (h_rc_f + h_rc_r) / 2.0
    LLT_el_f_ref = m * ay_ref * h_arm_ref / t_w * lltd_f_elastic
    LLT_el_r_ref = m * ay_ref * h_arm_ref / VP.get('track_rear', 1.18) * lltd_r_elastic

    total_lltd_f = (LLT_geo_f_ref + LLT_el_f_ref) / (LLT_ref + 1e-3)
    total_lltd_r = (LLT_geo_r_ref + LLT_el_r_ref) / (LLT_ref + 1e-3)

    # Tightened threshold to 0.05 for FSAE autocross conditions
    safety_margin = (total_lltd_r - total_lltd_f) - 0.05 

    return obj_grip, safety_margin


def compute_frequency_response_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Upgraded damping ratio objective.
    """
    from data.configs.vehicle_params import vehicle_params as VP

    k_f, k_r = params[0], params[1]
    c_f, c_r = params[4], params[5]

    mr_f = jnp.array(VP.get('motion_ratio_f_poly', [1.20]))[0]
    mr_r = jnp.array(VP.get('motion_ratio_r_poly', [1.15]))[0]
    
    wheel_rate_f = k_f / (mr_f ** 2)
    wheel_rate_r = k_r / (mr_r ** 2)
    damp_rate_f  = c_f / (mr_f ** 2)
    damp_rate_r  = c_r / (mr_r ** 2)

    m = VP.get('total_mass', VP.get('m', 230.0))
    m_s  = VP.get('sprung_mass', m * 0.85)
    m_us_f = VP.get('unsprung_mass_f', 10.0)
    m_us_r = VP.get('unsprung_mass_r', 11.0)
    
    Ix   = VP.get('Ix', 45.0)
    Iy   = VP.get('Iy', 85.0)
    t_w  = VP.get('track_front', 1.20)
    lf   = VP.get('lf', 0.680)
    lr   = VP.get('lr', 0.920)

    k_heave = wheel_rate_f * 2 + wheel_rate_r * 2
    c_heave = damp_rate_f  * 2 + damp_rate_r  * 2
    zeta_heave = c_heave / (2.0 * jnp.sqrt(k_heave * m_s) + 1e-3)

    Kroll_f = (wheel_rate_f) * (t_w ** 2) * 0.5
    Kroll_r = (wheel_rate_r) * (t_w ** 2) * 0.5
    k_roll = Kroll_f + Kroll_r
    c_roll = (damp_rate_f + damp_rate_r) * (t_w ** 2) * 0.5
    zeta_roll = c_roll / (2.0 * jnp.sqrt(k_roll * Ix) + 1e-3)

    k_pitch = wheel_rate_f * (lf**2) + wheel_rate_r * (lr**2)
    c_pitch = damp_rate_f  * (lf**2) + damp_rate_r  * (lr**2)
    zeta_pitch = c_pitch / (2.0 * jnp.sqrt(k_pitch * Iy) + 1e-3)

    k_us_f = wheel_rate_f + 50000.0
    k_us_r = wheel_rate_r + 50000.0
    zeta_us_f = damp_rate_f / (2.0 * jnp.sqrt(k_us_f * m_us_f) + 1e-3)
    zeta_us_r = damp_rate_r / (2.0 * jnp.sqrt(k_us_r * m_us_r) + 1e-3)

    resonance = (
        (zeta_heave - 0.65)**2 * 2.0 +   # heave weighted most heavily
        (zeta_roll  - 0.70)**2 * 1.5 +    # roll second
        (zeta_pitch - 0.60)**2 * 1.0 +    # pitch third
        (zeta_us_f  - 0.30)**2 * 0.5 +    # wheel hop lightly weighted
        (zeta_us_r  - 0.30)**2 * 0.5
    )

    return resonance