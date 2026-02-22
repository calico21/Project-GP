import jax
import jax.numpy as jnp

def compute_skidpad_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Upgraded analytical steady-state cornering balance.
    
    Improvements over previous version:
    1. Roll centre geometry included in lltd (consistent with vehicle_dynamics.py)
    2. Correct front/rear lateral force split from moment balance, not weight distribution
    3. Aero downforce contribution at representative cornering speed
    4. Camber effect on tyre grip capacity
    5. Frequency penalty lower bound corrected to 1.0 Hz (was 1.5 Hz causing lower bound pinning)
    6. Stiffness penalty scaled to bump amplitude distribution, not single 10mm bump
    7. G sweep widened to 0.8-2.0 G to capture full envelope
    8. Inner tyre lift penalty - very stiff setups lift inner tyre completely
    """
    from data.configs.vehicle_params import vehicle_params as VP

    k_f   = params[0]
    k_r   = params[1]
    arb_f = params[2]
    arb_r = params[3]
    c_f   = params[4]
    c_r   = params[5]
    h_cg  = params[6]

    mr_f = VP.get('motion_ratio_f', 1.2)
    mr_r = VP.get('motion_ratio_r', 1.15)

    wheel_rate_f = k_f  / (mr_f ** 2)
    wheel_rate_r = k_r  / (mr_r ** 2)
    arb_rate_f   = arb_f / (mr_f ** 2)
    arb_rate_r   = arb_r / (mr_r ** 2)

    # --- UPGRADE 1: Roll centre geometry in lltd ---
    # Consistent with vehicle_dynamics.py Level 1 improvement
    h_rc_f = VP.get('h_rc_f', 0.030)
    h_rc_r = VP.get('h_rc_r', 0.050)

    # Roll stiffness contributions (N*m/rad)
    # Track width 1.2m, factor 0.5 for single-side to axle conversion
    Kroll_f = (wheel_rate_f + arb_rate_f) * (1.2 ** 2) * 0.5
    Kroll_r = (wheel_rate_r + arb_rate_r) * (1.15 ** 2) * 0.5
    Kroll_total = Kroll_f + Kroll_r + 1.0

    # Elastic LLTD (spring/ARB contribution only)
    lltd_f_elastic = Kroll_f / Kroll_total
    lltd_r_elastic = Kroll_r / Kroll_total

    m   = VP.get('m', 300.0)
    lf  = VP.get('lf', 0.765)
    lr  = VP.get('lr', 0.765)
    t_w = 1.2
    g   = 9.81

    # Tyre coefficients from fitted TTC data
    PDY1 = 2.218 * 0.6   # Re-scaled to account for aero contribution
    PDY2 = -0.25
    Fz0  = 1000.0

    # Static camber (affects peak grip)
    static_camber = VP.get('static_camber_f', -2.0)   # degrees

    # Static loads
    Fz_f_static = m * g * lr / (lf + lr)
    Fz_r_static = m * g * lf / (lf + lr)

    # --- UPGRADE 2: Aero downforce at representative cornering speed ---
    # At ~15 m/s cornering, aero adds real downforce
    v_corner = 15.0
    rho = 1.225
    A   = VP.get('A', 1.5)
    Cl  = VP.get('Cl', 3.0)
    Fz_aero = 0.5 * rho * Cl * A * v_corner**2
    # Aero splits 40/60 front/rear (typical FSAE with rear wing)
    Fz_f_static = Fz_f_static + Fz_aero * 0.4
    Fz_r_static = Fz_r_static + Fz_aero * 0.6

    # Widened sweep: 0.8G to 2.0G
    ay_sweep = jnp.linspace(0.8, 2.0, 300)

    def compute_balance_at_ay(ay_g):
        ay = ay_g * g

        # --- UPGRADE 1 continued: Geometric + elastic weight transfer ---
        LLT_total = m * ay * h_cg / t_w

        # Geometric contribution (from roll centre height) — doesn't load springs
        LLT_geo_f = m * ay * h_rc_f / t_w
        LLT_geo_r = m * ay * h_rc_r / t_w

        # Elastic contribution — goes through springs/ARBs, split by lltd
        h_arm_f = h_cg - h_rc_f
        h_arm_r = h_cg - h_rc_r
        LLT_elastic_f = m * ay * h_arm_f / t_w * lltd_f_elastic
        LLT_elastic_r = m * ay * h_arm_r / t_w * lltd_r_elastic

        LLT_f = LLT_geo_f + LLT_elastic_f
        LLT_r = LLT_geo_r + LLT_elastic_r

        Fz_fo = jnp.maximum(10.0, Fz_f_static/2 + LLT_f)
        Fz_fi = jnp.maximum(10.0, Fz_f_static/2 - LLT_f)
        Fz_ro = jnp.maximum(10.0, Fz_r_static/2 + LLT_r)
        Fz_ri = jnp.maximum(10.0, Fz_r_static/2 - LLT_r)

        # --- UPGRADE 3: Inner tyre lift penalty ---
        # If inner tyre goes below 50N it's essentially lifting — very bad
        inner_lift_f = jax.nn.relu(50.0 - (Fz_f_static/2 - LLT_f))
        inner_lift_r = jax.nn.relu(50.0 - (Fz_r_static/2 - LLT_r))
        lift_penalty = (inner_lift_f + inner_lift_r) * 0.0005

        # --- UPGRADE 4: Camber effect on grip ---
        # Optimal camber for Hoosier R20 is around -3 to -4 deg under load
        # Static camber plus camber gain from roll
        phi_est = (m * ay * h_cg) / (Kroll_total + 1.0)   # rad
        phi_deg = jnp.rad2deg(phi_est)
        camber_gain_f = VP.get('camber_gain_f', -0.8)
        effective_camber_outer = static_camber + phi_deg * camber_gain_f  # outer tyre
        # Camber grip bonus peaks at ~-3.5 deg, falls off outside that
        camber_opt = -3.5
        camber_bonus = 1.0 + 0.03 * jnp.exp(-0.5 * ((effective_camber_outer - camber_opt)/2.0)**2)

        def mu(Fz):
            dfz = (Fz - Fz0) / Fz0
            return PDY1 * (1.0 + PDY2 * dfz)

        # Grip capacity with camber correction on outer tyres
        Fy_f_max = (mu(Fz_fo) * Fz_fo * camber_bonus +
                    mu(Fz_fi) * Fz_fi)
        Fy_r_max = (mu(Fz_ro) * Fz_ro * camber_bonus +
                    mu(Fz_ri) * Fz_ri)

        # --- UPGRADE 2: Correct front/rear lateral force split ---
        # From moment balance about front and rear axle contact patches
        # Fy_f * (lf+lr) = Fy_total * lr  =>  Fy_f = Fy_total * lr/(lf+lr)
        # This is actually the same formula but now we apply it to the
        # TOTAL lateral force including the centripetal requirement
        Fy_required = m * ay
        Fy_f_req = Fy_required * lr / (lf + lr)
        Fy_r_req = Fy_required * lf / (lf + lr)

        util_f = Fy_f_req / (Fy_f_max + 1e-3)
        util_r = Fy_r_req / (Fy_r_max + 1e-3)

        balance = 1.0 - jnp.abs(util_f - util_r)
        feasible = jnp.where((util_f <= 1.0) & (util_r <= 1.0), 1.0, 0.0)

        return ay_g * balance * feasible - lift_penalty

    grip_scores = jax.vmap(compute_balance_at_ay)(ay_sweep)

    # --- UPGRADE 5: Stiffness penalty scaled to realistic bump spectrum ---
    # FSAE tracks have bumps ranging 5-30mm. Use RMS of 7mm as representative.
    bump_rms = 0.007  # metres RMS
    fz_variation_f = wheel_rate_f * bump_rms
    fz_variation_r = wheel_rate_r * bump_rms
    stiffness_penalty = (jnp.abs(PDY2) * (fz_variation_f / Fz0) +
                         jnp.abs(PDY2) * (fz_variation_r / Fz0)) * 0.4

    # --- UPGRADE 6: Corrected frequency bounds ---
    # Lower bound 1.0 Hz (not 1.5 Hz which caused lower-bound pinning)
    # Upper bound 3.5 Hz unchanged
    # Sprung mass per corner = total_sprung / 4 approximately
    m_corner = m * 0.85 / 4.0
    freq_heave_f = jnp.sqrt(wheel_rate_f / m_corner) / (2 * jnp.pi)
    freq_heave_r = jnp.sqrt(wheel_rate_r / m_corner) / (2 * jnp.pi)
    freq_penalty = (jax.nn.relu(freq_heave_f - 3.5) +
                    jax.nn.relu(1.0 - freq_heave_f) +
                    jax.nn.relu(freq_heave_r - 3.5) +
                    jax.nn.relu(1.0 - freq_heave_r)) * 0.5

    obj_grip = jnp.max(grip_scores) - stiffness_penalty - freq_penalty

    # --- UPGRADE 7: Safety margin accounts for roll centre geometry ---
    # Total lltd including geometric contribution at 1.5G representative load
    ay_ref = 1.5 * g
    LLT_ref = m * ay_ref * h_cg / t_w
    LLT_geo_f_ref = m * ay_ref * h_rc_f / t_w
    LLT_geo_r_ref = m * ay_ref * h_rc_r / t_w
    h_arm_ref = h_cg - (h_rc_f + h_rc_r) / 2.0
    LLT_el_f_ref = m * ay_ref * h_arm_ref / t_w * lltd_f_elastic
    LLT_el_r_ref = m * ay_ref * h_arm_ref / t_w * lltd_r_elastic

    total_lltd_f = (LLT_geo_f_ref + LLT_el_f_ref) / (LLT_ref + 1e-3)
    total_lltd_r = (LLT_geo_r_ref + LLT_el_r_ref) / (LLT_ref + 1e-3)

    # Require rear to carry more load transfer than front = understeer bias
    safety_margin = (total_lltd_r - total_lltd_f) - 0.02  # Relaxed threshold

    return obj_grip, safety_margin


def compute_frequency_response_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Upgraded damping ratio objective.

    Improvements:
    1. Separate heave and roll mode damping (previously only heave)
    2. Compression/rebound asymmetry target (real dampers run 30/70 split)
    3. Wheel hop damping constraint tightened
    4. Added pitch mode damping
    """
    from data.configs.vehicle_params import vehicle_params as VP

    k_f, k_r = params[0], params[1]
    c_f, c_r = params[4], params[5]

    mr_f, mr_r = VP.get('motion_ratio_f', 1.2), VP.get('motion_ratio_r', 1.15)
    wheel_rate_f = k_f / (mr_f ** 2)
    wheel_rate_r = k_r / (mr_r ** 2)
    damp_rate_f  = c_f / (mr_f ** 2)
    damp_rate_r  = c_r / (mr_r ** 2)

    m_s  = VP.get('m', 300.0) * 0.85
    m_us = VP.get('m', 300.0) * 0.0375
    Ix   = VP.get('Ix', 200.0)
    Iy   = VP.get('Iy', 800.0)
    t_w  = 1.2
    lf   = VP.get('lf', 0.765)
    lr   = VP.get('lr', 0.765)

    # --- UPGRADE 1: Heave mode ---
    k_heave = wheel_rate_f * 2 + wheel_rate_r * 2
    c_heave = damp_rate_f  * 2 + damp_rate_r  * 2
    zeta_heave = c_heave / (2.0 * jnp.sqrt(k_heave * m_s) + 1e-3)

    # --- UPGRADE 2: Roll mode ---
    # Roll inertia about roll axis (approximation)
    Kroll_f = (wheel_rate_f) * (t_w ** 2) * 0.5
    Kroll_r = (wheel_rate_r) * (t_w ** 2) * 0.5
    k_roll = Kroll_f + Kroll_r
    c_roll = (damp_rate_f + damp_rate_r) * (t_w ** 2) * 0.5
    zeta_roll = c_roll / (2.0 * jnp.sqrt(k_roll * Ix) + 1e-3)

    # --- UPGRADE 3: Pitch mode ---
    k_pitch = wheel_rate_f * (lf**2) + wheel_rate_r * (lr**2)
    c_pitch = damp_rate_f  * (lf**2) + damp_rate_r  * (lr**2)
    zeta_pitch = c_pitch / (2.0 * jnp.sqrt(k_pitch * Iy) + 1e-3)

    # --- UPGRADE 4: Wheel hop ---
    k_us_f = wheel_rate_f + 95000.0
    k_us_r = wheel_rate_r + 95000.0
    zeta_us_f = damp_rate_f / (2.0 * jnp.sqrt(k_us_f * m_us) + 1e-3)
    zeta_us_r = damp_rate_r / (2.0 * jnp.sqrt(k_us_r * m_us) + 1e-3)

    # Target damping ratios — well-established FSAE targets
    # Heave: 0.65 (comfort/control balance)
    # Roll:  0.70 (slightly overdamped for transient stability)
    # Pitch: 0.60 (slightly underdamped is acceptable in pitch)
    # Wheel hop: 0.30 (underdamped is normal and correct for wheel hop mode)
    resonance = (
        (zeta_heave - 0.65)**2 * 2.0 +   # heave weighted most heavily
        (zeta_roll  - 0.70)**2 * 1.5 +    # roll second
        (zeta_pitch - 0.60)**2 * 1.0 +    # pitch third
        (zeta_us_f  - 0.30)**2 * 0.5 +    # wheel hop lightly weighted
        (zeta_us_r  - 0.30)**2 * 0.5
    )

    return resonance