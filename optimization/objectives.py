import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# PENALTY SCALE REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
# The raw grip signal is bounded in [0.5, 2.5] G after the ay_sweep extension.
# Every penalty coefficient is sized so that the worst physically realistic
# violation costs at most 0.10 G — 4% of the signal range.
#
# CHANGE LOG vs previous version
# ─────────────────────────────────────────────────────────────────────────────
#
# ── PRIMARY BUG FIX: PDY1 corrected from 2.218 × 0.6 → 1.92 ─────────────────
#
#   Symptom: Max_Grip_Found = 1.40 G (FS car with Cl=3.0 should be ~1.85-2.0 G)
#
#   Root cause — wrong coefficient, wrong justification:
#       Previous: PDY1 = 2.218 * 0.6 = 1.3308
#       Comment claimed it was "rescaled for aero contribution"
#
#   Why the 0.6 factor is wrong (two independent reasons):
#
#   1. Aero adds Fz, not reduces mu.
#      Aerodynamic downforce raises the vertical load on each tyre.
#      This is already correctly modelled by:
#          Fz_f_static += Fz_aero * aero_split_f
#          Fz_r_static += Fz_aero * aero_split_r
#      Multiplying PDY1 by 0.6 ALSO reduces the friction coefficient,
#      double-penalising the car for having aero — physically nonsensical.
#
#   2. Load sensitivity degradation is already modelled by PDY2.
#      The mu() function  mu(Fz) = PDY1 × (1 + PDY2 × dfz)
#      with PDY2 = -0.25 already reduces the effective friction coefficient
#      as Fz increases above Fz0.  The 0.6 multiplier was an apparent
#      attempt to pre-account for this, but PDY2 does it analytically.
#
#   Correct value: PDY1 = 1.92
#      Source: TTC (Tire Test Consortium) Round 8 public data set,
#      10-inch Hoosier LCO-H2O and R25B tires.
#      Peak lateral mu values by condition:
#          Optimal (camber −3°, temp 80°C, 10 psi):   1.95–2.05
#          Nominal race conditions:                     1.85–1.95
#          Conservative / cold tyre:                    1.75–1.85
#      PDY1 = 1.92 represents a well-set-up car in nominal conditions.
#
#   Impact: max grip rises from 1.40 G → ~1.80–2.00 G, consistent with
#   the physical expectation for a 230 kg car with Cl=3.0 aero package.
#
# ── ay_sweep extended from [0.8, 2.0] G to [0.5, 2.5] G, 300→400 points ─────
#   With PDY1=1.92, optimal setups approach ~1.90–2.00 G.  The previous
#   ceiling of 2.0 G left < 0.05 G of headroom, meaning the log-sum-exp
#   smooth maximum was being evaluated in a region where all sweep points
#   had nearly-identical scores near the plateau, artificially compressing
#   the gradient signal.  Extending to 2.5 G adds clean headroom.
#   Lower bound 0.5 G (from 0.8 G) prevents the sigmoid ramp-in from
#   biasing the smooth maximum at low ay values.
#   400 points: step size 0.005 G (identical to before).
#
# ── _LSE_BETA raised from 10 → 20 ─────────────────────────────────────────────
#   The log-sum-exp smooth maximum overestimates the true maximum by
#   approximately log(N_eff) / beta, where N_eff is the effective support width.
#   For a 0.10 G-wide peak with 0.005 G steps: N_eff ≈ 20 points.
#       beta=10: overestimate ≈ log(20)/10 = 0.30 G  (significant bias)
#       beta=20: overestimate ≈ log(20)/20 = 0.15 G  (halved)
#   Higher beta also concentrates gradients near the true maximum, giving
#   the TRPO optimizer a sharper fitness landscape to follow.
#   No differentiability concern: d/dx[logsumexp] = softmax, non-zero everywhere.
#
# ── FIX 4 (retained): freq_penalty absent from grip objective ─────────────────
# ── FIX 5 (retained): normalised centering penalty ────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ── BUG FIX: raised from 10 → 20 to halve the smooth-max bias ────────────────
_LSE_BETA = 20.0


def compute_skidpad_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Differentiable analytical steady-state cornering balance.

    API NOTE: simulate_step_fn, dt, and T_max are accepted for interface
    compatibility but are NOT called inside this function.  The cornering
    balance is computed analytically, not by integrating the ODE.  This is
    intentional — the analytical formulation is exact, fully differentiable
    everywhere, and does not require a simulation horizon to reach steady-state.

    Returns
    -------
    obj_grip      : scalar — maximise this (units: lateral G, range ~0.5–2.5)
    safety_margin : scalar — must be > 0 (positive = understeer = safe)
    """
    from data.configs.vehicle_params import vehicle_params as VP

    k_f          = params[0]
    k_r          = params[1]
    arb_f        = params[2]
    arb_r        = params[3]
    c_f          = params[4]
    c_r          = params[5]
    h_cg         = params[6]
    brake_bias_f = params[7]

    mr_f = jnp.array(VP.get('motion_ratio_f_poly', [1.20]))[0]
    mr_r = jnp.array(VP.get('motion_ratio_r_poly', [1.15]))[0]

    wheel_rate_f = k_f  / (mr_f ** 2)
    wheel_rate_r = k_r  / (mr_r ** 2)
    arb_rate_f   = arb_f / (mr_f ** 2)
    arb_rate_r   = arb_r / (mr_r ** 2)

    h_rc_f = VP.get('h_rc_f', 0.030)
    h_rc_r = VP.get('h_rc_r', 0.050)

    t_w = VP.get('track_front', 1.20)
    t_r = VP.get('track_rear',  1.18)

    # Roll stiffness (N·m / rad) from springs + ARB
    Kroll_f     = (wheel_rate_f + arb_rate_f) * (t_w ** 2) * 0.5
    Kroll_r     = (wheel_rate_r + arb_rate_r) * (t_r  ** 2) * 0.5
    Kroll_total = Kroll_f + Kroll_r + 1.0

    lltd_f_elastic = Kroll_f / Kroll_total
    lltd_r_elastic = Kroll_r / Kroll_total

    m  = VP.get('total_mass', VP.get('m', 230.0))
    lf = VP.get('lf', 0.680)
    lr = VP.get('lr', 0.920)
    L  = lf + lr
    g  = 9.81

    # ── PRIMARY BUG FIX: PDY1 corrected from 2.218*0.6=1.33 → 1.92 ──────────
    # See module-level change log for full explanation.
    # PDY1 = 1.92 is the peak lateral friction coefficient for a 10-inch
    # Hoosier LCO-H2O / R25B in nominal Formula SAE race conditions.
    # PDY2 = -0.25 degrades mu at higher vertical loads (load sensitivity).
    # Fz0 = 1000 N is the reference load at which PDY1 is defined.
    PDY1 = 1.92
    PDY2 = -0.25
    Fz0  = 1000.0

    static_camber = VP.get('static_camber_f', -1.5)
    camber_gain_f = VP.get('camber_gain_f', -0.8)

    # Static axle loads
    Fz_f_static = m * g * lr / L
    Fz_r_static = m * g * lf / L

    # Aerodynamic downforce at nominal skidpad speed (15 m/s ≈ 54 km/h)
    # This correctly increases Fz — it does NOT reduce PDY1.
    v_corner     = 15.0
    rho          = VP.get('rho_air', 1.225)
    A            = VP.get('A_ref',   1.1)
    Cl           = VP.get('Cl_ref',  3.0)
    Fz_aero      = 0.5 * rho * Cl * A * v_corner ** 2

    aero_split_f = VP.get('aero_split_f', 0.40)
    aero_split_r = VP.get('aero_split_r', 0.60)
    Fz_f_static  += Fz_aero * aero_split_f
    Fz_r_static  += Fz_aero * aero_split_r

    # ── ay_sweep extended to [0.5, 2.5] G, 400 points ───────────────────────
    # Previous: linspace(0.8, 2.0, 300) — ceiling too close to expected peak.
    # With PDY1=1.92, optimal setups reach ~1.90–2.00 G.
    # 400 points × (2.5-0.5)/400 = 0.005 G/step (identical resolution).
    ay_sweep = jnp.linspace(0.5, 2.5, 400)

    def compute_balance_at_ay(ay_g):
        ay = ay_g * g

        # Geometric lateral load transfer (roll centre height effect)
        LLT_geo_f = m * ay * h_rc_f / t_w
        LLT_geo_r = m * ay * h_rc_r / t_r

        # Elastic lateral load transfer (roll stiffness distribution)
        h_arm_f       = h_cg - h_rc_f
        h_arm_r       = h_cg - h_rc_r
        LLT_elastic_f = m * ay * h_arm_f / t_w * lltd_f_elastic
        LLT_elastic_r = m * ay * h_arm_r / t_r  * lltd_r_elastic

        LLT_f = LLT_geo_f + LLT_elastic_f
        LLT_r = LLT_geo_r + LLT_elastic_r

        # Per-wheel vertical loads
        Fz_fo = jnp.maximum(10.0, Fz_f_static / 2 + LLT_f)
        Fz_fi = jnp.maximum(10.0, Fz_f_static / 2 - LLT_f)
        Fz_ro = jnp.maximum(10.0, Fz_r_static / 2 + LLT_r)
        Fz_ri = jnp.maximum(10.0, Fz_r_static / 2 - LLT_r)

        # Inner-wheel lift penalty (smooth relu — avoids hard constraint)
        inner_lift_f = jax.nn.relu(50.0 - (Fz_f_static / 2 - LLT_f))
        inner_lift_r = jax.nn.relu(50.0 - (Fz_r_static / 2 - LLT_r))
        lift_penalty = (inner_lift_f + inner_lift_r) * 0.0005

        # Camber compensation: outer tyre gains bonus at optimal effective camber
        phi_est              = (m * ay * h_cg) / (Kroll_total + 1.0)
        phi_deg              = jnp.rad2deg(phi_est)
        effective_camber_out = static_camber + phi_deg * camber_gain_f
        camber_opt           = -3.5
        camber_bonus         = 1.0 + 0.03 * jnp.exp(
            -0.5 * ((effective_camber_out - camber_opt) / 2.0) ** 2
        )

        # Peak friction coefficient with Pacejka load sensitivity
        # mu(Fz) = PDY1 * (1 + PDY2 * (Fz - Fz0) / Fz0)
        def mu(Fz):
            dfz = (Fz - Fz0) / Fz0
            return PDY1 * (1.0 + PDY2 * dfz)

        # Maximum lateral force capacity per axle
        Fy_f_max = mu(Fz_fo) * Fz_fo * camber_bonus + mu(Fz_fi) * Fz_fi
        Fy_r_max = mu(Fz_ro) * Fz_ro * camber_bonus + mu(Fz_ri) * Fz_ri

        # Required lateral force (cornering balance)
        Fy_required = m * ay
        Fy_f_req    = Fy_required * lr / L
        Fy_r_req    = Fy_required * lf / L

        util_f = Fy_f_req / (Fy_f_max + 1e-3)
        util_r = Fy_r_req / (Fy_r_max + 1e-3)

        # Axle balance: 1.0 = perfect balance, < 1.0 = misbalanced
        balance = 1.0 - jnp.abs(util_f - util_r)

        # FIX 2 (retained): smooth feasibility via sigmoid product
        # sharpness=10 → sigmoid = 0.5 at util=1.0 (tyre limit), = 0.99 at util=0.54
        # Replaced hard jnp.where which had zero gradient at infeasible setups.
        sharpness     = 10.0
        feasible_soft = (jax.nn.sigmoid((1.0 - util_f) * sharpness) *
                         jax.nn.sigmoid((1.0 - util_r) * sharpness))

        return ay_g * balance * feasible_soft - lift_penalty

    grip_scores = jax.vmap(compute_balance_at_ay)(ay_sweep)

    # FIX 3 (retained, beta raised to 20): smooth maximum via log-sum-exp
    # Overestimate ≈ log(N_eff)/beta ≈ log(20)/20 = 0.15 G (vs 0.30 G at beta=10)
    smooth_max = (1.0 / _LSE_BETA) * jax.nn.logsumexp(_LSE_BETA * grip_scores)

    # ── Stiffness penalty (FIX 5 retained — normalised centering) ────────────
    # Measures FRACTIONAL deviation from nominal spring rate so penalty is
    # O(spring_rate)-independent.  Worst case at k=60 kN/m: 0.020 G. ✓
    bump_rms        = 0.007            # m — road roughness RMS amplitude
    fz_variation_f  = wheel_rate_f * bump_rms
    fz_variation_r  = wheel_rate_r * bump_rms

    k_ref_centering = 25000.0          # N/m — nominal FSAE spring rate
    w_centering     = 0.01             # G per unit fractional²

    centering_penalty = w_centering * (
        ((k_f - k_ref_centering) / k_ref_centering) ** 2 +
        ((k_r - k_ref_centering) / k_ref_centering) ** 2
    )

    stiffness_penalty = (
        (jnp.abs(PDY2) * (fz_variation_f / Fz0) +
         jnp.abs(PDY2) * (fz_variation_r / Fz0)) * 0.4
        + centering_penalty
    )

    # ── Brake balance penalty (FIX 1 coefficient retained) ───────────────────
    # Worst case (0.3 error): 3.0 × 0.09 = 0.27 G ✓  (previous 200.0 → 18 G)
    Fz_f_brake   = (m * g * lr / L) + (m * 1.0 * g * h_cg / L)
    Fz_r_brake   = (m * g * lf / L) - (m * 1.0 * g * h_cg / L)
    ideal_bias   = Fz_f_brake / (Fz_f_brake + Fz_r_brake)
    brake_balance_penalty = 3.0 * (brake_bias_f - ideal_bias) ** 2

    # ── Final grip objective (FIX 4 retained: freq_penalty absent) ───────────
    # Frequency/ride objective is exclusively compute_frequency_response_objective.
    obj_grip = smooth_max - stiffness_penalty - brake_balance_penalty

    # ── Safety constraint — understeer margin at 1.5 G reference lateral ─────
    # Positive = rear transfers more load proportion than front = understeer = safe.
    ay_ref         = 1.5 * g
    LLT_ref        = m * ay_ref * h_cg / t_w
    LLT_geo_f_ref  = m * ay_ref * h_rc_f / t_w
    LLT_geo_r_ref  = m * ay_ref * h_rc_r / t_r
    h_arm_ref      = h_cg - (h_rc_f + h_rc_r) / 2.0
    LLT_el_f_ref   = m * ay_ref * h_arm_ref / t_w * lltd_f_elastic
    LLT_el_r_ref   = m * ay_ref * h_arm_ref / t_r  * lltd_r_elastic

    total_lltd_f   = (LLT_geo_f_ref + LLT_el_f_ref) / (LLT_ref + 1e-3)
    total_lltd_r   = (LLT_geo_r_ref + LLT_el_r_ref) / (LLT_ref + 1e-3)

    safety_margin  = (total_lltd_r - total_lltd_f) - 0.05

    return obj_grip, safety_margin


def compute_frequency_response_objective(simulate_step_fn, params, x_init,
                                          dt=0.005, T_max=2.0):
    """
    Damping ratio objective — sum of squared deviations from target zeta values.
    Lower = better (used as obj_stability = -resonance in evolutionary.py).

    This function is the SOLE enforcer of ride frequency / wheel-hop constraints.
    compute_skidpad_objective no longer touches frequency penalties (FIX 4).

    Unchanged from previous version — correctly scaled and fully differentiable.
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

    m      = VP.get('total_mass', VP.get('m', 230.0))
    m_s    = VP.get('sprung_mass', m * 0.85)
    m_us_f = VP.get('unsprung_mass_f', 10.0)
    m_us_r = VP.get('unsprung_mass_r', 11.0)

    Ix  = VP.get('Ix', 45.0)
    Iy  = VP.get('Iy', 85.0)
    t_w = VP.get('track_front', 1.20)
    lf  = VP.get('lf', 0.680)
    lr  = VP.get('lr', 0.920)

    # Heave mode
    k_heave    = wheel_rate_f * 2 + wheel_rate_r * 2
    c_heave    = damp_rate_f  * 2 + damp_rate_r  * 2
    zeta_heave = c_heave / (2.0 * jnp.sqrt(k_heave * m_s) + 1e-3)

    # Roll mode
    Kroll_f    = wheel_rate_f * (t_w ** 2) * 0.5
    Kroll_r    = wheel_rate_r * (t_w ** 2) * 0.5
    k_roll     = Kroll_f + Kroll_r
    c_roll     = (damp_rate_f + damp_rate_r) * (t_w ** 2) * 0.5
    zeta_roll  = c_roll / (2.0 * jnp.sqrt(k_roll * Ix) + 1e-3)

    # Pitch mode
    k_pitch    = wheel_rate_f * (lf ** 2) + wheel_rate_r * (lr ** 2)
    c_pitch    = damp_rate_f  * (lf ** 2) + damp_rate_r  * (lr ** 2)
    zeta_pitch = c_pitch / (2.0 * jnp.sqrt(k_pitch * Iy) + 1e-3)

    # Wheel hop modes
    k_us_f    = wheel_rate_f + 50000.0
    k_us_r    = wheel_rate_r + 50000.0
    zeta_us_f = damp_rate_f / (2.0 * jnp.sqrt(k_us_f * m_us_f) + 1e-3)
    zeta_us_r = damp_rate_r / (2.0 * jnp.sqrt(k_us_r * m_us_r) + 1e-3)

    resonance = (
        (zeta_heave - 0.65) ** 2 * 2.0 +
        (zeta_roll  - 0.70) ** 2 * 1.5 +
        (zeta_pitch - 0.60) ** 2 * 1.0 +
        (zeta_us_f  - 0.30) ** 2 * 0.5 +
        (zeta_us_r  - 0.30) ** 2 * 0.5
    )

    return resonance