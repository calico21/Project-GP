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
# ── P10 SETUP FIX — compute_step_steer_objective now accepts 8 OR 28 params ──
#
#   Root cause:
#   compute_step_steer_objective passes setup_params directly to simulate_step_fn.
#   In P10, simulate_step → _compute_derivatives unpacks 28 elements from
#   setup_params.  Calling this function with the MORL's 8-element params
#   (k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f) caused a JAX
#   unpack error / shape mismatch.
#
#   Fix: expand 8-element params to 28 before passing to simulate_step_fn.
#   The expansion uses the same logic as _build_default_setup_28 in ocp_solver,
#   but implemented inline as pure JAX operations so it remains JIT-compatible
#   and differentiable through the full compute graph.
#
#   The 8 MORL parameters map to these 28-param positions:
#     [0] k_f          → [0]  k_f
#     [1] k_r          → [1]  k_r
#     [2] arb_f        → [4]  arb_f
#     [3] arb_r        → [5]  arb_r
#     [4] c_f          → [6]  c_ls_bump_f (×0.60)
#                      → [7]  c_hs_bump_f (×0.40)
#                      → [8]  c_ls_reb_f  (×0.90)   (LS bump × 1.5)
#                      → [9]  c_hs_reb_f  (×0.60)   (HS bump × 1.5)
#     [5] c_r          → [10] c_ls_bump_r (×0.60)
#                      → [11] c_hs_bump_r (×0.40)
#                      → [12] c_ls_reb_r  (×0.90)
#                      → [13] c_hs_reb_r  (×0.60)
#     [6] h_cg         → [21] h_cg_setup
#     [7] brake_bias_f → [22] brake_bias_f
#   Remaining 16 positions: PhysicsNormalizer means (sensible FS defaults).
#
# ── PRIMARY BUG FIX: PDY1 corrected from 2.218 × 0.6 → 1.92 ─────────────────
# ── ay_sweep extended from [0.8, 2.0] G to [0.5, 2.5] G, 300→1000 points ─────
# ── _LSE_BETA raised from 10 → 20 ─────────────────────────────────────────────
# ── FIX 4 (retained): freq_penalty absent from grip objective ─────────────────
# ── FIX 5 (retained): normalised centering penalty ────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ── BUG FIX: raised from 10 → 20 to halve the smooth-max bias ────────────────
_LSE_BETA = 20.0


def _expand_8_to_28_setup(params_8):
    """
    Expand the 8-element MORL setup vector to the 28-element P10 format.

    This is a pure JAX operation — JIT-compatible and differentiable.
    Used by compute_step_steer_objective so it can accept either an 8-element
    (MORL/analytical) or 28-element (full P10) setup vector.

    Parameters
    ----------
    params_8 : jnp.ndarray shape (8,)
        [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f]

    Returns
    -------
    params_28 : jnp.ndarray shape (28,)
        Full P10 setup vector.  Unmapped positions filled with
        PhysicsNormalizer.setup_mean values (sensible FS defaults).
    """
    import jax.numpy as jnp
    from models.vehicle_dynamics import PhysicsNormalizer

    k_f          = params_8[0]
    k_r          = params_8[1]
    arb_f        = params_8[2]
    arb_r        = params_8[3]
    c_f          = params_8[4]
    c_r          = params_8[5]
    h_cg         = params_8[6]
    brake_bias_f = params_8[7]

    # 4-way damper split — identical to _build_default_setup_28
    c_ls_bump_f = c_f * 0.60
    c_hs_bump_f = c_f * 0.40
    c_ls_reb_f  = c_f * 0.90    # 0.60 × 1.5
    c_hs_reb_f  = c_f * 0.60    # 0.40 × 1.5

    c_ls_bump_r = c_r * 0.60
    c_hs_bump_r = c_r * 0.40
    c_ls_reb_r  = c_r * 0.90
    c_hs_reb_r  = c_r * 0.60

    # Defaults for unmapped positions from PhysicsNormalizer.setup_mean
    m = PhysicsNormalizer.setup_mean

    return jnp.array([
        k_f,           k_r,            # [0-1]  springs
        m[2],          m[3],           # [2-3]  k_heave_f/r (default 5000 N/m)
        arb_f,         arb_r,          # [4-5]  ARBs
        c_ls_bump_f,   c_hs_bump_f,    # [6-7]  front damper bump
        c_ls_reb_f,    c_hs_reb_f,     # [8-9]  front damper rebound
        c_ls_bump_r,   c_hs_bump_r,    # [10-11] rear damper bump
        c_ls_reb_r,    c_hs_reb_r,     # [12-13] rear damper rebound
        m[14],         m[15],          # [14-15] v_knee_f/r (default 0.10 m/s)
        m[16],         m[17],          # [16-17] toe_f/r
        m[18],         m[19],          # [18-19] camber_f/r_deg
        m[20],                         # [20]    caster_deg
        h_cg,                          # [21]    h_cg_setup
        brake_bias_f,                  # [22]    brake_bias_f
        m[23],                         # [23]    diff_lock (default 1.0)
        m[24],         m[25],          # [24-25] arb_preload_f/r (default 0.0)
        m[26],         m[27],          # [26-27] bumpstop_gap_f/r
    ])


def compute_step_steer_objective(simulate_step_fn, setup_params, x_init):
    """
    Step-steer transient: applies δ=0.08 rad at t=0 on a straight.
    Measures yaw rate overshoot and settling time via 40-step rollout.
    Damping (c_f, c_r) strongly affects this — activates the zero-sensitivity gap.

    Returns: -overshoot_penalty (higher = better damped = more stable transient)
    A well-damped car: wz peaks once then settles. Overdamped: slow response.
    Target: critically damped response (ζ≈0.7), penalise both over and under.

    P10 SETUP NOTE
    ──────────────
    simulate_step_fn (DifferentiableMultiBodyVehicle.simulate_step) requires
    setup_params of shape (28,) in P10.  This function accepts EITHER:
      • shape (28,) — passed directly to simulate_step_fn unchanged
      • shape (8,)  — automatically expanded to (28,) via _expand_8_to_28_setup
                      so the MORL's analytical 8-param vector can be used here
                      without modifying the MORL caller.

    Callers passing 8-element params will lose fine damper control (4-way split
    is approximated from the single c_f/c_r value) but the yaw response signal
    is still physically meaningful and differentiable.
    """
    # ── P10 setup expansion ───────────────────────────────────────────────────
    # Python-level shape check is safe here because this function is not itself
    # decorated with @jax.jit — it is called inside jitted callers, but the
    # shape check executes at trace time (Python level) not at runtime.
    if setup_params.shape[-1] == 8:
        setup_params = _expand_8_to_28_setup(setup_params)
    elif setup_params.shape[-1] != 28:
        raise ValueError(
            f"compute_step_steer_objective: setup_params must have shape (8,) or (28,). "
            f"Got shape {setup_params.shape}. "
            f"Pass either the 8-element MORL vector or a full 28-element P10 setup."
        )

    dt = 0.005
    u_step = jnp.array([0.08, 500.0])   # steering step + mild throttle

    def rollout_step(carry, _):
        x = carry
        x = simulate_step_fn(x, u_step, setup_params, dt)
        return x, x[19]   # carry state, emit wz

    _, wz_history = jax.lax.scan(rollout_step, x_init, None, length=40)

    wz_peak    = jnp.max(jnp.abs(wz_history))
    wz_final   = jnp.abs(wz_history[-1])
    wz_initial = jnp.abs(wz_history[5])   # after first ~25ms

    # Overshoot ratio: ideal=1.0 (no overshoot), >1.3 = underdamped, <0.6 = overdamped
    overshoot_ratio = wz_peak / jnp.maximum(wz_initial, 0.01)
    overshoot_cost  = jnp.abs(overshoot_ratio - 1.0)   # 0=ideal

    # Settling: wz should decay, not oscillate
    settling_cost = wz_final / jnp.maximum(wz_initial, 0.01)

    # Combined: lower = better transient response
    return -(overshoot_cost + 0.5 * settling_cost)


def compute_skidpad_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Differentiable analytical steady-state cornering balance.

    API NOTE: simulate_step_fn, dt, and T_max are accepted for interface
    compatibility but are NOT called inside this function.  The cornering
    balance is computed analytically, not by integrating the ODE.  This is
    intentional — the analytical formulation is exact, fully differentiable
    everywhere, and does not require a simulation horizon to reach steady-state.

    SETUP PARAMS NOTE: this function reads only params[0..7] (k_f, k_r, arb_f,
    arb_r, c_f, c_r, h_cg, brake_bias_f) — it is compatible with both the
    8-element MORL vector and a full 28-element P10 vector.  No expansion needed.

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
    PDY1 = 1.92
    PDY2 = -0.25
    Fz0  = 1000.0

    static_camber = VP.get('static_camber_f', -1.5)
    camber_gain_f = VP.get('camber_gain_f', -0.8)

    # Static axle loads
    Fz_f_static = m * g * lr / L
    Fz_r_static = m * g * lf / L

    # Aerodynamic downforce at nominal skidpad speed (15 m/s ≈ 54 km/h)
    v_corner     = 15.0
    rho          = VP.get('rho_air', 1.225)
    A            = VP.get('A_ref',   1.1)
    Cl           = VP.get('Cl_ref',  3.0)
    Fz_aero      = 0.5 * rho * Cl * A * v_corner ** 2

    aero_split_f = VP.get('aero_split_f', 0.40)
    aero_split_r = VP.get('aero_split_r', 0.60)
    Fz_f_static  += Fz_aero * aero_split_f
    Fz_r_static  += Fz_aero * aero_split_r

    ay_sweep = jnp.linspace(0.5, 2.5, 1000)

    def compute_balance_at_ay(ay_g):
        ay = ay_g * g

        LLT_geo_f = m * ay * h_rc_f / t_w
        LLT_geo_r = m * ay * h_rc_r / t_r

        h_arm_f       = h_cg - h_rc_f
        h_arm_r       = h_cg - h_rc_r
        LLT_elastic_f = m * ay * h_arm_f / t_w * lltd_f_elastic
        LLT_elastic_r = m * ay * h_arm_r / t_r  * lltd_r_elastic

        LLT_f = LLT_geo_f + LLT_elastic_f
        LLT_r = LLT_geo_r + LLT_elastic_r

        Fz_fo = jnp.maximum(10.0, Fz_f_static / 2 + LLT_f)
        Fz_fi = jnp.maximum(10.0, Fz_f_static / 2 - LLT_f)
        Fz_ro = jnp.maximum(10.0, Fz_r_static / 2 + LLT_r)
        Fz_ri = jnp.maximum(10.0, Fz_r_static / 2 - LLT_r)

        inner_lift_f = jax.nn.relu(50.0 - (Fz_f_static / 2 - LLT_f))
        inner_lift_r = jax.nn.relu(50.0 - (Fz_r_static / 2 - LLT_r))
        lift_penalty = (inner_lift_f + inner_lift_r) * 0.0005

        phi_est              = (m * ay * h_cg) / (Kroll_total + 1.0)
        phi_deg              = jnp.rad2deg(phi_est)
        effective_camber_out = static_camber + phi_deg * camber_gain_f
        camber_opt           = -3.5
        camber_bonus         = 1.0 + 0.03 * jnp.exp(
            -0.5 * ((effective_camber_out - camber_opt) / 2.0) ** 2
        )

        def mu(Fz):
            dfz = (Fz - Fz0) / Fz0
            return PDY1 * (1.0 + PDY2 * dfz)

        Fy_f_max = mu(Fz_fo) * Fz_fo * camber_bonus + mu(Fz_fi) * Fz_fi
        Fy_r_max = mu(Fz_ro) * Fz_ro * camber_bonus + mu(Fz_ri) * Fz_ri

        Fy_required = m * ay
        Fy_f_req    = Fy_required * lr / L
        Fy_r_req    = Fy_required * lf / L

        util_f = Fy_f_req / (Fy_f_max + 1e-3)
        util_r = Fy_r_req / (Fy_r_max + 1e-3)

        balance = 1.0 - jnp.abs(util_f - util_r)

        sharpness     = 10.0
        feasible_soft = (jax.nn.sigmoid((1.0 - util_f) * sharpness) *
                         jax.nn.sigmoid((1.0 - util_r) * sharpness))

        return ay_g * balance * feasible_soft - lift_penalty

    grip_scores = jax.vmap(compute_balance_at_ay)(ay_sweep)

    smooth_max = (1.0 / _LSE_BETA) * jax.nn.logsumexp(_LSE_BETA * grip_scores)

    bump_rms        = 0.007
    fz_variation_f  = wheel_rate_f * bump_rms
    fz_variation_r  = wheel_rate_r * bump_rms

    k_ref_centering = 25000.0
    w_centering     = 0.01

    centering_penalty = w_centering * (
        ((k_f - k_ref_centering) / k_ref_centering) ** 2 +
        ((k_r - k_ref_centering) / k_ref_centering) ** 2
    )

    stiffness_penalty = (
        (jnp.abs(PDY2) * (fz_variation_f / Fz0) +
         jnp.abs(PDY2) * (fz_variation_r / Fz0)) * 0.4
        + centering_penalty
    )

    Fz_f_brake   = (m * g * lr / L) + (m * 1.0 * g * h_cg / L)
    Fz_r_brake   = (m * g * lf / L) - (m * 1.0 * g * h_cg / L)
    ideal_bias   = Fz_f_brake / (Fz_f_brake + Fz_r_brake)
    brake_balance_penalty = 3.0 * (brake_bias_f - ideal_bias) ** 2

    obj_grip = smooth_max - stiffness_penalty - brake_balance_penalty

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

    SETUP PARAMS NOTE: reads only params[0..1] (k_f, k_r) and params[4..5]
    (c_f, c_r) — compatible with both 8-element and 28-element setup vectors.

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