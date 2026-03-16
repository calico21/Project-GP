# optimization/objectives.py
# Project-GP — Setup Optimization Objective Functions
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX2)
# ─────────────────────────────────────────────────────────────────────────────
# BUGFIX-A : lf / lr wrong fallback values in both objectives
#   PREVIOUS: lf=0.680, lr=0.920 — these are the pre-BUGFIX-2 defaults that
#   vehicle_dynamics.py explicitly corrected. Ter26 geometry: lf=0.8525, lr=0.6975.
#   Impact: L=1.600 instead of 1.550; front force split lr/L=0.575 vs correct
#   0.450 — the entire front/rear balance was biased to the front by 28%.
#
# BUGFIX-B : ARB roll stiffness physics — ARB does NOT go through motion ratio
#   PREVIOUS: arb_rate_f = arb_f / mr_f²; Kroll_f = (wheel_rate_f + arb_rate_f) * t_w² * 0.5
#   From dynamics: F_arb = arb_f * z_roll_f / tf2 = arb_f * φ (full roll angle)
#   → K_arb_roll [N·m/rad] = arb_f * t_w  (NOT arb_f * t_w² / (2·mr²))
#   At defaults: old formula gives 46% of the correct value — ARB contribution
#   to LLTD was underestimated by 54%, corrupting front/rear balance and
#   safety_margin for every setup with non-zero ARB rates.
#
# BUGFIX-C : safety_margin sign inverted
#   PREVIOUS: (total_lltd_r - total_lltd_f) - 0.05 — positive for oversteer.
#   A safe (understeering) FS car has LLTD_f > LLTD_r; the margin should be
#   positive when the car is understeer-biased.
#   FIX: (total_lltd_f - total_lltd_r) - 0.05
#   DOWNSTREAM NOTE: evolutionary.py STABILITY_MAX should change from 5.0 to
#   0.0 — currently the constraint passes trivially for all setups. After this
#   fix, only setups with safety_margin > 0 (understeer) should be accepted.
#
# BUGFIX-D : h_rc_f / h_rc_r fallback inconsistency with vehicle_dynamics.py
#   PREVIOUS: VP.get('h_rc_f', 0.030), VP.get('h_rc_r', 0.050)
#   vehicle_dynamics.py: vp.get('h_rc_f', 0.040), vp.get('h_rc_r', 0.060)
#   When VP lacks these keys, objectives and dynamics use different roll center
#   heights — the LLTD model diverges from the physics engine.
#
# BUGFIX-E : Cl_ref fallback wrong (3.0 vs 4.14 in vehicle_dynamics.py)
#   27% downforce underestimate in corner load when VP lacks 'Cl_ref'.
#   Fixed to 4.14 to match vehicle_dynamics.py DifferentiableAeroMap.
#
# BUGFIX-F : jnp.maximum for corner Fz inside vmapped+differentiated function
#   compute_balance_at_ay is called via jax.vmap and the result flows into
#   the MORL Adam gradient via compute_skidpad_objective. jnp.maximum has
#   zero subgradient below the floor — gradient vanishes precisely when a
#   corner goes light, the most setup-sensitive regime. Replaced with
#   _softplus_floor for consistent gradient flow, matching UPGRADE-7 in
#   vehicle_dynamics.py.
#
# FIXED (from GP-vX1, retained):
#   · PDY1 corrected from 2.218×0.6=1.33 → 1.92
#   · ay_sweep extended from [0.8, 2.0] G to [0.5, 2.5] G, 300→1000 points
#   · _LSE_BETA raised from 10 → 20
#   · freq_penalty absent from grip objective
#   · normalised centering penalty
# ═══════════════════════════════════════════════════════════════════════════════

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _softplus_floor(x: jax.Array, floor: float) -> jax.Array:
    """
    Smooth lower bound consistent with vehicle_dynamics.py UPGRADE-7.
    df/dx = sigmoid(x - floor) ∈ (0,1) — never zero.
    Replaces jnp.maximum whose sub-gradient is zero below the floor,
    killing optimizer signal when corner loads go light.
    """
    return floor + jax.nn.softplus(x - floor)


# ─────────────────────────────────────────────────────────────────────────────
# PENALTY SCALE REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
# The raw grip signal is bounded in [0.5, 2.5] G after the ay_sweep extension.
# Every penalty coefficient is sized so that the worst physically realistic
# violation costs at most 0.10 G — 4% of the signal range.

# BUG FIX: raised from 10 → 20 to halve the smooth-max bias
_LSE_BETA = 20.0


def _expand_8_to_28_setup(params_8: jax.Array) -> jax.Array:
    """
    Expand the 8-element MORL setup vector to the 28-element P10 format.
    Delegates to SuspensionSetup.from_legacy_8 — canonical construction path.

    8-param MORL layout → 28-param SuspensionSetup indices (SETUP_NAMES order):
      params_8[0]  k_f          → setup[0]   k_f
      params_8[1]  k_r          → setup[1]   k_r
      params_8[2]  arb_f        → setup[2]   arb_f
      params_8[3]  arb_r        → setup[3]   arb_r
      params_8[4]  c_f          → setup[4]   c_low_f
      params_8[5]  c_r          → setup[5]   c_low_r
      params_8[6]  h_cg         → setup[25]  h_cg
      params_8[7]  brake_bias_f → setup[24]  brake_bias_f
    All other 28-param fields populated from DEFAULT_SETUP.
    """
    from models.vehicle_dynamics import SuspensionSetup
    return SuspensionSetup.from_legacy_8(params_8).to_vector()


def compute_step_steer_objective(simulate_step_fn, setup_params, x_init):
    """
    Step-steer transient: applies δ=0.08 rad at t=0 on a straight.
    Measures yaw rate overshoot and settling time via 40-step rollout.

    Returns: -overshoot_penalty (higher = better damped = more stable transient)

    P10 SETUP NOTE: accepts shape (28,) or (8,); 8-element vectors are
    automatically expanded via _expand_8_to_28_setup.
    """
    if setup_params.shape[-1] == 8:
        setup_params = _expand_8_to_28_setup(setup_params)
    elif setup_params.shape[-1] != 28:
        raise ValueError(
            f"compute_step_steer_objective: setup_params must be shape (8,) or "
            f"(28,). Got {setup_params.shape}."
        )

    dt     = 0.005
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
    overshoot_cost  = jnp.abs(overshoot_ratio - 1.0)

    settling_cost   = wz_final / jnp.maximum(wz_initial, 0.01)

    return -(overshoot_cost + 0.5 * settling_cost)


def compute_skidpad_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Differentiable analytical steady-state cornering balance.
    Penalty functions are scaled so max violation costs ~0.10 G.

    Returns (obj_grip, safety_margin) where:
      obj_grip      — smooth-max lateral acceleration [G], penalised for
                      stiffness imbalance and brake bias error.
      safety_margin — (LLTD_f - LLTD_r) - 0.05
                      POSITIVE = understeering (safe, preferred for FS)
                      NEGATIVE = oversteering (unsafe)
                      NOTE: evolutionary.py STABILITY_MAX should be 0.0,
                      not 5.0, to make this constraint physically meaningful.
    """
    from data.configs.vehicle_params import vehicle_params as VP

    # ── Unpack setup parameters ───────────────────────────────────────────────
    if params.shape[-1] == 28:
        k_f          = params[0]
        k_r          = params[1]
        arb_f        = params[2]    # anti-roll bar front [N/m equivalent]
        arb_r        = params[3]    # anti-roll bar rear  [N/m equivalent]
        c_f          = params[4]    # c_low_f [N·s/m]
        c_r          = params[5]    # c_low_r [N·s/m]
        h_cg         = params[25]   # CG height [m]
        brake_bias_f = params[24]   # front brake bias [-]
    else:
        k_f          = params[0]
        k_r          = params[1]
        arb_f        = params[2]
        arb_r        = params[3]
        c_f          = params[4]
        c_r          = params[5]
        h_cg         = params[6]
        brake_bias_f = params[7]

    # BUGFIX-A: correct Ter26 geometry defaults (was lf=0.680, lr=0.920)
    # BUGFIX-D: correct roll center defaults (was h_rc_f=0.030, h_rc_r=0.050)
    # BUGFIX-E: correct aero Cl default (was Cl_ref=3.0)
    # All fallbacks now consistent with vehicle_dynamics.py.
    # BUGFIX-A (motion ratio): was [1.20]/[1.15]; vehicle_dynamics uses [1.14]/[1.16]
    mr_f = jnp.array(VP.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))[0]
    mr_r = jnp.array(VP.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))[0]

    wheel_rate_f = k_f / (mr_f ** 2)
    wheel_rate_r = k_r / (mr_r ** 2)

    h_rc_f = VP.get('h_rc_f', 0.040)    # BUGFIX-D: was 0.030
    h_rc_r = VP.get('h_rc_r', 0.060)    # BUGFIX-D: was 0.050

    t_w = VP.get('track_front', 1.20)
    t_r = VP.get('track_rear',  1.18)

    # ── Roll stiffness (N·m / rad) ─────────────────────────────────────────────
    # Spring contribution: K_spring_roll = wheel_rate * (t/2)^2 * 2 = wheel_rate * t^2 / 2
    # ARB contribution: from dynamics F_arb = arb * φ → K_arb_roll = arb * t
    # BUGFIX-B: ARB goes through track width t, NOT through motion ratio MR.
    #   Old: arb_rate = arb / MR² → K_arb = arb * t² / (2*MR²) — 54% too low.
    #   New: K_arb = arb * t  — derived directly from dynamics EOM.
    Kroll_f_spring = wheel_rate_f * (t_w ** 2) * 0.5
    Kroll_r_spring = wheel_rate_r * (t_r  ** 2) * 0.5
    Kroll_f_arb    = arb_f * t_w      # BUGFIX-B: arb * track, not arb/MR² * t²/2
    Kroll_r_arb    = arb_r * t_r      # BUGFIX-B
    Kroll_f        = Kroll_f_spring + Kroll_f_arb
    Kroll_r        = Kroll_r_spring + Kroll_r_arb
    Kroll_total    = Kroll_f + Kroll_r + 1.0   # +1 for numerical stability

    lltd_f_elastic = Kroll_f / Kroll_total
    lltd_r_elastic = Kroll_r / Kroll_total

    m  = VP.get('total_mass', VP.get('m', 230.0))
    lf = VP.get('lf', 0.8525)    # BUGFIX-A: was 0.680
    lr = VP.get('lr', 0.6975)    # BUGFIX-A: was 0.920
    L  = lf + lr
    g  = 9.81

    # PRIMARY BUG FIX (GP-vX1, retained): PDY1 corrected 2.218×0.6=1.33 → 1.92
    PDY1 = 1.92
    PDY2 = -0.25
    Fz0  = 1000.0

    static_camber = VP.get('static_camber_f', -1.5)
    camber_gain_f = VP.get('camber_gain_f', -0.8)

    # Static axle loads
    Fz_f_static = m * g * lr / L
    Fz_r_static = m * g * lf / L

    # Aerodynamic downforce at nominal skidpad speed (15 m/s)
    v_corner     = 15.0
    rho          = VP.get('rho_air', 1.225)
    A            = VP.get('A_ref',   1.1)
    Cl           = VP.get('Cl_ref',  4.14)    # BUGFIX-E: was 3.0
    Fz_aero      = 0.5 * rho * Cl * A * v_corner ** 2
    aero_split_f = VP.get('aero_split_f', 0.40)
    aero_split_r = VP.get('aero_split_r', 0.60)
    Fz_f_static  = Fz_f_static + Fz_aero * aero_split_f
    Fz_r_static  = Fz_r_static + Fz_aero * aero_split_r

    ay_sweep = jnp.linspace(0.5, 2.5, 1000)

    def compute_balance_at_ay(ay_g):
        ay = ay_g * g

        LLT_geo_f     = m * ay * h_rc_f / t_w
        LLT_geo_r     = m * ay * h_rc_r / t_r

        h_arm_f       = h_cg - h_rc_f
        h_arm_r       = h_cg - h_rc_r
        LLT_elastic_f = m * ay * h_arm_f / t_w * lltd_f_elastic
        LLT_elastic_r = m * ay * h_arm_r / t_r * lltd_r_elastic

        LLT_f = LLT_geo_f + LLT_elastic_f
        LLT_r = LLT_geo_r + LLT_elastic_r

        # BUGFIX-F: _softplus_floor replaces jnp.maximum — gradient alive at floor.
        # Inside vmap + MORL Adam gradient, jnp.maximum produces zero subgradient
        # precisely when corner loads go light — the most setup-sensitive regime.
        Fz_fo = _softplus_floor(Fz_f_static / 2.0 + LLT_f, 10.0)
        Fz_fi = _softplus_floor(Fz_f_static / 2.0 - LLT_f, 10.0)
        Fz_ro = _softplus_floor(Fz_r_static / 2.0 + LLT_r, 10.0)
        Fz_ri = _softplus_floor(Fz_r_static / 2.0 - LLT_r, 10.0)

        inner_lift_f = jax.nn.relu(50.0 - (Fz_f_static / 2.0 - LLT_f))
        inner_lift_r = jax.nn.relu(50.0 - (Fz_r_static / 2.0 - LLT_r))
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

    k_ref_centering   = 25000.0
    w_centering       = 0.01
    centering_penalty = w_centering * (
        ((k_f - k_ref_centering) / k_ref_centering) ** 2 +
        ((k_r - k_ref_centering) / k_ref_centering) ** 2
    )

    stiffness_penalty = (
        (jnp.abs(PDY2) * (fz_variation_f / Fz0) +
         jnp.abs(PDY2) * (fz_variation_r / Fz0)) * 0.4
        + centering_penalty
    )

    Fz_f_brake          = (m * g * lr / L) + (m * 1.0 * g * h_cg / L)
    Fz_r_brake          = (m * g * lf / L) - (m * 1.0 * g * h_cg / L)
    ideal_bias          = Fz_f_brake / (Fz_f_brake + Fz_r_brake)
    brake_balance_penalty = 3.0 * (brake_bias_f - ideal_bias) ** 2

    obj_grip = smooth_max - stiffness_penalty - brake_balance_penalty

    # ── Safety margin: LLTD front-rear split ──────────────────────────────────
    # Compute at reference lateral acceleration 1.5G
    ay_ref        = 1.5 * g
    LLT_ref       = m * ay_ref * h_cg / ((t_w + t_r) * 0.5 + 1e-3)
    LLT_geo_f_ref = m * ay_ref * h_rc_f / t_w
    LLT_geo_r_ref = m * ay_ref * h_rc_r / t_r
    h_arm_ref_f   = h_cg - h_rc_f
    h_arm_ref_r   = h_cg - h_rc_r
    LLT_el_f_ref  = m * ay_ref * h_arm_ref_f / t_w * lltd_f_elastic
    LLT_el_r_ref  = m * ay_ref * h_arm_ref_r / t_r * lltd_r_elastic

    total_lltd_f  = (LLT_geo_f_ref + LLT_el_f_ref) / (LLT_ref + 1e-3)
    total_lltd_r  = (LLT_geo_r_ref + LLT_el_r_ref) / (LLT_ref + 1e-3)

    # BUGFIX-C: sign flipped. Previous formula (lltd_r - lltd_f) was positive
    # for oversteer (unsafe). Correct: positive = understeer (safe, LLTD_f > LLTD_r).
    # IMPORTANT: evolutionary.py STABILITY_MAX must be updated from 5.0 → 0.0
    # to enforce the understeer constraint. With STABILITY_MAX=5.0 the constraint
    # passes trivially regardless of sign convention.
    safety_margin = (total_lltd_f - total_lltd_r) - 0.05

    return obj_grip, safety_margin


def compute_frequency_response_objective(simulate_step_fn, params, x_init,
                                         dt=0.005, T_max=2.0):
    """
    Analytical modal damping ratio objective.
    Penalises deviation from target damping ratios for heave, roll, pitch,
    and wheel hop modes. Fully differentiable — no simulation rollout required.

    Returns: resonance penalty (lower = better modal behaviour)
    """
    from data.configs.vehicle_params import vehicle_params as VP

    if params.shape[-1] == 28:
        k_f, k_r = params[0], params[1]
        c_f, c_r = params[4], params[5]    # c_low_f, c_low_r
    else:
        k_f, k_r = params[0], params[1]
        c_f, c_r = params[4], params[5]

    # BUGFIX-A: correct motion ratio fallbacks (was [1.20]/[1.15])
    mr_f = jnp.array(VP.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))[0]
    mr_r = jnp.array(VP.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))[0]

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
    lf  = VP.get('lf', 0.8525)     # BUGFIX-A: was 0.680
    lr  = VP.get('lr', 0.6975)     # BUGFIX-A: was 0.920

    # Heave mode: all four corners contribute
    k_heave    = wheel_rate_f * 2.0 + wheel_rate_r * 2.0
    c_heave    = damp_rate_f  * 2.0 + damp_rate_r  * 2.0
    zeta_heave = c_heave / (2.0 * jnp.sqrt(k_heave * m_s) + 1e-3)

    # Roll mode: anti-symmetric heave (spring only; ARB contribution via wheel rate)
    Kroll_f    = wheel_rate_f * (t_w ** 2) * 0.5
    Kroll_r    = wheel_rate_r * (t_w ** 2) * 0.5
    k_roll     = Kroll_f + Kroll_r
    c_roll     = (damp_rate_f + damp_rate_r) * (t_w ** 2) * 0.5
    zeta_roll  = c_roll / (2.0 * jnp.sqrt(k_roll * Ix) + 1e-3)

    # Pitch mode: front/rear asymmetric heave
    k_pitch    = wheel_rate_f * (lf ** 2) + wheel_rate_r * (lr ** 2)
    c_pitch    = damp_rate_f  * (lf ** 2) + damp_rate_r  * (lr ** 2)
    zeta_pitch = c_pitch / (2.0 * jnp.sqrt(k_pitch * Iy) + 1e-3)

    # Wheel hop modes (unsprung vs ground)
    k_us_f    = wheel_rate_f + 50000.0   # tire radial stiffness ~50 kN/m
    k_us_r    = wheel_rate_r + 50000.0
    zeta_us_f = damp_rate_f / (2.0 * jnp.sqrt(k_us_f * m_us_f) + 1e-3)
    zeta_us_r = damp_rate_r / (2.0 * jnp.sqrt(k_us_r * m_us_r) + 1e-3)

    # Target damping ratios — weighted penalty for deviation
    # ζ_heave=0.65: comfortable road holding; ζ_roll=0.70: controlled transient;
    # ζ_pitch=0.60: acceptable nose dive; ζ_us=0.30: classic unsprung target.
    resonance = (
        (zeta_heave - 0.65) ** 2 * 2.0 +
        (zeta_roll  - 0.70) ** 2 * 1.5 +
        (zeta_pitch - 0.60) ** 2 * 1.0 +
        (zeta_us_f  - 0.30) ** 2 * 0.5 +
        (zeta_us_r  - 0.30) ** 2 * 0.5
    )

    return resonance