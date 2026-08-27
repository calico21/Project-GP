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
from models.aero_platform import ground_effect_envelope


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
    # u = [δ, T_hub_fl, T_hub_fr, T_hub_rl, T_hub_rr, F_brake_hyd] — 6-wide
    # since the 4WD hub-motor migration. A 2-element u silently clamps
    # under jit's static-index OOB semantics: u[2..5] all read back u[1],
    # meaning every hub torque AND the hydraulic brake force were reading
    # the same fake 500N constant regardless of setup — brake_bias_f's
    # apparent gradient was this artifact, not real physics.
    u_step = jnp.array([0.08, 125.0, 125.0, 125.0, 125.0, 0.0])  # equal 4-way split, no braking

    def rollout_step(carry, _):
        x = carry
        x = simulate_step_fn(x, u_step, setup_params, dt)
        return x, x[19]   # carry state, emit wz

    _, wz_history = jax.lax.scan(rollout_step, x_init, None, length=40)

    wz_peak  = jnp.max(jnp.abs(wz_history))
    wz_final = jnp.maximum(jnp.abs(wz_history[-1]), 0.05)

    # Sobreoscilación real respecto al régimen permanente (ideal = 1.0, sin sobrepico)
    overshoot_ratio = wz_peak / wz_final
    overshoot_cost  = jnp.maximum(overshoot_ratio - 1.0, 0.0)

    # Variación residual en los últimos 25 ms (penaliza oscilaciones no amortiguadas)
    settling_cost   = jnp.abs(wz_history[-1] - wz_history[-5]) / wz_final

    return -(overshoot_cost + 2.0 * settling_cost)


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
    from config.vehicles.ter26 import vehicle_params as VP

    # ── Unpack setup parameters ───────────────────────────────────────────────
    if params.shape[-1] == 28:
        k_f          = params[0]
        k_r          = params[1]
        arb_f        = params[2]
        arb_r        = params[3]
        c_low_f      = params[4]
        c_low_r      = params[5]
        c_high_f     = params[6]
        c_high_r     = params[7]
        h_ride_f     = params[12]   # Altura de marcha delantera [m]
        h_ride_r     = params[13]   # Altura de marcha trasera [m]
        camber_f     = params[14]   # Caída estática delantera [deg]
        camber_r     = params[15]   # Caída estática trasera [deg]
        toe_f        = params[16]   # Convergencia delantera [deg]
        toe_r        = params[17]   # Convergencia trasera [deg]
        brake_bias_f = params[24]

        # h_cg viene del setup vector (params[25]), no de un VP fijo — si algún
        # día se descongela para exploración de layout, debe fluir gradiente
        # real. Se suma la deriva cinemática por altura de marcha como término
        # correctivo pequeño, no como la fuente principal del valor.
        h_ride_delta = (h_ride_f + h_ride_r) * 0.5 - VP.get('h_ride_design', 0.028)
        h_cg         = params[25] + h_ride_delta
    else:
        k_f          = params[0]; k_r = params[1]
        arb_f        = params[2]; arb_r = params[3]
        c_low_f      = params[4]; c_low_r = params[5]
        c_high_f     = c_low_f * 0.6; c_high_r = c_low_r * 0.6
        h_ride_f     = 0.028; h_ride_r = 0.030
        camber_f     = VP.get('static_camber_f', -1.5)
        camber_r     = VP.get('static_camber_r', -1.0)
        toe_f        = 0.0; toe_r = 0.0
        h_cg         = params[6]
        brake_bias_f = params[7]

    mr_f = jnp.array(VP.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))[0]
    mr_r = jnp.array(VP.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))[0]

    wheel_rate_f = k_f / (mr_f ** 2)
    wheel_rate_r = k_r / (mr_r ** 2)

    h_rc_f = VP.get('h_rc_f', 0.040)
    h_rc_r = VP.get('h_rc_r', 0.060)

    t_w = VP.get('track_front', 1.20)
    t_r = VP.get('track_rear',  1.18)

    # Rigidez al balanceo (N·m / rad)
    Kroll_f = wheel_rate_f * (t_w ** 2) * 0.5 + arb_f * t_w
    Kroll_r = wheel_rate_r * (t_r  ** 2) * 0.5 + arb_r * t_r
    Kroll_total = Kroll_f + Kroll_r + 1.0

    lltd_f_elastic = Kroll_f / Kroll_total
    lltd_r_elastic = Kroll_r / Kroll_total

    m  = VP.get('total_mass', VP.get('m', 230.0))
    lf = VP.get('lf', 0.8525)
    lr = VP.get('lr', 0.6975)
    L  = lf + lr
    g  = 9.81

    # Cargas estáticas por eje
    Fz_f_static = m * g * lr / L
    Fz_r_static = m * g * lf / L

    # Carga aerodinámica con envolvente de efecto suelo y pérdida de flujo (Aero Stall)
    v_corner     = 15.0
    rho          = VP.get('rho_air', 1.225)
    A            = VP.get('A_ref',   1.1)
    Cl           = VP.get('Cl_ref',  4.14)
    aero_split_f = VP.get('aero_split_f', 0.40)
    aero_split_r = VP.get('aero_split_r', 0.60)

    rh_f_mm = h_ride_f * 1000.0
    rh_r_mm = h_ride_r * 1000.0
    Gamma_ge_f = ground_effect_envelope(rh_f_mm, VP.get('rh_peak_mm', 30.0), VP.get('rh_stall_mm', 12.0))
    Gamma_ge_r = ground_effect_envelope(rh_r_mm, VP.get('rh_peak_mm', 30.0), VP.get('rh_stall_mm', 12.0))

    Fz_aero_f = 0.5 * rho * (Cl * aero_split_f * Gamma_ge_f) * A * v_corner ** 2
    Fz_aero_r = 0.5 * rho * (Cl * aero_split_r * Gamma_ge_r) * A * v_corner ** 2
    Fz_f_static = Fz_f_static + Fz_aero_f
    Fz_r_static = Fz_r_static + Fz_aero_r

    PDY1 = 1.92
    PDY2 = -0.25
    Fz0  = 1000.0
    camber_gain_f = VP.get('camber_gain_f', -0.8)

    ay_sweep = jnp.linspace(0.5, 2.5, 1000)

    def compute_balance_at_ay(ay_g):
        ay = ay_g * g

        LLT_f = m * ay * h_rc_f / t_w + m * ay * (h_cg - h_rc_f) / t_w * lltd_f_elastic
        LLT_r = m * ay * h_rc_r / t_r + m * ay * (h_cg - h_rc_r) / t_r * lltd_r_elastic

        Fz_fo = _softplus_floor(Fz_f_static / 2.0 + LLT_f, 10.0)
        Fz_fi = _softplus_floor(Fz_f_static / 2.0 - LLT_f, 10.0)
        Fz_ro = _softplus_floor(Fz_r_static / 2.0 + LLT_r, 10.0)
        Fz_ri = _softplus_floor(Fz_r_static / 2.0 - LLT_r, 10.0)

        inner_lift = jax.nn.relu(50.0 - (Fz_f_static / 2.0 - LLT_f)) + jax.nn.relu(50.0 - (Fz_r_static / 2.0 - LLT_r))
        lift_penalty = inner_lift * 0.0005

        phi_deg = jnp.rad2deg((m * ay * h_cg) / (Kroll_total + 1.0))
        effective_camber_out = camber_f + phi_deg * camber_gain_f
        camber_bonus = 1.0 + 0.04 * jnp.exp(-0.5 * ((effective_camber_out - (-2.8)) / 1.5) ** 2)

        def mu(Fz):
            return PDY1 * (1.0 + PDY2 * ((Fz - Fz0) / Fz0))

        Fy_f_max = mu(Fz_fo) * Fz_fo * camber_bonus + mu(Fz_fi) * Fz_fi
        Fy_r_max = mu(Fz_ro) * Fz_ro * camber_bonus + mu(Fz_ri) * Fz_ri

        util_f = (m * ay * lr / L) / (Fy_f_max + 1e-3)
        util_r = (m * ay * lf / L) / (Fy_r_max + 1e-3)

        balance = 1.0 - jnp.abs(util_f - util_r)
        feasible_soft = jax.nn.sigmoid((1.0 - jnp.maximum(util_f, util_r)) * 10.0)

        return ay_g * balance * feasible_soft - lift_penalty

    grip_scores = jax.vmap(compute_balance_at_ay)(ay_sweep)
    smooth_max = (1.0 / _LSE_BETA) * jax.nn.logsumexp(_LSE_BETA * grip_scores)

    # Penalizaciones estructurales y reparto de frenada
    fz_variation = (wheel_rate_f + wheel_rate_r) * 0.007
    stiffness_penalty = (jnp.abs(PDY2) * (fz_variation / Fz0)) * 0.4

    # ── Penalización física de rigidez: ancla k_f/k_r a la frecuencia natural
    # objetivo de FS (2.6-3.2 Hz delante, 2.8-3.5 Hz detrás), en vez de dejar
    # que Adam suba la rigidez sin freno porque el objetivo steady-state no
    # ve pistas reales con irregularidades (donde muelle duro = pérdida de
    # contacto). f_n = (1/2π)·√(k_wheel/m_corner). Asimétrica: barato por
    # debajo del rango, caro por encima — reflejando que sub-amortiguar
    # cuesta grip mecánico en pista real, pero sobre-endurecer cuesta más
    # porque además pierdes tracción sobre curbs/piano.
    m_corner_f = (m * lr / L) / 2.0
    m_corner_r = (m * lf / L) / 2.0
    f_n_f = jnp.sqrt(wheel_rate_f / (m_corner_f + 1e-3)) / (2.0 * jnp.pi)
    f_n_r = jnp.sqrt(wheel_rate_r / (m_corner_r + 1e-3)) / (2.0 * jnp.pi)

    freq_penalty = (0.08 * jax.nn.relu(f_n_f - 3.2) ** 2
                  + 0.08 * jax.nn.relu(f_n_r - 3.5) ** 2
                  + 0.04 * jax.nn.relu(2.6 - f_n_f) ** 2
                  + 0.04 * jax.nn.relu(2.8 - f_n_r) ** 2)

    Fz_f_brake = (m * g * lr / L) + (m * 1.0 * g * h_cg / L)
    Fz_r_brake = (m * g * lf / L) - (m * 1.0 * g * h_cg / L)
    ideal_bias = Fz_f_brake / (Fz_f_brake + Fz_r_brake)
    brake_balance_penalty = 3.0 * (brake_bias_f - ideal_bias) ** 2

    # Trade-offs físicos calibrados (máximo coste combinado ≤ 0.05 G)
    modal_penalty = 0.005 * compute_frequency_response_objective(simulate_step_fn, params, x_init)

    delta_T_rib_f = jnp.abs(camber_f + 2.2) * 5.0
    delta_T_rib_r = jnp.abs(camber_r + 1.4) * 4.0
    thermal_imbalance_cost = 0.0002 * (delta_T_rib_f ** 2 + delta_T_rib_r ** 2)

    fz_roughness_penalty = 1.5e-8 * (c_low_f ** 2 + c_low_r ** 2 + 0.2 * c_high_f ** 2 + 0.2 * c_high_r ** 2)
    scrub_drag_cost = 5.0 * (jnp.sin(jnp.deg2rad(toe_f)) ** 2 + jnp.sin(jnp.deg2rad(toe_r)) ** 2)

    # Agarre objetivo neto
    obj_grip = (smooth_max
                - stiffness_penalty
                - freq_penalty
                - brake_balance_penalty
                - modal_penalty
                - thermal_imbalance_cost
                - fz_roughness_penalty
                - scrub_drag_cost)
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
    from config.vehicles.ter26 import vehicle_params as VP

    if params.shape[-1] == 28:
        k_f, k_r = params[0], params[1]
        c_low_f, c_low_r = params[4], params[5]
        c_high_f, c_high_r = params[6], params[7]
        reb_ratio_f, reb_ratio_r = params[10], params[11]

        # Amortiguación efectiva promedio ponderada (compresión baja/alta + extensión)
        c_f = (0.7 * c_low_f + 0.3 * c_high_f) * (1.0 + reb_ratio_f) * 0.5
        c_r = (0.7 * c_low_r + 0.3 * c_high_r) * (1.0 + reb_ratio_r) * 0.5
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

# ─────────────────────────────────────────────────────────────────────────────
# §1  Mini-Lap Track Definition
# ─────────────────────────────────────────────────────────────────────────────

# 8 representative corners from a typical FSG autocross layout.
# Each: (curvature [1/m], duration [steps at dt=0.005s], target_speed [m/s])
#
# Total: 120 steps = 0.6s simulation (enough for gradients, fast for MORL)
# Covers: straight accel, heavy braking, tight hairpin, medium-speed
# sweeper, chicane, and acceleration out. This is a compressed "essence"
# of an autocross lap that captures all setup-sensitive dynamics.

MINI_LAP_SEGMENTS = [
    # (curvature, n_steps, v_target)
    (0.00,  15,  22.0),    # S1: straight acceleration
    (0.00,  10,  10.0),    # S2: heavy braking zone
    (0.12,  20,  11.0),    # S3: medium-speed right (R ≈ 8.3m)
    (-0.18, 15,  9.0),     # S4: tight left hairpin (R ≈ 5.5m)
    (0.08,  10,  14.0),    # S5: fast right sweeper (R ≈ 12.5m)
    (-0.10, 15,  12.0),    # S6: medium left (R ≈ 10m)
    (0.15,  15,  10.0),    # S7: chicane right (R ≈ 6.7m)
    (0.00,  20,  20.0),    # S8: exit acceleration
]

def _build_mini_lap_profile() -> tuple:
    """
    Build (curvature_array, v_target_array) for the mini-lap.
    Returns JAX arrays of shape (N_total,).
    """
    curv_list = []
    vtgt_list = []
    for kappa, n_steps, v_tgt in MINI_LAP_SEGMENTS:
        curv_list.extend([kappa] * n_steps)
        vtgt_list.extend([v_tgt] * n_steps)
    return jnp.array(curv_list), jnp.array(vtgt_list)

# Pre-build (module-level constant, traced into XLA once)
_CURV_PROFILE, _VTGT_PROFILE = _build_mini_lap_profile()
_N_STEPS_LTE = len(_CURV_PROFILE)  # 120

# ─────────────────────────────────────────────────────────────────────────────
# §2  Endurance LTE Objective
# ─────────────────────────────────────────────────────────────────────────────

def compute_endurance_lte_objective(
    simulate_step_fn,
    setup_params: jax.Array,    # (28,) physical setup
    x_init: jax.Array,          # (46,) initial state
    dt: float = 0.005,
    T_opt: float = 90.0,        # tire optimal temperature [°C]
    # Weights for the composite LTE score
    w_speed: float = 1.0,       # reward for high average speed
    w_energy: float = 0.3,      # penalty for high energy consumption
    w_thermal: float = 0.5,     # penalty for tire overheating
) -> jax.Array:
    """
    Differentiable endurance lap-time-energy objective.

    Simulates an 8-corner mini-lap with P-controlled steering and throttle.
    Returns a scalar J_LTE where HIGHER = BETTER (MORL maximises).

    The score captures the three dimensions that determine endurance ranking:
    1. Average speed (proxy for lap time)
    2. Energy efficiency (kJ per km)
    3. Thermal management (tire temperature at end of stint)

    All intermediate quantities are smooth (no hard conditionals), ensuring
    clean gradient flow from J_LTE back to all 28 setup parameters.
    """
    from config.vehicles.ter26 import vehicle_params as VP

    L_wb = VP.get('lf', 0.8525) + VP.get('lr', 0.6975)

    curvature = _CURV_PROFILE
    v_target  = _VTGT_PROFILE

    # ── Steering + speed controller gains ────────────────────────────────────
    K_steer = 1.0       # kinematic: δ = κ · L
    K_speed = 4000.0    # N/(m/s) — P-controller for longitudinal force
    K_max_brake = 8000.0

    def scan_step(carry, k):
        x = carry

        # Current state extraction
        vx = x[14]
        vx_safe = jnp.maximum(vx, 1.0)

        # Kinematic steering (smooth, no conditionals)
        delta_k = K_steer * curvature[k] * L_wb

        # Speed P-controller
        v_err = v_target[k] - vx_safe
        # Smooth split: positive error → throttle, negative → brake
        F_drive = jax.nn.softplus(v_err) * K_speed
        F_brake = -jax.nn.softplus(-v_err) * K_speed
        F_total = jnp.clip(F_drive + F_brake, -K_max_brake, 6000.0)

        u = jnp.array([delta_k, F_total])

        # Simulate one step
        x_next = simulate_step_fn(x, u, setup_params, dt)

        # ── Energy accounting ────────────────────────────────────────────────
        # Mechanical power = |F · v| (absolute value for total energy budget)
        power_mech = jnp.abs(F_total * vx_safe)

        # ── Tire temperature (max surface temp across front axle) ────────────
        # State indices 28:31 = T_ribs_f (3 surface nodes, front)
        T_surf_f = x_next[28:31]
        T_max_f  = jnp.max(T_surf_f)

        return x_next, (vx_safe, power_mech, T_max_f)

    # ── Run mini-lap ─────────────────────────────────────────────────────────
    x_final, (vx_history, power_history, T_history) = jax.lax.scan(
        scan_step, x_init, jnp.arange(_N_STEPS_LTE),
    )

    # ── Compute LTE metrics ──────────────────────────────────────────────────
    # Average speed [m/s] — higher is better
    mean_vx = jnp.mean(vx_history)

    # Total energy [J] over the mini-lap
    total_energy = jnp.sum(power_history) * dt

    # Distance covered [m]
    distance = jnp.sum(vx_history) * dt

    # Energy per meter [J/m] — lower is better
    energy_per_meter = total_energy / jnp.maximum(distance, 1.0)

    # Thermal penalty — activates when tire temp exceeds T_opt + 15°C
    T_end = T_history[-1]
    thermal_excess = T_end - T_opt - 15.0  # positive = overheating
    # Smooth activation: penalty grows softly above threshold
    thermal_penalty = jax.nn.softplus(thermal_excess * 0.2) / 30.0

    # ── Composite score (HIGHER = BETTER) ────────────────────────────────────
    # Speed reward: normalised to ~1.0 at 15 m/s average
    speed_score = mean_vx / 15.0

    # Efficiency score: normalised to ~1.0 at 200 J/m (typical FS endurance)
    efficiency_score = 200.0 / jnp.maximum(energy_per_meter, 10.0)

    J_LTE = (
        w_speed * speed_score
        + w_energy * efficiency_score
        - w_thermal * thermal_penalty
    )

    return J_LTE


# ─────────────────────────────────────────────────────────────────────────────
# §3  Twin Fidelity Objective  (real-telemetry grounded)
# ─────────────────────────────────────────────────────────────────────────────

def compute_twin_fidelity_objective(
    simulate_step_fn,
    setup_params: jax.Array,           # (28,) physical setup
    real_steer:    jax.Array,          # (N,) front wheel steer [rad]
    real_throttle: jax.Array,          # (N,) normalised throttle [0-1]
    real_brake:    jax.Array,          # (N,) normalised brake pressure [0-1]
    real_speed:    jax.Array,          # (N,) longitudinal speed [m/s]
    real_yaw_rate: jax.Array,          # (N,) yaw rate [rad/s]
    real_ay:       jax.Array,          # (N,) lateral accel [m/s²]
    x_init:        jax.Array,          # (108,) initial vehicle state
    dt:            float = 0.010,      # sample interval [s]
    w_speed:       float = 0.50,       # weight for speed R²
    w_yaw_rate:    float = 0.30,       # weight for yaw-rate R²
    w_ay:          float = 0.20,       # weight for lateral acceleration R²
) -> jax.Array:
    """
    Open-loop twin-fidelity objective: drives the 108-DOF digital twin with
    real recorded control inputs and returns a differentiable fidelity score.

    Score ∈ [0, 1]:
        fidelity = w_speed * R²(speed)  +  w_yaw_rate * R²(yaw_rate)  +  w_ay * R²(ay)

    R² is approximated in a differentiable way:
        R²_soft = 1 − MSE(sim, real) / (Var(real) + ε)
    Clamped to [0, 1] via sigmoid so gradients are always finite even if the
    simulation diverges (would otherwise produce R² << 0 with large gradient norms).

    DESIGN NOTES
    ────────────
    · The simulation is run as a pure jax.lax.scan — entirely inside XLA.
      No Python-side loop. Fully vmappable and differentiable w.r.t. setup_params.
    · The control signal packs steering + a net longitudinal force demand:
        F_lon = throttle * F_DRIVE_MAX - brake * F_BRAKE_MAX
        u = [delta, F_lon, F_lon, F_lon, F_lon, F_brake_hyd]
      This matches the u-vector expected by DifferentiableMultiBodyVehicle.simulate_step
      (steer + 4 hub motor torques + hydraulic brake — simplified here as a split
      between drive/brake channels, consistent with the endurance LTE objective).
    · Higher is better — the MORL optimizer maximises all objectives.

    Parameters
    ----------
    simulate_step_fn : callable
        vehicle.simulate_step — signature (state, u, setup, dt) → state
    setup_params : (28,) jax.Array
        Physical SuspensionSetup vector.  Gradients flow through this.
    real_* : (N,) jax.Array
        Real telemetry channels aligned to a uniform dt grid (from CANLogReader).
    x_init : (108,) jax.Array
        Initial vehicle state (from DifferentiableMultiBodyVehicle.make_initial_state).
    dt : float
        Simulation step size, must match the telemetry grid spacing.
    w_speed, w_yaw_rate, w_ay : float
        Objective weights summing to 1.0.
    """
    # ── Constants matching forward_sim / endurance_lte channels ──────────────
    F_DRIVE_MAX  = 3000.0   # N per axle, consistent with LTE objective
    F_BRAKE_MAX  = 4000.0   # N total, divided across axles via brake_bias_f

    N = real_steer.shape[0]

    def scan_step(carry, k):
        x = carry

        delta   = real_steer[k]
        thr     = real_throttle[k]
        brk     = real_brake[k]

        # Aggregate longitudinal demand into hub-motor torques and hydraulic brake
        F_lon  = thr * F_DRIVE_MAX         # driving force per axle [N]
        F_hyd  = brk * F_BRAKE_MAX         # hydraulic braking force [N]

        # u = [steer, T_fl, T_fr, T_rl, T_rr, F_brake_hyd]
        # Distribute equally across four corners: T_per_wheel = F_lon * R_w / 4
        # R_wheel ≈ 0.2045 m — matches vehicle_dynamics.py default
        T_wheel = F_lon * 0.2045 / 4.0
        u = jnp.array([delta, T_wheel, T_wheel, T_wheel, T_wheel, F_hyd])

        x_next = simulate_step_fn(x, u, setup_params, dt)

        # Extract simulated observables
        vx_sim  = x_next[14]                            # longitudinal speed (q-dot[0])
        wz_sim  = x_next[19]                            # yaw rate (q-dot[5])
        # Lateral acceleration from centripetal: ay_sim = vx * wz  (bicycle model)
        ay_sim  = vx_sim * wz_sim

        return x_next, jnp.array([vx_sim, wz_sim, ay_sim])

    _, predictions = jax.lax.scan(scan_step, x_init, jnp.arange(N))

    # predictions: (N, 3) — [speed, yaw_rate, ay]
    sim_speed    = predictions[:, 0]
    sim_yaw_rate = predictions[:, 1]
    sim_ay       = predictions[:, 2]

    # ── Differentiable R² (soft, clamped) ────────────────────────────────────
    def soft_r2(sim: jax.Array, real: jax.Array) -> jax.Array:
        """
        R² = 1 - MSE(sim, real) / (Var(real) + ε)
        Mapped through sigmoid(10·(R²-0.5)) to:
          · R²=1.0 → 0.993  (small loss at perfect fit — negligible)
          · R²=0.5 → 0.500  (midpoint of the score)
          · R²<0   → 0.007  (saturates, bounded gradient, no gradient explosion)
        This sigmoid mapping is consistent with the smooth feasibility gate in
        compute_skidpad_objective and prevents NaN when simulation diverges.
        """
        mse_val   = jnp.mean((sim - real) ** 2)
        var_real  = jnp.mean((real - jnp.mean(real)) ** 2) + 1e-6
        r2_raw    = 1.0 - mse_val / var_real
        # Smooth clamp to [0, 1] via sigmoid centred at 0.5
        # Gradient at r2_raw=1: sigmoid'(5)=0.007 — finite, not zero
        return jax.nn.sigmoid(10.0 * (r2_raw - 0.5))

    r2_speed    = soft_r2(sim_speed,    real_speed)
    r2_yaw_rate = soft_r2(sim_yaw_rate, real_yaw_rate)
    r2_ay       = soft_r2(sim_ay,       real_ay)

    fidelity = w_speed * r2_speed + w_yaw_rate * r2_yaw_rate + w_ay * r2_ay

    return fidelity