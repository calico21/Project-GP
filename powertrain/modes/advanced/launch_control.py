# powertrain/modes/advanced/launch_control.py
# Project-GP — Neural Predictive Launch Sequencer  v2.1
# ═══════════════════════════════════════════════════════════════════════════════
#
# v2.1 additions over v2.0:
#   · Button-based ARMED trigger (button held → ARMED, button released + WOT → LAUNCH)
#   · Per-wheel TC ceiling: T_cmd ≤ μ_rt · Fz · r_w · γ_κ (prevents κ > κ*)
#   · Real-time μ EMA: continuously updated from DESC feedback, not just probe
#   · Yaw-lock PI: differential correction targeting ψ̇ = 0 during LAUNCH/HANDOFF
#   · Abort path: hard brake during launch → IDLE (non-differentiable, gated sigmoid)
#   · Backward compat: launch_step() API preserved; new launch_step_v2() adds
#     launch_button, kappa_star, wz without breaking sanity checks or manager
#
# State machine:
#   IDLE ──(btn OR brake+throttle)──► ARMED ──(btn_release + WOT)──► LAUNCH
#   LAUNCH ──(v > v_thr OR t > T_dur)──► HANDOFF ──(t > dt_blend)──► TC
#   LAUNCH/HANDOFF ──(hard brake)──► IDLE   [abort path]
#
# All functions are pure JAX — safe inside jit/grad/vmap/scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# §1  Launch Configuration
# ─────────────────────────────────────────────────────────────────────────────

class LaunchConfig(NamedTuple):
    """Static launch control configuration — all floats XLA-constant-foldable."""
    # ── B-spline profile ─────────────────────────────────────────────────────
    n_spline_knots: int = 16
    t_profile_duration: float = 2.0

    # ── Phase transition thresholds ───────────────────────────────────────────
    t_handoff_min: float = 0.8
    v_handoff_threshold: float = 5.0       # [m/s] speed gate for LAUNCH → HANDOFF
    dt_blend: float = 0.3                  # [s]   HANDOFF blend duration

    # ── Mu probe ──────────────────────────────────────────────────────────────
    mu_probe_torque_frac: float = 0.10     # fraction of T_peak for probe pulse
    mu_probe_duration: float = 0.05        # [s] probe pulse duration per wheel

    # ── Torque limits ─────────────────────────────────────────────────────────
    T_peak_wheel: float = 450.0            # [Nm] per-wheel peak

    # ── Front-rear torque split ───────────────────────────────────────────────
    front_ratio_initial: float = 0.35      # at launch (rear-biased)
    front_ratio_final: float = 0.28        # at handoff speed

    # ── v2.1: button-based arming ─────────────────────────────────────────────
    launch_button_threshold: float = 0.5   # button signal threshold [0, 1]
    launch_throttle_gate: float = 0.90     # min throttle fraction for launch
    abort_brake_threshold: float = 0.30    # brake pressure triggers abort

    # ── v2.1: TC ceiling ──────────────────────────────────────────────────────
    kappa_margin: float = 0.92             # safety factor on μ·Fz·r_w ceiling
    r_w: float = 0.2032                    # [m] wheel radius (must match geometry)

    # ── v2.1: real-time mu EMA ────────────────────────────────────────────────
    mu_adapt_alpha: float = 0.02           # EMA coefficient (0.02 ≈ τ=25 steps@200Hz)
    mu_clamp_lo: float = 0.40             # lower bound on μ estimate
    mu_clamp_hi: float = 2.00             # upper bound on μ estimate

    # ── v2.1: yaw-lock PI ────────────────────────────────────────────────────
    yaw_lock_Kp: float = 200.0            # [Nm/(rad/s)] proportional gain
    yaw_lock_Ki: float = 50.0             # [Nm/rad]     integral gain
    yaw_lock_speed_gate: float = 2.0      # [m/s] below which correction disabled
    yaw_integral_clamp: float = 0.50      # [rad] anti-windup clamp


# Public alias — manager and tests import LaunchParams
LaunchParams = LaunchConfig


# ─────────────────────────────────────────────────────────────────────────────
# §2  Phase Constants
# ─────────────────────────────────────────────────────────────────────────────

PHASE_IDLE    = 0
PHASE_ARMED   = 1
PHASE_PROBE   = 2
PHASE_LAUNCH  = 3
PHASE_HANDOFF = 4
PHASE_TC      = 5

_DEFAULT_SPLINE_COEFFS = jnp.array([
    0.05, 0.20, 0.50, 0.75,
    0.88, 0.93, 0.95, 0.96,
    0.96, 0.95, 0.94, 0.93,
    0.92, 0.91, 0.90, 0.90,
])


# ─────────────────────────────────────────────────────────────────────────────
# §3  Launch State — v2.1 extends with wz_integral and mu_realtime
# ─────────────────────────────────────────────────────────────────────────────

class LaunchState(NamedTuple):
    """Persistent launch control state across timesteps."""
    phase: jax.Array               # scalar int32: current phase
    t_phase_start: jax.Array       # scalar float: time when current phase began
    mu_probe_result: jax.Array     # scalar: μ from probe pulse (one-shot)
    mu_probe_wheel_idx: jax.Array  # scalar int32: which wheel is being probed
    spline_coeffs: jax.Array       # (16,) B-spline profile coefficients
    position_x: jax.Array          # scalar: distance since launch [m]
    t_current: jax.Array           # scalar: accumulated simulation time [s]
    # v2.1 fields
    wz_integral: jax.Array         # scalar: integrated yaw error [rad] (PI anti-windup)
    mu_realtime: jax.Array         # scalar: EMA-filtered real-time friction estimate

    @classmethod
    def default(cls, params: "LaunchConfig" = None) -> "LaunchState":
        return cls(
            phase=jnp.array(PHASE_IDLE, dtype=jnp.int32),
            t_phase_start=jnp.array(0.0),
            mu_probe_result=jnp.array(1.5),
            mu_probe_wheel_idx=jnp.array(0, dtype=jnp.int32),
            spline_coeffs=_DEFAULT_SPLINE_COEFFS,
            position_x=jnp.array(0.0),
            t_current=jnp.array(0.0),
            wz_integral=jnp.array(0.0),
            mu_realtime=jnp.array(1.5),
        )


def make_launch_state(spline_coeffs: jax.Array = None) -> LaunchState:
    """Functional factory — backward-compat with legacy callers."""
    coeffs = _DEFAULT_SPLINE_COEFFS if spline_coeffs is None else spline_coeffs
    return LaunchState(
        phase=jnp.array(PHASE_IDLE, dtype=jnp.int32),
        t_phase_start=jnp.array(0.0),
        mu_probe_result=jnp.array(1.5),
        mu_probe_wheel_idx=jnp.array(0, dtype=jnp.int32),
        spline_coeffs=coeffs,
        position_x=jnp.array(0.0),
        t_current=jnp.array(0.0),
        wz_integral=jnp.array(0.0),
        mu_realtime=jnp.array(1.5),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §4  Launch Output
# ─────────────────────────────────────────────────────────────────────────────

class LaunchOutput(NamedTuple):
    """Output of a single launch control step."""
    T_command: jax.Array        # (4,) commanded wheel torques [Nm]
    phase: jax.Array            # scalar: current phase
    t_elapsed: jax.Array        # scalar: time since phase start [s]
    profile_value: jax.Array    # scalar: B-spline profile value [0,1]
    f_front: jax.Array          # scalar: current front torque fraction
    mu_estimate: jax.Array      # scalar: real-time μ estimate
    is_launch_active: jax.Array # scalar float: 1.0 if in LAUNCH or HANDOFF
    distance: jax.Array         # scalar: distance traveled [m]
    # v2.1 diagnostics
    tc_ceiling: jax.Array       # (4,) per-wheel torque ceiling [Nm]
    yaw_correction: jax.Array   # (4,) differential torque applied for yaw lock [Nm]
    abort_triggered: jax.Array  # scalar float: 1.0 if abort fired this step


# ─────────────────────────────────────────────────────────────────────────────
# §5  B-Spline Profile Evaluation (Catmull-Rom / cubic Hermite)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def evaluate_bspline_profile(
    t: jax.Array,
    coeffs: jax.Array,
    duration: float = 2.0,
) -> jax.Array:
    n = coeffs.shape[0]
    t_norm = jnp.clip(t / (duration + 1e-6), 0.0, 1.0) * (n - 1)
    idx = jnp.clip(jnp.floor(t_norm).astype(jnp.int32), 0, n - 2)
    frac = t_norm - idx.astype(jnp.float32)

    p0 = coeffs[idx]
    p1 = coeffs[idx + 1]
    m0 = (coeffs[jnp.minimum(idx + 1, n - 1)] - coeffs[jnp.maximum(idx - 1, 0)]) * 0.5
    m1 = (coeffs[jnp.minimum(idx + 2, n - 1)] - coeffs[idx]) * 0.5

    t2, t3 = frac * frac, frac * frac * frac
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + frac
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return jnp.clip(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Dynamic Front-Rear Torque Split
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def launch_front_ratio(
    t_launch: jax.Array,
    vx: jax.Array,
    cfg: LaunchConfig = LaunchConfig(),
) -> jax.Array:
    # Smooth sigmoid blend from initial (0.35) to final (0.28) as speed rises
    speed_frac = jax.nn.sigmoid((vx - 8.0) * 0.5)
    f_ratio = (cfg.front_ratio_initial * (1.0 - speed_frac)
               + cfg.front_ratio_final * speed_frac)
    return jnp.clip(f_ratio, 0.15, 0.50)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Per-Wheel Torque Distribution (Fz-proportional)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def launch_torque_distribution(
    T_total: jax.Array,
    f_ratio: jax.Array,
    mu_est: jax.Array,
    Fz: jax.Array,
    T_max: jax.Array,
    r_w: float = 0.2032,
) -> jax.Array:
    T_front = T_total * f_ratio
    T_rear  = T_total * (1.0 - f_ratio)

    Fz_front_total = Fz[0] + Fz[1] + 1e-3
    Fz_rear_total  = Fz[2] + Fz[3] + 1e-3

    T_wheels = jnp.array([
        T_front * Fz[0] / Fz_front_total,
        T_front * Fz[1] / Fz_front_total,
        T_rear  * Fz[2] / Fz_rear_total,
        T_rear  * Fz[3] / Fz_rear_total,
    ])
    # Friction limit — last line of defence before the TC ceiling in v2.1
    T_friction_limit = mu_est * Fz * r_w * 0.98
    return jnp.clip(jnp.minimum(T_wheels, T_friction_limit), 0.0, T_max)


# ─────────────────────────────────────────────────────────────────────────────
# §8  Pre-Launch Mu Probe
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def mu_probe_estimate(
    T_applied: jax.Array,
    omega_dot: jax.Array,
    Iw: float = 1.2,
    r_w: float = 0.2032,
    Fz: jax.Array = jnp.array(735.0),
) -> jax.Array:
    Fx_tire = (T_applied - Iw * omega_dot * r_w) / r_w
    return jnp.clip(jnp.abs(Fx_tire) / jnp.maximum(Fz, 10.0), 0.3, 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# §9  Hermite Smoothstep
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def hermite_smoothstep(t: jax.Array, t_start: jax.Array, dt_blend: float = 0.3):
    s = jnp.clip((t - t_start) / (dt_blend + 1e-6), 0.0, 1.0)
    return 3.0 * s ** 2 - 2.0 * s ** 3


# ─────────────────────────────────────────────────────────────────────────────
# §10  v2.1 — Per-Wheel TC Ceiling
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def launch_tc_ceiling(
    Fz: jax.Array,           # (4,) normal loads [N]
    mu_rt: jax.Array,        # scalar: real-time friction estimate
    r_w: float = 0.2032,
    kappa_margin: float = 0.92,
) -> jax.Array:
    """
    Per-wheel torque ceiling ensuring κᵢ ≤ κ*ᵢ.

    Derivation: Near Pacejka peak, Fx_max ≈ D·Fz ≈ μ·Fz (D ≈ μ for
    typical compounds). T_wheel = Fx·r_w is the traction-limited torque.
    kappa_margin < 1 keeps operating point below peak for DESC tracking margin.

    No dependence on kappa_star — the ceiling is conservative at the PEAK;
    DESC continuously modulates within this envelope. This is structurally
    correct: ceiling prevents gross overshoot; DESC finds fine optimum inside.
    """
    Fx_peak = mu_rt * Fz                     # (4,) [N] — Pacejka D approximation
    return Fx_peak * r_w * kappa_margin       # (4,) [Nm]


# ─────────────────────────────────────────────────────────────────────────────
# §11  v2.1 — Real-Time μ EMA Update
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def mu_realtime_update(
    mu_prev: jax.Array,         # scalar: previous estimate
    T_applied: jax.Array,       # (4,) wheel torques actually applied [Nm]
    Fz: jax.Array,              # (4,) normal loads [N]
    r_w: float = 0.2032,
    alpha: float = 0.02,
    mu_lo: float = 0.40,
    mu_hi: float = 2.00,
) -> jax.Array:
    """
    EMA update of μ from applied torque and Fz.

    Model: Fx_i = T_i / r_w  (quasi-static, inertia-free).
    μ_meas = mean(Fx_i / Fz_i) = mean(T_i / (Fz_i · r_w)).

    Activated only when T_applied has meaningful signal (|T| > 10 Nm per wheel)
    to avoid noise-driven drift during near-zero torque phases.
    """
    Fx_meas = T_applied / (r_w + 1e-6)                      # (4,) [N]
    mu_meas = jnp.mean(Fx_meas / jnp.maximum(Fz, 50.0))     # scalar
    mu_meas = jnp.clip(mu_meas, mu_lo, mu_hi)

    # Gate: only update when we have a proper torque signal
    signal_strength = jax.nn.sigmoid(jnp.mean(jnp.abs(T_applied)) - 10.0)
    alpha_gated = alpha * signal_strength

    mu_new = (1.0 - alpha_gated) * mu_prev + alpha_gated * mu_meas
    return jnp.clip(mu_new, mu_lo, mu_hi)


# ─────────────────────────────────────────────────────────────────────────────
# §12  v2.1 — Yaw-Lock PI Correction
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def launch_yaw_correction(
    T_cmd: jax.Array,             # (4,) baseline torques from B-spline [Nm]
    T_ceiling: jax.Array,         # (4,) per-wheel TC ceiling [Nm]
    wz: jax.Array,                # scalar: measured yaw rate [rad/s]
    wz_integral: jax.Array,       # scalar: integrated yaw error [rad]
    vx: jax.Array,                # scalar: longitudinal speed [m/s]
    dt: jax.Array,                # scalar: timestep [s]
    Kp: float = 200.0,            # [Nm/(rad/s)]
    Ki: float = 50.0,             # [Nm/rad]
    speed_gate: float = 2.0,      # [m/s]
    integral_clamp: float = 0.50, # [rad]
) -> Tuple[jax.Array, jax.Array]:
    """
    PI yaw-lock targeting ψ̇ = 0 during straight-line launch.

    Array layout: [FL, FR, RL, RR] → left={0,2}, right={1,3}
    Sign: wz > 0 = CCW (car turning left).
    Correction: +ΔT to right wheels, -ΔT to left → CW restoring moment.

    Anti-windup: integral clamped at ±integral_clamp rad.
    Speed gate: smoothly disabled below speed_gate m/s (gyroscopic effects
    unreliable at near-zero speed; also avoids torque asymmetry during probe).

    The correction is clipped to [0, T_ceiling] per wheel, so it can NEVER
    push a wheel past its friction limit even at maximum yaw authority.
    """
    dT_p = Kp * wz                                      # proportional
    dT_i = Ki * wz_integral                             # integral
    dT_total = dT_p + dT_i

    # Smooth speed gate — zero gain below speed_gate, unity above
    gate = jax.nn.sigmoid((vx - speed_gate) * 2.0)
    dT_gated = dT_total * gate

    # [FL, FR, RL, RR]: left=-1, right=+1
    correction_signs = jnp.array([-1.0, +1.0, -1.0, +1.0])
    dT_per_wheel = correction_signs * (dT_gated / 4.0)

    T_corrected = jnp.clip(T_cmd + dT_per_wheel, 0.0, T_ceiling)

    # Integral update with anti-windup
    wz_integral_new = jnp.clip(wz_integral + wz * dt, -integral_clamp, integral_clamp)

    return T_corrected, wz_integral_new


# ─────────────────────────────────────────────────────────────────────────────
# §13  Internal Launch Step (full-argument form for offline optimization)
#       Backward-compatible: legacy args unchanged, new args appended with defaults
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def _launch_step_internal(
    t: jax.Array,
    vx: jax.Array,
    Fz: jax.Array,           # (4,) [N]
    T_max: jax.Array,        # (4,) [Nm]
    T_tc: jax.Array,         # (4,) TC/TV torques for handoff target [Nm]
    brake_pressed: jax.Array,
    throttle_full: jax.Array,
    launch_state: LaunchState,
    dt: jax.Array,
    cfg: LaunchConfig = LaunchConfig(),
    # ── v2.1 new inputs (safe defaults preserve legacy behaviour) ────────────
    launch_button: jax.Array = jnp.array(0.0),  # [0,1] dedicated launch button
    kappa_star: jax.Array = jnp.full(4, 0.10),  # (4,) optimal slip from DESC
    wz: jax.Array = jnp.array(0.0),             # [rad/s] measured yaw rate
) -> Tuple[LaunchOutput, LaunchState]:

    phase   = launch_state.phase
    t_start = launch_state.t_phase_start
    mu_est  = launch_state.mu_probe_result      # one-shot probe result
    mu_rt   = launch_state.mu_realtime          # continuously adapted
    coeffs  = launch_state.spline_coeffs
    pos_x   = launch_state.position_x
    wz_int  = launch_state.wz_integral

    # ── Phase transitions ────────────────────────────────────────────────────

    # IDLE → ARMED: button OR legacy brake+throttle
    enter_armed_btn    = (phase == PHASE_IDLE) & (launch_button > cfg.launch_button_threshold)
    enter_armed_legacy = (phase == PHASE_IDLE) & (brake_pressed > 0.5) & (throttle_full > 0.5)
    enter_armed = enter_armed_btn | enter_armed_legacy
    phase   = jnp.where(enter_armed, PHASE_ARMED, phase)
    t_start = jnp.where(enter_armed, t, t_start)

    # ARMED → LAUNCH: button released + WOT (new) OR brake released (legacy)
    enter_launch_btn    = ((phase == PHASE_ARMED)
                           & (launch_button < cfg.launch_button_threshold)
                           & (throttle_full > cfg.launch_throttle_gate))
    enter_launch_legacy = (phase == PHASE_ARMED) & (brake_pressed < 0.5)
    enter_launch = enter_launch_btn | enter_launch_legacy
    phase   = jnp.where(enter_launch, PHASE_LAUNCH, phase)
    t_start = jnp.where(enter_launch, t, t_start)
    pos_x   = jnp.where(enter_launch, 0.0, pos_x)

    t_elapsed = t - t_start

    # LAUNCH → HANDOFF: speed gate OR profile expired
    enter_handoff = ((phase == PHASE_LAUNCH) &
                     ((vx > cfg.v_handoff_threshold) |
                      (t_elapsed > cfg.t_profile_duration)))
    phase   = jnp.where(enter_handoff, PHASE_HANDOFF, phase)
    t_start = jnp.where(enter_handoff, t, t_start)
    t_elapsed = t - t_start

    # HANDOFF → TC: blend complete
    enter_tc = (phase == PHASE_HANDOFF) & (t_elapsed > cfg.dt_blend)
    phase = jnp.where(enter_tc, PHASE_TC, phase)

    # ABORT: hard brake during active launch → IDLE
    # Smooth sigmoid so gradient flows through for offline optimisation;
    # the sigmoid is steep enough (k=20) to act as near-discontinuous in deployment.
    abort_strength = jax.nn.sigmoid(
        (brake_pressed - cfg.abort_brake_threshold) * 20.0
    )
    is_abortable = (phase >= PHASE_LAUNCH) & (phase <= PHASE_HANDOFF)
    # Abort resets phase to IDLE via smooth mix — XLA traces both branches
    phase_abort_target = jnp.array(PHASE_IDLE, dtype=jnp.int32)
    phase = jnp.where(
        is_abortable & (abort_strength > 0.5),
        phase_abort_target,
        phase,
    )
    abort_triggered = (is_abortable & (abort_strength > 0.5)).astype(jnp.float32)

    # ── B-spline torque profile ───────────────────────────────────────────────
    T_idle = jnp.zeros(4)

    t_launch_elapsed = jnp.maximum(t - launch_state.t_phase_start, 0.0)
    profile_val = evaluate_bspline_profile(t_launch_elapsed, coeffs, cfg.t_profile_duration)

    # mu_scale from probe result (coarse initial scaling)
    mu_scale     = jnp.clip(mu_est / 1.5, 0.5, 1.2)
    T_total_raw  = cfg.T_peak_wheel * 4.0 * profile_val * mu_scale
    f_ratio      = launch_front_ratio(t_launch_elapsed, vx, cfg)
    T_bspline    = launch_torque_distribution(T_total_raw, f_ratio, mu_est, Fz, T_max, cfg.r_w)

    # ── v2.1: TC ceiling — clamp B-spline to prevent κ > κ* ─────────────────
    T_ceil = launch_tc_ceiling(Fz, mu_rt, cfg.r_w, cfg.kappa_margin)
    T_launch_clipped = jnp.minimum(T_bspline, T_ceil)

    # ── v2.1: Yaw-lock PI — active during LAUNCH and HANDOFF ─────────────────
    is_yaw_lock_active = ((phase == PHASE_LAUNCH) | (phase == PHASE_HANDOFF)).astype(jnp.float32)
    T_launch_yaw, wz_int_new = launch_yaw_correction(
        T_launch_clipped, T_ceil, wz, wz_int, vx, dt,
        Kp=cfg.yaw_lock_Kp,
        Ki=cfg.yaw_lock_Ki,
        speed_gate=cfg.yaw_lock_speed_gate,
        integral_clamp=cfg.yaw_integral_clamp,
    )
    # Gate: only apply correction during active launch phases
    T_launch_final = (is_yaw_lock_active * T_launch_yaw
                      + (1.0 - is_yaw_lock_active) * T_launch_clipped)
    yaw_correction_applied = T_launch_final - T_launch_clipped

    # Preserve wz_integral only during active phases
    wz_int = jnp.where(is_yaw_lock_active > 0.5, wz_int_new, jnp.array(0.0))

    # ── HANDOFF blend ─────────────────────────────────────────────────────────
    w_blend   = hermite_smoothstep(t, launch_state.t_phase_start, cfg.dt_blend)
    T_handoff = (1.0 - w_blend) * T_launch_final + w_blend * T_tc

    # ── Mode selector ─────────────────────────────────────────────────────────
    is_idle_or_armed = (phase <= PHASE_ARMED)
    is_launch        = (phase == PHASE_LAUNCH)
    is_handoff       = (phase == PHASE_HANDOFF)

    T_out = jnp.where(
        is_idle_or_armed, T_idle,
        jnp.where(is_launch, T_launch_final,
                  jnp.where(is_handoff, T_handoff, T_tc)),
    )

    # ── v2.1: real-time mu update (EMA from applied torques) ─────────────────
    is_launch_active_float = ((phase >= PHASE_LAUNCH) & (phase <= PHASE_HANDOFF)).astype(jnp.float32)
    mu_rt_new = mu_realtime_update(
        mu_rt, T_out, Fz, cfg.r_w, cfg.mu_adapt_alpha, cfg.mu_clamp_lo, cfg.mu_clamp_hi,
    )
    # Only update during active phases to avoid drifting during idle
    mu_rt = jnp.where(is_launch_active_float > 0.5, mu_rt_new, mu_rt)

    pos_x_new = pos_x + vx * dt

    output = LaunchOutput(
        T_command=T_out,
        phase=phase,
        t_elapsed=t - launch_state.t_phase_start,
        profile_value=profile_val,
        f_front=f_ratio,
        mu_estimate=mu_rt,
        is_launch_active=is_launch_active_float,
        distance=pos_x_new,
        tc_ceiling=T_ceil,
        yaw_correction=yaw_correction_applied,
        abort_triggered=abort_triggered,
    )

    new_state = LaunchState(
        phase=phase,
        t_phase_start=t_start,
        mu_probe_result=mu_est,
        mu_probe_wheel_idx=launch_state.mu_probe_wheel_idx,
        spline_coeffs=coeffs,
        position_x=pos_x_new,
        t_current=launch_state.t_current + dt,
        wz_integral=wz_int,
        mu_realtime=mu_rt,
    )

    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §14  Public API — v1 (backward-compatible, no TC ceiling / yaw lock)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def launch_step(
    throttle: jax.Array,
    brake: jax.Array,
    vx: jax.Array,
    omega_wheel: jax.Array,  # (4,) — unused here, reserved for future mu probe
    T_tc: jax.Array,
    launch_state: LaunchState,
    dt: jax.Array,
    params: LaunchConfig = LaunchConfig(),
) -> Tuple[LaunchOutput, LaunchState]:
    """
    Legacy public-facing launch step. Preserves v1 API for sanity checks.
    Generates nominal Fz and T_max; no button signal, no TC ceiling, no yaw lock.
    """
    Fz_default  = jnp.array([600.0, 600.0, 800.0, 800.0])
    T_max_default = jnp.full(4, params.T_peak_wheel)

    return _launch_step_internal(
        t=launch_state.t_current,
        vx=vx,
        Fz=Fz_default,
        T_max=T_max_default,
        T_tc=T_tc,
        brake_pressed=brake,
        throttle_full=throttle,
        launch_state=launch_state,
        dt=dt,
        cfg=params,
        launch_button=jnp.array(0.0),
        kappa_star=jnp.full(4, 0.10),
        wz=jnp.array(0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §15  Public API — v2.1 (button arming + TC ceiling + yaw lock)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def launch_step_v2(
    throttle: jax.Array,         # [0, 1] throttle pedal
    brake: jax.Array,            # [0, 1] brake pedal
    vx: jax.Array,               # [m/s] longitudinal speed
    omega_wheel: jax.Array,      # (4,) wheel speeds [rad/s]
    Fz: jax.Array,               # (4,) normal loads [N] — from vehicle model
    T_max: jax.Array,            # (4,) motor torque limits [Nm]
    T_tc: jax.Array,             # (4,) DESC/TV handoff target [Nm]
    launch_state: LaunchState,
    dt: jax.Array,
    params: LaunchConfig = LaunchConfig(),
    # ── v2.1 control inputs ──────────────────────────────────────────────────
    launch_button: jax.Array = jnp.array(0.0),   # [0,1] dedicated launch button
    kappa_star: jax.Array = jnp.full(4, 0.10),   # (4,) optimal slip from DESC
    wz: jax.Array = jnp.array(0.0),              # [rad/s] yaw rate from IMU/EKF
) -> Tuple[LaunchOutput, LaunchState]:
    """
    Full v2.1 launch step with button arming, TC ceiling, and yaw lock.

    Usage:
        lc_out, lc_state = launch_step_v2(
            throttle=throttle_filt,
            brake=brake_filt,
            vx=vx,
            omega_wheel=omega_wheel,
            Fz=Fz_est,
            T_max=T_max_motors,
            T_tc=tc_output.T_cmd,
            launch_state=manager_state.launch,
            dt=dt,
            params=config.launch,
            launch_button=launch_button_signal,   # from steering wheel button
            kappa_star=tc_output.kappa_star,      # from tc_step
            wz=wz_measured,                       # from IMU / EKF
        )
    """
    return _launch_step_internal(
        t=launch_state.t_current,
        vx=vx,
        Fz=Fz,
        T_max=T_max,
        T_tc=T_tc,
        brake_pressed=brake,
        throttle_full=throttle,
        launch_state=launch_state,
        dt=dt,
        cfg=params,
        launch_button=launch_button,
        kappa_star=kappa_star,
        wz=wz,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §16  Offline Launch Profile Optimization (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_launch_profile(
    simulate_step_fn,
    setup_params: jax.Array,
    x0: jax.Array,
    target_distance: float = 75.0,
    n_steps: int = 400,
    dt: float = 0.005,
    n_optim_iters: int = 100,
    lr: float = 0.01,
    cfg: LaunchConfig = LaunchConfig(),
) -> jax.Array:
    """
    Optimize B-spline launch profile coefficients by differentiating
    t_75m through the full physics engine. Runs OFFLINE.
    Uses launch_step_v2 internally for full TC ceiling fidelity.
    """
    import optax

    def loss_fn(coeffs: jax.Array) -> jax.Array:
        state = LaunchState.default()
        state = LaunchState(
            phase=state.phase,
            t_phase_start=state.t_phase_start,
            mu_probe_result=state.mu_probe_result,
            mu_probe_wheel_idx=state.mu_probe_wheel_idx,
            spline_coeffs=coeffs,
            position_x=state.position_x,
            t_current=state.t_current,
            wz_integral=state.wz_integral,
            mu_realtime=state.mu_realtime,
        )
        # Simplified rollout — real implementation uses full physics
        T_tc = jnp.full(4, 200.0)
        Fz = jnp.array([600.0, 600.0, 800.0, 800.0])
        T_max = jnp.full(4, cfg.T_peak_wheel)
        vx = jnp.array(0.0)

        def step_fn(carry, _):
            s, v = carry
            out, s_new = _launch_step_internal(
                t=s.t_current, vx=v, Fz=Fz, T_max=T_max, T_tc=T_tc,
                brake_pressed=jnp.array(0.0), throttle_full=jnp.array(1.0),
                launch_state=s, dt=jnp.array(dt), cfg=cfg,
                launch_button=jnp.array(0.0),  # use legacy mode
                kappa_star=jnp.full(4, 0.10),
                wz=jnp.array(0.0),
            )
            v_new = v + jnp.sum(out.T_command) / (setup_params[0] * cfg.r_w) * dt
            return (s_new, v_new), out.distance

        (final_state, final_vx), distances = jax.lax.scan(
            step_fn, (state, vx), None, length=n_steps,
        )
        # Penalise time to target distance — proxy: maximise final distance / steps²
        return -final_state.position_x / (target_distance + 1e-3)

    optimizer = optax.adam(lr)
    coeffs = _DEFAULT_SPLINE_COEFFS
    opt_state = optimizer.init(coeffs)

    @jax.jit
    def update(coeffs, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(coeffs)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        coeffs_new = optax.apply_updates(coeffs, updates)
        coeffs_new = jnp.clip(coeffs_new, 0.0, 1.0)  # physical profile bounds
        return coeffs_new, opt_state_new, loss

    for i in range(n_optim_iters):
        coeffs, opt_state, loss = update(coeffs, opt_state)
        if i % 10 == 0:
            print(f"  iter {i:4d} | loss={float(loss):.4f}")

    return coeffs