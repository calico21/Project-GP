# powertrain/modes/intermediate/launch_control.py
# Project-GP — Intermediate Launch Sequencer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Positioned between simple (quintic ramp, static μ, no button) and
# advanced (16-knot B-spline, DESC GP, yaw lock PD+I).
#
# Feature delta vs simple:
#   + Button arming (with legacy brake+throttle fallback)
#   + 8-knot Catmull-Rom B-spline profile (optimisable offline)
#   + Per-wheel TC ceiling from Pacejka κ* (Fz-adaptive, same model as tc_intermediate)
#   + Real-time μ EMA from torque feedback
#   + Proportional yaw-lock during LAUNCH (no integral — simpler, still effective)
#   + Abort on hard brake
#
# Feature delta vs advanced:
#   - No GP/DESC signal (no kappa_star from desc_step)
#   - Proportional-only yaw lock (no integral accumulation)
#   - 8 knots (vs 16), optimised separately
#   - No DESC convergence diagnostics
#
# All functions are pure JAX — safe inside jit/grad/vmap/scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple

# Phase constants — public (sanity checks import these)
ILC_IDLE    = 0
ILC_ARMED   = 1
ILC_LAUNCH  = 2
ILC_HANDOFF = 3
ILC_TC      = 4


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateLCParams(NamedTuple):
    """
    Intermediate launch tuning.

    B-spline knots: offline-optimised profile that builds torque in
    0.3s to avoid spinning, holds plateau 0.3–1.2s, then steps back
    to avoid wheelspin as rear wing unloads at speed.
    """
    # ── Arming ────────────────────────────────────────────────────────────
    btn_threshold: float       = 0.50   # launch button threshold [0,1]
    brake_arm_threshold: float = 0.50   # legacy arming: brake fraction
    throttle_gate: float       = 0.90   # minimum throttle fraction for launch
    abort_brake: float         = 0.30   # hard brake → IDLE (abort)
    # ── B-spline profile ─────────────────────────────────────────────────
    n_knots: int               = 8      # number of Catmull-Rom control points
    t_profile_s: float         = 2.0    # total profile duration [s]
    # Default knots: tuned for Ter27 AWD on Hoosier R20
    # [early build, rise, plateau, taper] — units: fraction of T_peak
    # ── Torque ───────────────────────────────────────────────────────────
    T_peak_wheel: float        = 250.0  # [Nm] peak per wheel (conservative vs advanced)
    # Pacejka params for κ* ceiling (must match IntermediateTCParams)
    B0: float                  = 14.0
    B1: float                  = -0.0018
    C_pac: float               = 1.65
    kappa_margin: float        = 0.90   # safety factor below κ*
    r_w: float                 = 0.2032
    # ── μ adaptation ─────────────────────────────────────────────────────
    mu_nom: float              = 1.50   # initial μ assumption
    mu_adapt_alpha: float      = 0.015  # EMA coefficient
    mu_lo: float               = 0.40
    mu_hi: float               = 2.00
    # ── Front-rear split ─────────────────────────────────────────────────
    front_ratio_initial: float = 0.35   # at launch start
    front_ratio_final: float   = 0.28   # at handoff speed
    # ── Yaw lock (proportional only) ─────────────────────────────────────
    Kp_yaw_lock: float         = 150.0  # [Nm/(rad/s)]
    yaw_speed_gate: float      = 2.0    # [m/s] below which yaw lock disabled
    # ── Phase transitions ────────────────────────────────────────────────
    v_handoff: float           = 6.0    # [m/s] LAUNCH → HANDOFF speed gate
    t_max_launch: float        = 3.0    # [s] safety timeout
    dt_blend: float            = 0.25   # [s] Hermite handoff blend
    # ── Nominal Fz (for static ceiling when Fz not wired) ────────────────
    Fz_front_nom: float        = 650.0  # [N] per front wheel
    Fz_rear_nom: float         = 800.0  # [N] per rear wheel


# Public alias for manager
IntermediateLCConfig = IntermediateLCParams

_DEFAULT_ILC_KNOTS = jnp.array([
    0.04, 0.25, 0.65, 0.88,
    0.93, 0.92, 0.90, 0.88,
], dtype=jnp.float32)


# ─────────────────────────────────────────────────────────────────────────────
# §2  State
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateLCState(NamedTuple):
    phase: jax.Array           # scalar int32
    t_phase_start: jax.Array   # scalar float
    t_current: jax.Array       # scalar float
    position_x: jax.Array      # scalar float: distance since launch [m]
    spline_knots: jax.Array    # (8,) B-spline control points (optimisable)
    mu_realtime: jax.Array     # scalar: EMA surface μ

    @classmethod
    def default(cls, params: IntermediateLCParams = None) -> "IntermediateLCState":
        mu0 = params.mu_nom if params is not None else 1.5
        return cls(
            phase=jnp.array(ILC_IDLE, dtype=jnp.int32),
            t_phase_start=jnp.array(0.0),
            t_current=jnp.array(0.0),
            position_x=jnp.array(0.0),
            spline_knots=_DEFAULT_ILC_KNOTS,
            mu_realtime=jnp.array(mu0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3  Output
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateLCOutput(NamedTuple):
    T_command: jax.Array         # (4,) wheel torques [Nm]
    phase: jax.Array             # scalar phase code
    is_launch_active: jax.Array  # scalar float: 1.0 during LAUNCH/HANDOFF
    profile_value: jax.Array     # scalar: B-spline output [0, 1]
    tc_ceiling: jax.Array        # (4,) per-wheel torque ceiling [Nm]
    mu_estimate: jax.Array       # scalar: real-time μ
    yaw_correction: jax.Array    # (4,) differential yaw correction [Nm]
    distance: jax.Array          # scalar: distance [m]
    abort_triggered: jax.Array   # scalar float: 1.0 if abort fired


# ─────────────────────────────────────────────────────────────────────────────
# §4  Catmull-Rom B-Spline (8 knots)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _eval_spline(
    t: jax.Array,
    knots: jax.Array,
    duration: float,
) -> jax.Array:
    """Cubic Hermite (Catmull-Rom) interpolation across 8 knots."""
    n = knots.shape[0]
    t_norm = jnp.clip(t / (duration + 1e-6), 0.0, 1.0) * (n - 1)
    idx = jnp.clip(jnp.floor(t_norm).astype(jnp.int32), 0, n - 2)
    frac = t_norm - idx.astype(jnp.float32)

    p0 = knots[idx]
    p1 = knots[idx + 1]
    m0 = (knots[jnp.minimum(idx + 1, n - 1)] - knots[jnp.maximum(idx - 1, 0)]) * 0.5
    m1 = (knots[jnp.minimum(idx + 2, n - 1)] - knots[idx]) * 0.5

    t2, t3 = frac * frac, frac * frac * frac
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + frac
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return jnp.clip(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Hermite Smoothstep (blend)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _hermite(t: jax.Array, t_start: jax.Array, dt: float) -> jax.Array:
    s = jnp.clip((t - t_start) / (dt + 1e-6), 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


# ─────────────────────────────────────────────────────────────────────────────
# §6  Per-Wheel TC Ceiling from Pacejka κ*
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _tc_ceiling(
    Fz: jax.Array,
    mu_rt: jax.Array,
    params: IntermediateLCParams,
) -> jax.Array:
    """
    T_ceil_i = μ_rt · Fz_i · r_w · κ*/B_eff · kappa_margin

    Simplified: approximate D·Fz ≈ μ·Fz (Pacejka D ≈ μ by definition).
    κ* from analytical Pacejka peak (same formula as tc_intermediate):
      κ*(Fz) = tan(π/(2C)) / B(Fz)
    Ceiling: T_ceil = μ · Fz · r_w (friction limit) × kappa_margin.
    """
    B_eff = jnp.maximum(params.B0 + params.B1 * Fz, 2.0)
    kappa_star = jnp.clip(
        jnp.tan(jnp.pi / (2.0 * params.C_pac)) / B_eff, 0.04, 0.22,
    )
    # At κ*, Fx ≈ D·Fz ≈ μ·Fz (by Pacejka D definition)
    # T_ceil = Fx_max · r_w · kappa_margin
    return mu_rt * Fz * params.r_w * params.kappa_margin


# ─────────────────────────────────────────────────────────────────────────────
# §7  μ EMA Update
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _mu_update(
    mu_prev: jax.Array,
    T_applied: jax.Array,
    Fz: jax.Array,
    params: IntermediateLCParams,
) -> jax.Array:
    Fx = T_applied / (params.r_w + 1e-6)
    mu_meas = jnp.clip(
        jnp.mean(Fx / jnp.maximum(Fz, 100.0)), params.mu_lo, params.mu_hi,
    )
    gate = jax.nn.sigmoid(jnp.mean(jnp.abs(T_applied)) - 15.0)
    alpha_g = params.mu_adapt_alpha * gate
    return jnp.clip((1.0 - alpha_g) * mu_prev + alpha_g * mu_meas, params.mu_lo, params.mu_hi)


# ─────────────────────────────────────────────────────────────────────────────
# §8  Front-Rear Split (speed-adaptive sigmoid)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _front_ratio(vx: jax.Array, params: IntermediateLCParams) -> jax.Array:
    s = jax.nn.sigmoid((vx - 7.0) * 0.5)
    return params.front_ratio_initial * (1.0 - s) + params.front_ratio_final * s


# ─────────────────────────────────────────────────────────────────────────────
# §9  Torque Distribution (Fz-proportional)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _distribute(
    T_total: jax.Array,
    f_ratio: jax.Array,
    Fz: jax.Array,
    T_ceil: jax.Array,
    T_max_hw: jax.Array,
) -> jax.Array:
    T_f = T_total * f_ratio
    T_r = T_total * (1.0 - f_ratio)
    Fz_f = Fz[0] + Fz[1] + 1e-3
    Fz_r = Fz[2] + Fz[3] + 1e-3
    T_raw = jnp.array([
        T_f * Fz[0] / Fz_f,
        T_f * Fz[1] / Fz_f,
        T_r * Fz[2] / Fz_r,
        T_r * Fz[3] / Fz_r,
    ])
    return jnp.minimum(jnp.minimum(T_raw, T_ceil), T_max_hw)


# ─────────────────────────────────────────────────────────────────────────────
# §10  Proportional Yaw Lock
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _yaw_lock(
    T_cmd: jax.Array,
    T_ceil: jax.Array,
    wz: jax.Array,
    vx: jax.Array,
    params: IntermediateLCParams,
) -> Tuple[jax.Array, jax.Array]:
    """
    P-only yaw lock for straight-line stability.
    Proportional only (no integral) — simpler, no windup risk.
    Correction gated below yaw_speed_gate m/s.
    """
    dT = params.Kp_yaw_lock * wz / 4.0
    signs = jnp.array([-1.0, +1.0, -1.0, +1.0])   # [FL, FR, RL, RR]
    gate = jax.nn.sigmoid((vx - params.yaw_speed_gate) * 2.0)
    T_corr = jnp.clip(T_cmd + signs * dT * gate, 0.0, T_ceil)
    correction = T_corr - T_cmd
    return T_corr, correction


# ─────────────────────────────────────────────────────────────────────────────
# §11  Main Step
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def intermediate_launch_step(
    throttle: jax.Array,            # [0, 1]
    brake: jax.Array,               # [0, 1]
    vx: jax.Array,                  # [m/s]
    wz: jax.Array,                  # [rad/s] yaw rate for yaw lock
    Fz: jax.Array,                  # (4,) normal loads [N]
    T_tc: jax.Array,                # (4,) tc_intermediate output for handoff [Nm]
    T_max_hw: jax.Array,            # (4,) hardware ceiling [Nm]
    lc_state: IntermediateLCState,
    dt: jax.Array,
    params: IntermediateLCParams = IntermediateLCParams(),
    launch_button: jax.Array = jnp.array(0.0),  # [0,1] steering wheel button
) -> Tuple[IntermediateLCOutput, "IntermediateLCState"]:

    phase   = lc_state.phase
    t_start = lc_state.t_phase_start
    t       = lc_state.t_current

    # ── Phase transitions ──────────────────────────────────────────────────
    enter_armed = (
        (phase == ILC_IDLE)
        & ((launch_button > params.btn_threshold)
           | ((brake > params.brake_arm_threshold) & (throttle > params.throttle_gate)))
    )
    phase   = jnp.where(enter_armed, ILC_ARMED, phase)
    t_start = jnp.where(enter_armed, t, t_start)

    # Button-release with WOT OR brake release
    enter_launch = (
        (phase == ILC_ARMED)
        & (((launch_button < params.btn_threshold) & (throttle > params.throttle_gate))
           | (brake < 0.3))
    )
    phase   = jnp.where(enter_launch, ILC_LAUNCH, phase)
    t_start = jnp.where(enter_launch, t, t_start)

    t_elapsed = t - t_start

    enter_handoff = (
        (phase == ILC_LAUNCH)
        & ((vx > params.v_handoff) | (t_elapsed > params.t_max_launch))
    )
    phase   = jnp.where(enter_handoff, ILC_HANDOFF, phase)
    t_start = jnp.where(enter_handoff, t, t_start)
    t_elapsed = t - t_start

    enter_tc = (phase == ILC_HANDOFF) & (t_elapsed > params.dt_blend)
    phase    = jnp.where(enter_tc, ILC_TC, phase)

    # Abort: hard brake during active launch
    abort = ((phase >= ILC_LAUNCH) & (phase <= ILC_HANDOFF)
             & (brake > params.abort_brake))
    abort_f = abort.astype(jnp.float32)
    phase   = jnp.where(abort, jnp.array(ILC_IDLE, dtype=jnp.int32), phase)

    # ── Torque computation ─────────────────────────────────────────────────
    t_launch_elapsed = jnp.maximum(t - t_start, 0.0)

    # μ EMA update from last commanded torques
    mu_rt = _mu_update(lc_state.mu_realtime, T_tc, Fz, params)

    # Per-wheel ceiling
    T_ceil = _tc_ceiling(Fz, mu_rt, params)

    # B-spline profile
    mu_scale = jnp.clip(mu_rt / 1.5, 0.5, 1.2)
    profile_val = _eval_spline(t_launch_elapsed, lc_state.spline_knots, params.t_profile_s)
    T_total = params.T_peak_wheel * 4.0 * profile_val * mu_scale

    f_ratio  = _front_ratio(vx, params)
    T_launch = _distribute(T_total, f_ratio, Fz, T_ceil, T_max_hw)

    # Yaw lock
    is_active = ((phase == ILC_LAUNCH) | (phase == ILC_HANDOFF)).astype(jnp.float32)
    T_yaw, yaw_corr = _yaw_lock(T_launch, T_ceil, wz, vx, params)
    T_launch_final = is_active * T_yaw + (1.0 - is_active) * T_launch

    # Handoff blend
    w = _hermite(t, t_start, params.dt_blend)
    T_handoff = (1.0 - w) * T_launch_final + w * T_tc

    # Mode select
    T_idle = jnp.zeros(4)
    T_out = jnp.where(
        phase <= ILC_ARMED, T_idle,
        jnp.where(phase == ILC_LAUNCH, T_launch_final,
                  jnp.where(phase == ILC_HANDOFF, T_handoff, T_tc)),
    )

    pos_new = lc_state.position_x + vx * dt

    output = IntermediateLCOutput(
        T_command=T_out,
        phase=phase,
        is_launch_active=is_active,
        profile_value=profile_val,
        tc_ceiling=T_ceil,
        mu_estimate=mu_rt,
        yaw_correction=yaw_corr,
        distance=pos_new,
        abort_triggered=abort_f,
    )
    new_state = IntermediateLCState(
        phase=phase,
        t_phase_start=t_start,
        t_current=t + dt,
        position_x=pos_new,
        spline_knots=lc_state.spline_knots,
        mu_realtime=mu_rt,
    )
    return output, new_state