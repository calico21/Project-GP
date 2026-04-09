# powertrain/modes/simple/launch_control.py
# Project-GP — Simple Launch Sequencer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Design philosophy: dead-reliable, zero exotic dependencies, auditable.
# Three phases only — ARMED, LAUNCH, TC — with Hermite-smooth transitions.
#
# Launch sequence (simple mode):
#   IDLE  ──(brake > 0.5 AND throttle > 0.5)──► ARMED
#   ARMED ──(brake < 0.3 AND throttle > 0.9)──► LAUNCH
#   LAUNCH──(vx > v_thr OR t > t_dur)──────────► TC
#
# Torque profile: quintic polynomial ramp (C² continuous, zero jerk at endpoints).
# Per-wheel ceiling: μ_static · Fz · r_w · γ (static μ; no real-time adaptation).
# Handoff: Hermite smoothstep blend to tc_simple output.
#
# Intentional omissions vs intermediate/advanced:
#   No B-spline (overkill for reliable base)
#   No button input (legacy two-pedal only — add if steering wheel has button)
#   No yaw lock (simple mode has no reliable wz measurement path)
#   No real-time μ EMA (no DESC signal available)
#
# All functions are pure JAX — safe inside jit/grad/vmap/scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# §1  Phase constants
# ─────────────────────────────────────────────────────────────────────────────
SLC_IDLE    = 0
SLC_ARMED   = 1
SLC_LAUNCH  = 2
SLC_TC      = 3


# ─────────────────────────────────────────────────────────────────────────────
# §2  Configuration
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLCParams(NamedTuple):
    """
    Simple launch control tuning — all XLA compile-time constants.

    Profile note: quintic ramp is C²-continuous by construction.
    t_ramp_end controls where the torque plateau is reached; after that
    the profile holds at 1.0 until handoff. This avoids the torque droop
    visible in the advanced B-spline default coefficients at t>1.0s.
    """
    # ── Arming thresholds ─────────────────────────────────────────────────
    brake_arm_threshold: float    = 0.50   # brake fraction to arm
    throttle_arm_threshold: float = 0.50   # throttle fraction to arm
    brake_launch_threshold: float = 0.30   # brake below this → launch
    throttle_launch_gate: float   = 0.90   # throttle above this required
    # ── Profile ──────────────────────────────────────────────────────────
    T_peak_wheel: float           = 200.0  # [Nm] per wheel (conservative for simple)
    mu_static: float              = 1.50   # assumed surface μ (no probe)
    kappa_margin: float           = 0.90   # safety factor on friction ceiling
    t_ramp_end: float             = 0.80   # [s] time to reach plateau
    # ── Handoff ──────────────────────────────────────────────────────────
    v_handoff: float              = 6.0    # [m/s] speed gate to TC
    t_max_launch: float           = 3.0    # [s] maximum launch duration (safety)
    dt_blend: float               = 0.25   # [s] Hermite blend to TC
    # ── Hardware ─────────────────────────────────────────────────────────
    r_w: float                    = 0.2032 # [m] wheel radius
    # ── Drive config ─────────────────────────────────────────────────────
    front_torque_fraction: float  = 0.00   # 0.0 = RWD (Ter26); 0.5 = balanced AWD
    # Fz nominal (static, no load-transfer model in simple mode)
    Fz_front_nom: float           = 650.0  # [N] per front wheel
    Fz_rear_nom: float            = 800.0  # [N] per rear wheel


# ─────────────────────────────────────────────────────────────────────────────
# §3  State
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLCState(NamedTuple):
    phase: jax.Array         # scalar int32
    t_phase_start: jax.Array # scalar float: time current phase began [s]
    t_current: jax.Array     # scalar float: accumulated simulation time [s]
    position_x: jax.Array   # scalar: distance since launch start [m]

    @classmethod
    def default(cls) -> "SimpleLCState":
        return cls(
            phase=jnp.array(SLC_IDLE, dtype=jnp.int32),
            t_phase_start=jnp.array(0.0),
            t_current=jnp.array(0.0),
            position_x=jnp.array(0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §4  Output
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLCOutput(NamedTuple):
    T_command: jax.Array        # (4,) wheel torques [Nm]
    phase: jax.Array            # scalar: current phase code
    is_launch_active: jax.Array # scalar float: 1.0 during LAUNCH
    profile_value: jax.Array    # scalar: ramp value [0, 1]
    tc_ceiling: jax.Array       # (4,) torque ceiling [Nm]
    distance: jax.Array         # scalar: distance traveled [m]


# ─────────────────────────────────────────────────────────────────────────────
# §5  Quintic Polynomial Ramp
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def _quintic_ramp(t: jax.Array, t_ramp: float) -> jax.Array:
    """
    Quintic Hermite from 0→1 over [0, t_ramp], plateau at 1.0 after.

    Boundary conditions: f(0)=0, f'(0)=0, f''(0)=0, f(1)=1, f'(1)=0, f''(1)=0.
    Coefficients: 6t⁵ − 15t⁴ + 10t³  (classic smootherstep).
    C² at both endpoints → zero jerk at launch start (no torque spike) and
    smooth plateau entry (no resonance with drivetrain compliance).
    """
    s = jnp.clip(t / (t_ramp + 1e-6), 0.0, 1.0)
    return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3


# ─────────────────────────────────────────────────────────────────────────────
# §6  Hermite Smoothstep (handoff blend)
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def _hermite_blend(t: jax.Array, t_start: jax.Array, dt_blend: float) -> jax.Array:
    s = jnp.clip((t - t_start) / (dt_blend + 1e-6), 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


# ─────────────────────────────────────────────────────────────────────────────
# §7  Per-Wheel TC Ceiling (static μ, nominal Fz)
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def _simple_tc_ceiling(params: SimpleLCParams) -> jax.Array:
    """
    Conservative traction ceiling from static μ and nominal Fz.
    No load-transfer model — safe-side estimate for simple mode.
    """
    Fz = jnp.array([
        params.Fz_front_nom,
        params.Fz_front_nom,
        params.Fz_rear_nom,
        params.Fz_rear_nom,
    ])
    return params.mu_static * Fz * params.r_w * params.kappa_margin


# ─────────────────────────────────────────────────────────────────────────────
# §8  Torque Distribution
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def _distribute_torque(
    profile_val: jax.Array,
    T_ceiling: jax.Array,
    T_max_hw: jax.Array,
    params: SimpleLCParams,
) -> jax.Array:
    """
    Front/rear split controlled by front_torque_fraction.
    Per-wheel ceiling enforced before hardware limit.
    """
    T_rear_total  = params.T_peak_wheel * 2.0 * (1.0 - params.front_torque_fraction) * profile_val
    T_front_total = params.T_peak_wheel * 2.0 * params.front_torque_fraction * profile_val

    T_cmd = jnp.array([
        T_front_total * 0.5,
        T_front_total * 0.5,
        T_rear_total  * 0.5,
        T_rear_total  * 0.5,
    ])
    return jnp.minimum(jnp.minimum(T_cmd, T_ceiling), T_max_hw)


# ─────────────────────────────────────────────────────────────────────────────
# §9  Main Step
# ─────────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnums=())
def simple_launch_step(
    throttle: jax.Array,         # [0, 1]
    brake: jax.Array,            # [0, 1]
    vx: jax.Array,               # [m/s]
    T_tc: jax.Array,             # (4,) tc_simple output for handoff [Nm]
    T_max_hw: jax.Array,         # (4,) hardware torque ceiling [Nm]
    lc_state: SimpleLCState,
    dt: jax.Array,
    params: SimpleLCParams = SimpleLCParams(),
) -> Tuple[SimpleLCOutput, SimpleLCState]:
    """
    Single-step simple launch controller.

    Caller wires:
        T_tc  ← tc_simple output (used only during HANDOFF blend)
        T_max_hw ← per-wheel hardware motor limit
    """
    phase   = lc_state.phase
    t_start = lc_state.t_phase_start
    t       = lc_state.t_current

    # ── Phase transitions ──────────────────────────────────────────────────
    enter_armed = (
        (phase == SLC_IDLE)
        & (brake    > params.brake_arm_threshold)
        & (throttle > params.throttle_arm_threshold)
    )
    phase   = jnp.where(enter_armed, SLC_ARMED, phase)
    t_start = jnp.where(enter_armed, t, t_start)

    enter_launch = (
        (phase == SLC_ARMED)
        & (brake    < params.brake_launch_threshold)
        & (throttle > params.throttle_launch_gate)
    )
    phase   = jnp.where(enter_launch, SLC_LAUNCH, phase)
    t_start = jnp.where(enter_launch, t, t_start)

    t_elapsed = t - t_start

    enter_tc = (
        (phase == SLC_LAUNCH)
        & ((vx > params.v_handoff) | (t_elapsed > params.t_max_launch))
    )
    phase   = jnp.where(enter_tc, SLC_TC, phase)
    t_start = jnp.where(enter_tc, t, t_start)

    # Hard abort: heavy braking at any active phase
    abort = (phase >= SLC_LAUNCH) & (brake > 0.5)
    phase = jnp.where(abort, jnp.array(SLC_IDLE, dtype=jnp.int32), phase)

    # ── Torque computation ─────────────────────────────────────────────────
    t_launch_elapsed = jnp.maximum(t - t_start, 0.0)

    T_ceiling = _simple_tc_ceiling(params)
    profile_val = _quintic_ramp(t_launch_elapsed, params.t_ramp_end)
    T_launch = _distribute_torque(profile_val, T_ceiling, T_max_hw, params)

    # Hermite blend from LAUNCH to TC
    w_blend   = _hermite_blend(t, t_start, params.dt_blend)
    T_handoff = (1.0 - w_blend) * T_launch + w_blend * T_tc

    is_launch = (phase == SLC_LAUNCH).astype(jnp.float32)
    is_tc_blend = (phase == SLC_TC).astype(jnp.float32)
    T_idle = jnp.zeros(4)

    T_out = jnp.where(
        phase <= SLC_ARMED, T_idle,
        jnp.where(phase == SLC_LAUNCH, T_launch,
                  jnp.where(phase == SLC_TC, T_handoff, T_tc)),
    )

    is_launch_active = is_launch

    pos_new = lc_state.position_x + vx * dt

    output = SimpleLCOutput(
        T_command=T_out,
        phase=phase,
        is_launch_active=is_launch_active,
        profile_value=profile_val,
        tc_ceiling=T_ceiling,
        distance=pos_new,
    )
    new_state = SimpleLCState(
        phase=phase,
        t_phase_start=t_start,
        t_current=t + dt,
        position_x=pos_new,
    )
    return output, new_state