# powertrain/launch_control.py
# Project-GP — Neural Predictive Launch Sequencer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Manages the 0–75 m acceleration event via three phases:
#
#   ARMED:   Pre-launch mu probe (50 ms torque pulse per wheel)
#   LAUNCH:  Open-loop B-spline torque profile (offline-optimized)
#   HANDOFF: C1-continuous Hermite smoothstep → closed-loop DESC TC
#
# Public API (used by powertrain_manager and sanity checks):
#   launch_step(throttle, brake, vx, omega_wheel, T_tc, launch_state, dt, params)
#
# Internal API (used by optimize_launch_profile):
#   _launch_step_internal(t, vx, Fz, T_max, T_tc, brake_pressed, throttle_full,
#                         launch_state, dt, cfg)
#
# All functions are pure JAX — safe inside jit/grad/vmap/scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# §1  Launch Configuration
# ─────────────────────────────────────────────────────────────────────────────

class LaunchConfig(NamedTuple):
    """Static launch control configuration."""
    n_spline_knots: int = 16
    t_profile_duration: float = 2.0
    t_handoff_min: float = 0.8
    v_handoff_threshold: float = 5.0
    dt_blend: float = 0.3
    mu_probe_torque_frac: float = 0.10
    mu_probe_duration: float = 0.05
    T_peak_wheel: float = 450.0
    front_ratio_initial: float = 0.35
    front_ratio_final: float = 0.28


# Public alias — manager and tests import LaunchParams; LaunchConfig is the
# canonical internal name. Both point to the same NamedTuple class.
LaunchParams = LaunchConfig


# ─────────────────────────────────────────────────────────────────────────────
# §2  Launch State
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


class LaunchState(NamedTuple):
    """Persistent launch control state across timesteps."""
    phase: jax.Array              # scalar int32: current phase
    t_phase_start: jax.Array      # scalar float: time when current phase began
    mu_probe_result: jax.Array    # scalar: estimated mu from probe pulse
    mu_probe_wheel_idx: jax.Array # scalar int32: which wheel is being probed
    spline_coeffs: jax.Array      # (16,) B-spline coefficients for launch profile
    position_x: jax.Array         # scalar: distance traveled since launch [m]
    t_current: jax.Array          # scalar: accumulated simulation time [s]

    @classmethod
    def default(cls, params: "LaunchConfig" = None) -> "LaunchState":
        """
        Factory classmethod. `params` is accepted for API compatibility with
        PowertrainManagerState.default(config) but is currently unused —
        all state is initialised to physical zero/default values.
        """
        return cls(
            phase=jnp.array(PHASE_IDLE, dtype=jnp.int32),
            t_phase_start=jnp.array(0.0),
            mu_probe_result=jnp.array(1.5),
            mu_probe_wheel_idx=jnp.array(0, dtype=jnp.int32),
            spline_coeffs=_DEFAULT_SPLINE_COEFFS,
            position_x=jnp.array(0.0),
            t_current=jnp.array(0.0),
        )


def make_launch_state(spline_coeffs: jax.Array = None) -> LaunchState:
    """Functional factory (legacy path, kept for backward compat)."""
    coeffs = _DEFAULT_SPLINE_COEFFS if spline_coeffs is None else spline_coeffs
    return LaunchState(
        phase=jnp.array(PHASE_IDLE, dtype=jnp.int32),
        t_phase_start=jnp.array(0.0),
        mu_probe_result=jnp.array(1.5),
        mu_probe_wheel_idx=jnp.array(0, dtype=jnp.int32),
        spline_coeffs=coeffs,
        position_x=jnp.array(0.0),
        t_current=jnp.array(0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §3  B-Spline Launch Profile Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def evaluate_bspline_profile(
    t: jax.Array,
    coeffs: jax.Array,
    duration: float = 2.0,
) -> jax.Array:
    n = coeffs.shape[0]
    t_norm = jnp.clip(t / (duration + 1e-6), 0.0, 1.0) * (n - 1)
    idx = jnp.floor(t_norm).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n - 2)
    frac = t_norm - idx.astype(jnp.float32)

    p0 = coeffs[idx]
    p1 = coeffs[idx + 1]
    idx_prev = jnp.maximum(idx - 1, 0)
    idx_next = jnp.minimum(idx + 2, n - 1)
    m0 = (coeffs[idx + 1] - coeffs[idx_prev]) * 0.5
    m1 = (coeffs[idx_next] - coeffs[idx]) * 0.5

    t2 = frac * frac
    t3 = t2 * frac
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + frac
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    return jnp.clip(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Dynamic Front-Rear Torque Split
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def launch_front_ratio(
    t_launch: jax.Array,
    vx: jax.Array,
    cfg: LaunchConfig = LaunchConfig(),
) -> jax.Array:
    speed_frac = jax.nn.sigmoid((vx - 8.0) * 0.5)
    f_ratio = (cfg.front_ratio_initial * (1.0 - speed_frac)
               + cfg.front_ratio_final * speed_frac)
    return jnp.clip(f_ratio, 0.15, 0.50)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Per-Wheel Torque Distribution
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
    T_wheels = jnp.clip(T_wheels, 0.0, T_max)
    T_friction_limit = mu_est * Fz * r_w * 0.95
    return jnp.minimum(T_wheels, T_friction_limit)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Pre-Launch Mu Probe
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
# §7  Hermite Smoothstep
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def hermite_smoothstep(t: jax.Array, t_start: jax.Array, dt_blend: float = 0.3):
    s = jnp.clip((t - t_start) / (dt_blend + 1e-6), 0.0, 1.0)
    return 3.0 * s ** 2 - 2.0 * s ** 3


# ─────────────────────────────────────────────────────────────────────────────
# §8  Launch Control Output
# ─────────────────────────────────────────────────────────────────────────────

class LaunchOutput(NamedTuple):
    """Output of a single launch control step."""
    T_command: jax.Array        # (4,) commanded wheel torques [Nm]  ← was T_wheel
    phase: jax.Array            # scalar: current phase
    t_elapsed: jax.Array        # scalar: time since launch start [s]
    profile_value: jax.Array    # scalar: B-spline profile value [0,1]
    f_front: jax.Array          # scalar: current front torque fraction  ← was front_ratio
    mu_estimate: jax.Array      # scalar: estimated friction coefficient
    is_launch_active: jax.Array # scalar: 1 if in launch/handoff phase
    distance: jax.Array         # scalar: distance traveled [m]


# ─────────────────────────────────────────────────────────────────────────────
# §9  Internal Launch Step (full-argument form for offline optimization)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def _launch_step_internal(
    t: jax.Array,
    vx: jax.Array,
    Fz: jax.Array,
    T_max: jax.Array,
    T_tc: jax.Array,
    brake_pressed: jax.Array,
    throttle_full: jax.Array,
    launch_state: LaunchState,
    dt: jax.Array,
    cfg: LaunchConfig = LaunchConfig(),
) -> tuple[LaunchOutput, LaunchState]:
    phase   = launch_state.phase
    t_start = launch_state.t_phase_start
    mu_est  = launch_state.mu_probe_result
    coeffs  = launch_state.spline_coeffs
    pos_x   = launch_state.position_x

    t_elapsed = t - t_start

    enter_armed = (phase == PHASE_IDLE) & (brake_pressed > 0.5) & (throttle_full > 0.5)
    phase   = jnp.where(enter_armed, PHASE_ARMED, phase)
    t_start = jnp.where(enter_armed, t, t_start)

    enter_launch = (phase == PHASE_ARMED) & (brake_pressed < 0.5)
    phase   = jnp.where(enter_launch, PHASE_LAUNCH, phase)
    t_start = jnp.where(enter_launch, t, t_start)
    pos_x   = jnp.where(enter_launch, 0.0, pos_x)

    t_elapsed = t - t_start

    enter_handoff = ((phase == PHASE_LAUNCH) &
                     ((vx > cfg.v_handoff_threshold) |
                      (t_elapsed > cfg.t_profile_duration)))
    phase   = jnp.where(enter_handoff, PHASE_HANDOFF, phase)
    t_start = jnp.where(enter_handoff, t, t_start)
    t_elapsed = t - t_start

    enter_tc = (phase == PHASE_HANDOFF) & (t_elapsed > cfg.dt_blend)
    phase = jnp.where(enter_tc, PHASE_TC, phase)

    T_idle = jnp.zeros(4)

    t_launch_elapsed = jnp.maximum(t - launch_state.t_phase_start, 0.0)
    profile_val = evaluate_bspline_profile(t_launch_elapsed, coeffs, cfg.t_profile_duration)
    mu_scale    = jnp.clip(mu_est / 1.5, 0.5, 1.2)
    T_total_launch = cfg.T_peak_wheel * 4.0 * profile_val * mu_scale
    f_ratio    = launch_front_ratio(t_launch_elapsed, vx, cfg)
    T_launch   = launch_torque_distribution(T_total_launch, f_ratio, mu_est, Fz, T_max)

    w_blend   = hermite_smoothstep(t, launch_state.t_phase_start, cfg.dt_blend)
    T_handoff = (1.0 - w_blend) * T_launch + w_blend * T_tc

    is_idle_or_armed = (phase <= PHASE_ARMED)
    is_launch        = (phase == PHASE_LAUNCH)
    is_handoff       = (phase == PHASE_HANDOFF)

    T_out = jnp.where(
        is_idle_or_armed, T_idle,
        jnp.where(is_launch, T_launch,
                  jnp.where(is_handoff, T_handoff, T_tc)),
    )

    pos_x_new = pos_x + vx * dt
    is_launch_active = ((phase >= PHASE_LAUNCH) & (phase <= PHASE_HANDOFF)).astype(jnp.float32)

    output = LaunchOutput(
        T_command=T_out,
        phase=phase,
        t_elapsed=t - launch_state.t_phase_start,
        profile_value=profile_val,
        f_front=f_ratio,
        mu_estimate=mu_est,
        is_launch_active=is_launch_active,
        distance=pos_x_new,
    )

    new_state = LaunchState(
        phase=phase,
        t_phase_start=t_start,
        mu_probe_result=mu_est,
        mu_probe_wheel_idx=launch_state.mu_probe_wheel_idx,
        spline_coeffs=coeffs,
        position_x=pos_x_new,
        t_current=launch_state.t_current + dt,
    )

    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §10  Public Launch Step — 8-arg API (manager + sanity checks)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def launch_step(
    throttle: jax.Array,         # [0, 1] throttle pedal position
    brake: jax.Array,            # [0, 1] brake pedal position
    vx: jax.Array,               # longitudinal velocity [m/s]
    omega_wheel: jax.Array,      # (4,) wheel speeds [rad/s] — for future mu probe
    T_tc: jax.Array,             # (4,) TC/TV torques for handoff target [Nm]
    launch_state: LaunchState,
    dt: jax.Array,
    params: LaunchConfig = LaunchConfig(),
) -> tuple[LaunchOutput, LaunchState]:
    """
    Public-facing launch step. Uses accumulated t_current from LaunchState
    so callers never need to track simulation time externally.

    Generates nominal Fz (static + acceleration load transfer) and T_max
    from LaunchConfig. For higher fidelity, call _launch_step_internal directly
    with explicit Fz and T_max.
    """
    # Static + pitch-load Fz estimate (conservative for allocation purposes)
    Fz_default = jnp.array([600.0, 600.0, 800.0, 800.0])   # rear-heavy under accel
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# §11  Offline Launch Profile Optimization Interface
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
    t_75m through the full physics engine. Runs OFFLINE (10-30 min on CPU).
    """
    import optax

    def forward_sim(coeffs):
        def scan_body(carry, step_idx):
            x, pos = carry
            t_step = step_idx * dt
            frac = evaluate_bspline_profile(t_step, coeffs, cfg.t_profile_duration)
            F_total = cfg.T_peak_wheel * 4.0 * frac / 0.2032
            u = jnp.array([0.0, F_total])
            x_next = simulate_step_fn(x, u, setup_params, dt=dt, n_substeps=5)
            pos_next = pos + jnp.maximum(x_next[14], 0.0) * dt
            return (x_next, pos_next), pos_next

        (_, pos_final), positions = jax.lax.scan(
            scan_body, (x0, jnp.array(0.0)), jnp.arange(n_steps),
        )
        weights  = jax.nn.softmax(-50.0 * jax.nn.relu(target_distance - positions))
        t_target = jnp.sum(weights * jnp.arange(n_steps) * dt)
        return t_target

    coeffs    = _DEFAULT_SPLINE_COEFFS
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(coeffs)
    grad_fn   = jax.jit(jax.grad(forward_sim))

    print(f"[LaunchOpt] Optimizing 16 B-spline coefficients over {n_steps} steps...")
    for i in range(n_optim_iters):
        g = grad_fn(coeffs)
        updates, opt_state = optimizer.update(g, opt_state)
        coeffs = optax.apply_updates(coeffs, updates)
        coeffs = jnp.clip(coeffs, 0.01, 1.0)
        if i % 20 == 0:
            print(f"  Iter {i:3d}: t_75m = {float(forward_sim(coeffs)):.4f} s | "
                  f"max_coeff = {float(jnp.max(coeffs)):.3f}")

    print(f"[LaunchOpt] Final: t_75m = {float(forward_sim(coeffs)):.4f} s")
    return coeffs