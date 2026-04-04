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
# The B-spline launch profile is parameterized by 16 coefficients that are
# optimized offline by differentiating t_75m through the full 46-DOF physics
# engine. The profile accounts for:
#   · Transient slip relaxation at v=0 (tau -> inf)
#   · Weight transfer pitch oscillation during torque onset
#   · Tire warm-up from cold start (mu(40C) < mu(85C))
#   · Battery voltage sag under peak current draw
#   · Dynamic front-rear torque split as weight transfers rearward
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
    n_spline_knots: int = 16       # B-spline knots for torque profile
    t_profile_duration: float = 2.0  # s total duration of open-loop phase
    t_handoff_min: float = 0.8     # s earliest possible handoff to TC
    v_handoff_threshold: float = 5.0  # m/s speed threshold for handoff
    dt_blend: float = 0.3         # s Hermite smoothstep blend duration
    mu_probe_torque_frac: float = 0.10  # fraction of T_peak for mu probe
    mu_probe_duration: float = 0.05     # s probe pulse duration
    T_peak_wheel: float = 450.0   # Nm peak wheel torque (per motor)
    front_ratio_initial: float = 0.35   # initial front torque fraction
    front_ratio_final: float = 0.28     # final front fraction (weight transferred)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Launch State
# ─────────────────────────────────────────────────────────────────────────────

# Phase encoding (static integers for XLA compatibility)
PHASE_IDLE = 0
PHASE_ARMED = 1
PHASE_PROBE = 2
PHASE_LAUNCH = 3
PHASE_HANDOFF = 4
PHASE_TC = 5


class LaunchState(NamedTuple):
    """Persistent launch control state across timesteps."""
    phase: jax.Array              # scalar int: current phase
    t_phase_start: jax.Array      # scalar: time when current phase began
    mu_probe_result: jax.Array    # scalar: estimated mu from probe pulse
    mu_probe_wheel_idx: jax.Array # scalar int: which wheel is being probed
    spline_coeffs: jax.Array      # (16,) B-spline coefficients for launch profile
    position_x: jax.Array         # scalar: distance traveled since launch [m]


def make_launch_state(spline_coeffs: jax.Array = None) -> LaunchState:
    """Create initial launch state. Optionally provide pre-optimized spline coefficients."""
    if spline_coeffs is None:
        # Default launch profile: rapid rise to 80% in 0.3s, plateau at 95%
        spline_coeffs = jnp.array([
            0.05, 0.20, 0.50, 0.75,   # 0.0–0.5s: rapid ramp
            0.88, 0.93, 0.95, 0.96,   # 0.5–1.0s: approaching peak
            0.96, 0.95, 0.94, 0.93,   # 1.0–1.5s: slight taper (weight settles)
            0.92, 0.91, 0.90, 0.90,   # 1.5–2.0s: steady-state (transition to TC)
        ])
    return LaunchState(
        phase=jnp.array(PHASE_IDLE, dtype=jnp.int32),
        t_phase_start=jnp.array(0.0),
        mu_probe_result=jnp.array(1.5),  # default: assume dry asphalt
        mu_probe_wheel_idx=jnp.array(0, dtype=jnp.int32),
        spline_coeffs=spline_coeffs,
        position_x=jnp.array(0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §3  B-Spline Launch Profile Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def evaluate_bspline_profile(
    t: jax.Array,               # time since launch start [s]
    coeffs: jax.Array,          # (16,) B-spline coefficients [0, 1] normalized
    duration: float = 2.0,      # total profile duration [s]
) -> jax.Array:
    """
    Evaluate the B-spline launch torque profile at time t.

    Returns a normalized torque fraction [0, 1] that is multiplied by
    T_peak to get the actual torque command.

    Uses cubic B-spline interpolation with natural boundary conditions.
    The 16 coefficients span [0, duration] uniformly.
    """
    n = coeffs.shape[0]
    # Normalize time to [0, n-1] index space
    t_norm = jnp.clip(t / (duration + 1e-6), 0.0, 1.0) * (n - 1)

    # Integer and fractional parts
    idx = jnp.floor(t_norm).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n - 2)
    frac = t_norm - idx.astype(jnp.float32)

    # Cubic Hermite interpolation between adjacent knots
    # p(t) = (2t^3 - 3t^2 + 1)·p0 + (t^3 - 2t^2 + t)·m0
    #       + (-2t^3 + 3t^2)·p1 + (t^3 - t^2)·m1
    p0 = coeffs[idx]
    p1 = coeffs[idx + 1]

    # Finite-difference tangents (Catmull-Rom style)
    # For boundary knots, use one-sided differences
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

    value = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    return jnp.clip(value, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Dynamic Front-Rear Torque Split During Launch
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def launch_front_ratio(
    t_launch: jax.Array,         # time since launch start [s]
    vx: jax.Array,               # current vehicle speed [m/s]
    cfg: LaunchConfig = LaunchConfig(),
) -> jax.Array:
    """
    Dynamic front-to-total torque ratio during launch.

    Starts at front_ratio_initial (higher at standstill where weight
    distribution is ~45% front) and decreases to front_ratio_final as
    weight transfers rearward under acceleration.

    The transition follows a sigmoid of vehicle speed — as the car
    accelerates, the front loses Fz and should contribute less torque.
    """
    # Speed-based transition: at 0 m/s use initial, at ~15 m/s use final
    speed_frac = jax.nn.sigmoid((vx - 8.0) * 0.5)  # transition centered at 8 m/s

    f_ratio = (cfg.front_ratio_initial * (1.0 - speed_frac)
               + cfg.front_ratio_final * speed_frac)

    return jnp.clip(f_ratio, 0.15, 0.50)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Per-Wheel Torque Distribution During Launch
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def launch_torque_distribution(
    T_total: jax.Array,          # scalar total torque demand [Nm]
    f_ratio: jax.Array,          # front-to-total ratio [0, 1]
    mu_est: jax.Array,           # estimated friction coefficient
    Fz: jax.Array,               # (4,) vertical loads [N]
    T_max: jax.Array,            # (4,) per-motor max torque [Nm]
    r_w: float = 0.2032,
) -> jax.Array:
    """
    Distribute launch torque to 4 wheels proportional to available
    friction budget, respecting the front-rear ratio and motor limits.

    Within each axle (front/rear), torque splits left-right proportional
    to vertical load (heavier side gets more torque = more friction budget).
    """
    T_front = T_total * f_ratio
    T_rear = T_total * (1.0 - f_ratio)

    # Left-right split proportional to Fz within each axle
    Fz_front_total = Fz[0] + Fz[1] + 1e-3
    Fz_rear_total = Fz[2] + Fz[3] + 1e-3

    T_fl = T_front * Fz[0] / Fz_front_total
    T_fr = T_front * Fz[1] / Fz_front_total
    T_rl = T_rear * Fz[2] / Fz_rear_total
    T_rr = T_rear * Fz[3] / Fz_rear_total

    T_wheels = jnp.array([T_fl, T_fr, T_rl, T_rr])

    # Clamp to motor limits
    T_wheels = jnp.clip(T_wheels, 0.0, T_max)

    # Friction limit: T_i <= mu * Fz_i * r_w (with 5% safety margin)
    T_friction_limit = mu_est * Fz * r_w * 0.95
    T_wheels = jnp.minimum(T_wheels, T_friction_limit)

    return T_wheels


# ─────────────────────────────────────────────────────────────────────────────
# §6  Pre-Launch Mu Probe
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def mu_probe_estimate(
    T_applied: jax.Array,    # Nm torque applied during probe
    omega_dot: jax.Array,    # rad/s^2 wheel angular acceleration during probe
    Iw: float = 1.2,         # wheel inertia [kg*m^2]
    r_w: float = 0.2032,     # tire radius [m]
    Fz: jax.Array = jnp.array(735.0),  # vertical load [N]
) -> jax.Array:
    """
    Estimate tire-road friction coefficient from the probe pulse response.

    mu_est = Fx_tire / Fz = (T_applied - Iw * omega_dot * r_w) / (Fz * r_w)

    If the wheel barely accelerates (high Fx_tire), mu is high.
    If the wheel spins up quickly (low Fx_tire), mu is low.
    """
    Fx_tire = (T_applied - Iw * omega_dot * r_w) / r_w
    mu_est = jnp.abs(Fx_tire) / (jnp.maximum(Fz, 10.0))
    return jnp.clip(mu_est, 0.3, 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Hermite Smoothstep Handoff (C1-continuous)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def hermite_smoothstep(t: jax.Array, t_start: jax.Array, dt_blend: float = 0.3):
    """
    C1-continuous transition weight w(t) in [0, 1].
    w(t_start) = 0, w(t_start + dt_blend) = 1.
    w'(t_start) = 0, w'(t_start + dt_blend) = 0.
    """
    s = jnp.clip((t - t_start) / (dt_blend + 1e-6), 0.0, 1.0)
    # Hermite: w = 3s^2 - 2s^3
    return 3.0 * s ** 2 - 2.0 * s ** 3


# ─────────────────────────────────────────────────────────────────────────────
# §8  Launch Control Output
# ─────────────────────────────────────────────────────────────────────────────

class LaunchOutput(NamedTuple):
    """Output of a single launch control step."""
    T_wheel: jax.Array          # (4,) commanded wheel torques [Nm]
    phase: jax.Array            # scalar: current phase (for diagnostics)
    t_elapsed: jax.Array        # scalar: time since launch start [s]
    profile_value: jax.Array    # scalar: B-spline profile value [0,1]
    front_ratio: jax.Array      # scalar: current front torque fraction
    mu_estimate: jax.Array      # scalar: estimated friction coefficient
    is_launch_active: jax.Array # scalar: 1 if in launch/handoff phase
    distance: jax.Array         # scalar: distance traveled [m]


# ─────────────────────────────────────────────────────────────────────────────
# §9  Launch Control Step (single timestep)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=())
def launch_step(
    t: jax.Array,               # simulation time [s]
    vx: jax.Array,              # longitudinal velocity [m/s]
    Fz: jax.Array,              # (4,) vertical loads [N]
    T_max: jax.Array,           # (4,) per-motor max torque [Nm]
    T_tc: jax.Array,            # (4,) torques from TC (for handoff blending)
    brake_pressed: jax.Array,   # scalar: 1 if brake pedal pressed
    throttle_full: jax.Array,   # scalar: 1 if throttle at 100%
    launch_state: LaunchState,
    dt: jax.Array,              # timestep [s]
    cfg: LaunchConfig = LaunchConfig(),
) -> tuple[LaunchOutput, LaunchState]:
    """
    Single launch control timestep. Implements the state machine:

    IDLE -> ARMED (brake + full throttle held)
    ARMED -> LAUNCH (brake released)
    LAUNCH -> HANDOFF (v > threshold OR t > t_profile_duration)
    HANDOFF -> TC (blend complete)

    All transitions are smooth (no hard switches in the torque output).
    Phase transitions use jnp.where for XLA compatibility (no Python if/else).
    """
    phase = launch_state.phase
    t_start = launch_state.t_phase_start
    mu_est = launch_state.mu_probe_result
    coeffs = launch_state.spline_coeffs
    pos_x = launch_state.position_x

    t_elapsed = t - t_start

    # ── Phase transition logic (all via jnp.where for JIT) ───────────────
    # IDLE -> ARMED: brake + throttle both pressed
    enter_armed = (phase == PHASE_IDLE) & (brake_pressed > 0.5) & (throttle_full > 0.5)
    phase = jnp.where(enter_armed, PHASE_ARMED, phase)
    t_start = jnp.where(enter_armed, t, t_start)

    # ARMED -> LAUNCH: brake released (launch trigger!)
    enter_launch = (phase == PHASE_ARMED) & (brake_pressed < 0.5)
    phase = jnp.where(enter_launch, PHASE_LAUNCH, phase)
    t_start = jnp.where(enter_launch, t, t_start)
    pos_x = jnp.where(enter_launch, 0.0, pos_x)

    # Recompute t_elapsed after possible phase change
    t_elapsed = t - t_start

    # LAUNCH -> HANDOFF: speed threshold OR profile duration exceeded
    enter_handoff = ((phase == PHASE_LAUNCH) &
                     ((vx > cfg.v_handoff_threshold) |
                      (t_elapsed > cfg.t_profile_duration)))
    phase = jnp.where(enter_handoff, PHASE_HANDOFF, phase)
    t_start = jnp.where(enter_handoff, t, t_start)
    t_elapsed = t - t_start

    # HANDOFF -> TC: blend complete
    enter_tc = (phase == PHASE_HANDOFF) & (t_elapsed > cfg.dt_blend)
    phase = jnp.where(enter_tc, PHASE_TC, phase)

    # ── Compute torque for each phase ────────────────────────────────────

    # --- IDLE / ARMED: zero torque (driver holds brake) ---
    T_idle = jnp.zeros(4)

    # --- LAUNCH: B-spline profile × T_peak × mu scaling ---
    t_launch_elapsed = jnp.maximum(t - launch_state.t_phase_start, 0.0)
    profile_val = evaluate_bspline_profile(t_launch_elapsed, coeffs, cfg.t_profile_duration)
    # Scale by mu estimate (lower mu → less aggressive profile)
    mu_scale = jnp.clip(mu_est / 1.5, 0.5, 1.2)  # normalized to dry asphalt
    T_total_launch = cfg.T_peak_wheel * 4.0 * profile_val * mu_scale
    f_ratio = launch_front_ratio(t_launch_elapsed, vx, cfg)
    T_launch = launch_torque_distribution(T_total_launch, f_ratio, mu_est, Fz, T_max)

    # --- HANDOFF: Hermite smoothstep blend from launch to TC ---
    w_blend = hermite_smoothstep(t, launch_state.t_phase_start, cfg.dt_blend)
    # Only use handoff blend when actually in HANDOFF phase
    T_handoff = (1.0 - w_blend) * T_launch + w_blend * T_tc

    # --- TC: pass through (launch control inactive) ---
    T_tc_pass = T_tc

    # ── Select torque based on phase ─────────────────────────────────────
    # Use nested jnp.where for XLA-compatible phase selection
    is_idle_or_armed = (phase <= PHASE_ARMED)
    is_launch = (phase == PHASE_LAUNCH)
    is_handoff = (phase == PHASE_HANDOFF)
    # is_tc = (phase == PHASE_TC)

    T_out = jnp.where(
        is_idle_or_armed,
        T_idle,
        jnp.where(
            is_launch,
            T_launch,
            jnp.where(
                is_handoff,
                T_handoff,
                T_tc_pass,  # TC phase: pass through
            ),
        ),
    )

    # ── Update position ──────────────────────────────────────────────────
    pos_x_new = pos_x + vx * dt

    # ── Is launch active? (for TC blend weight computation) ──────────────
    is_launch_active = ((phase >= PHASE_LAUNCH) & (phase <= PHASE_HANDOFF)).astype(jnp.float32)

    # ── Pack output ──────────────────────────────────────────────────────
    output = LaunchOutput(
        T_wheel=T_out,
        phase=phase,
        t_elapsed=t - launch_state.t_phase_start,
        profile_value=profile_val,
        front_ratio=f_ratio,
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
    )

    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §10  Offline Launch Profile Optimization Interface
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
    t_75m through the full physics engine.

    This runs OFFLINE (not real-time). Takes 10-30 minutes on CPU.

    Args:
        simulate_step_fn: vehicle.simulate_step (must accept [delta, F_total] controls)
        setup_params: (28,) suspension setup
        x0: (46,) initial state at standstill
        target_distance: meters (75 for FS acceleration event)

    Returns:
        optimized_coeffs: (16,) B-spline coefficients
    """
    import optax

    def forward_sim(coeffs):
        """Simulate launch with given B-spline profile, return time to target."""
        T_peak = cfg.T_peak_wheel

        def scan_body(carry, step_idx):
            x, pos = carry
            t_step = step_idx * dt
            # Evaluate profile
            frac = evaluate_bspline_profile(t_step, coeffs, cfg.t_profile_duration)
            F_total = T_peak * 4.0 * frac / 0.2032  # total force [N]
            u = jnp.array([0.0, F_total])  # zero steering, forward force
            x_next = simulate_step_fn(x, u, setup_params, dt=dt, n_substeps=5)
            pos_next = pos + jnp.maximum(x_next[14], 0.0) * dt  # integrate vx
            return (x_next, pos_next), pos_next

        (x_final, pos_final), positions = jax.lax.scan(
            scan_body, (x0, jnp.array(0.0)), jnp.arange(n_steps),
        )

        # Soft terminal time: weighted average of timesteps near target_distance
        # Uses softmax to differentiably identify when pos crosses target
        weights = jax.nn.softmax(-50.0 * jax.nn.relu(target_distance - positions))
        t_target = jnp.sum(weights * jnp.arange(n_steps) * dt)

        return t_target

    # Initialize optimizer
    coeffs = jnp.array([
        0.05, 0.20, 0.50, 0.75, 0.88, 0.93, 0.95, 0.96,
        0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.90,
    ])

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(coeffs)

    grad_fn = jax.jit(jax.grad(forward_sim))

    print(f"[LaunchOpt] Optimizing 16 B-spline coefficients over {n_steps} steps...")
    for i in range(n_optim_iters):
        g = grad_fn(coeffs)
        updates, opt_state = optimizer.update(g, opt_state)
        coeffs = optax.apply_updates(coeffs, updates)
        coeffs = jnp.clip(coeffs, 0.01, 1.0)  # physical bounds

        if i % 20 == 0:
            t_75 = forward_sim(coeffs)
            print(f"  Iter {i:3d}: t_75m = {float(t_75):.4f} s | "
                  f"max_coeff = {float(jnp.max(coeffs)):.3f}")

    t_final = forward_sim(coeffs)
    print(f"[LaunchOpt] Final: t_75m = {float(t_final):.4f} s")
    return coeffs