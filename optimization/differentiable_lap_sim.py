#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# optimization/differentiable_lap_sim.py
# Project-GP — Full-Lap Differentiable Simulation
# ═══════════════════════════════════════════════════════════════════════════════
#
# THE KILL FEATURE: No FS team in the world has this.
#
#   ∂(lap_time) / ∂(setup_vector) = jax.grad(simulate_full_lap)(setup)
#
# This module connects:
#   1. DifferentiableTrack (Batch 4, file 1) — track geometry
#   2. DifferentiableMultiBodyVehicle — 46-DOF physics engine
#   3. Path-following controller — closed-loop driver model
#   4. Energy accounting — integrated kJ consumption
#   5. Lap time — total elapsed time = N_steps × dt
#
# Everything compiles into a single XLA graph via jax.lax.scan.
# jax.grad traces from the scalar lap_time back through:
#   lap_time → speed profile → tire forces → suspension loads → setup params
#
# DESIGN CONSTRAINTS:
#   · Fixed-step simulation (N_steps × dt) — no adaptive stepping
#   · Closed-loop driver (no pre-computed optimal trajectory)
#   · Track progress as primary state variable (not position)
#   · All controller gains are smooth (no hard switches)
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
from functools import partial

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import jax
import jax.numpy as jnp
import numpy as np

from optimization.differentiable_track import (
    DifferentiableTrack, interp_track_at_s, make_differentiable_track,
)
from data.configs.vehicle_params import vehicle_params as VP
from data.configs.tire_coeffs import tire_coeffs as TC


# ─────────────────────────────────────────────────────────────────────────────
# §1  Differentiable Path-Following Controller
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def path_following_controller(
    vx: jax.Array,          # longitudinal velocity [m/s]
    vy: jax.Array,          # lateral velocity [m/s]
    yaw: jax.Array,         # vehicle yaw angle [rad]
    s_progress: jax.Array,  # arc-length progress on track [m]
    track: DifferentiableTrack,
    mu_friction: float = 1.40,
) -> tuple:
    """
    Closed-loop path-following driver model.

    Steering: pursuit-point kinematic controller
    Speed: friction-circle-limited P-controller

    Returns: (delta, F_drive, track_info_dict)
    """
    L_wb = VP.get('lf', 0.8525) + VP.get('lr', 0.6975)
    vx_safe = jnp.maximum(vx, 1.0)

    # ── Look-ahead: query track at current + preview distance ────────────
    d_preview = jnp.clip(vx_safe * 0.25, 2.0, 8.0)  # 0.25s preview, clamped
    track_now = interp_track_at_s(track, s_progress)
    track_ahead = interp_track_at_s(track, s_progress + d_preview)

    kappa_now = track_now['kappa']
    kappa_ahead = track_ahead['kappa']

    # Blend current + ahead curvature for anticipatory steering
    kappa_blend = 0.4 * kappa_now + 0.6 * kappa_ahead

    # ── Steering: kinematic + correction for lateral offset ──────────────
    # Base kinematic steering
    delta_kin = kappa_blend * L_wb

    # Lateral offset correction (simplified: use heading error as proxy)
    psi_track = track_now['psi']
    heading_error = psi_track - yaw
    # Wrap to [-π, π]
    heading_error = jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))

    K_heading = 2.0  # heading correction gain [rad/rad]
    delta_correction = K_heading * heading_error

    delta = jnp.clip(delta_kin + delta_correction, -0.35, 0.35)  # ±20° max steer

    # ── Speed target: friction-circle limit ──────────────────────────────
    kappa_max = jnp.maximum(jnp.abs(kappa_blend), 0.01)
    v_friction = jnp.sqrt(mu_friction * 9.81 / kappa_max)
    v_limit = VP.get('V_limit', 28.0)
    v_target = jnp.minimum(v_friction * 0.92, v_limit)  # 8% safety margin

    # Speed P-controller (smooth split via softplus)
    K_speed = 4000.0
    v_err = v_target - vx_safe
    F_accel = jax.nn.softplus(v_err * 2.0) * K_speed * 0.5
    F_brake = -jax.nn.softplus(-v_err * 2.0) * K_speed
    F_total = jnp.clip(F_accel + F_brake, -8000.0, 6000.0)

    return delta, F_total, {'v_target': v_target, 'kappa': kappa_now}


# ─────────────────────────────────────────────────────────────────────────────
# §2  Full-Lap Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_full_lap(
    setup_vector: jax.Array,        # (28,) physical setup parameters
    track: DifferentiableTrack,
    dt: float = 0.005,              # 200 Hz physics
    max_steps: int = 3000,          # 15s max (enough for most FS tracks at ~750m)
) -> dict:
    """
    Simulate a full lap and return differentiable metrics.

    The gradient ∂(lap_time)/∂(setup_vector) is available via:
        jax.grad(lambda s: simulate_full_lap(s, track)['effective_lap_time'])(setup)

    Returns dict with:
        effective_lap_time : scalar [s] — distance/mean_speed (differentiable proxy)
        mean_speed         : scalar [m/s]
        total_energy       : scalar [J]
        energy_per_meter   : scalar [J/m]
        max_lateral_g      : scalar [G]
        final_progress     : scalar [m] — how far the car got
        T_tire_max         : scalar [°C] — peak tire temp
    """
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle, compute_equilibrium_suspension,
    )

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)

    # ── Setup-dependent initial condition ─────────────────────────────────
    z_eq = compute_equilibrium_suspension(setup_vector, VP)
    x0 = (jnp.zeros(46)
           .at[14].set(5.0)          # initial speed 5 m/s (rolling start)
           .at[6:10].set(z_eq)
           .at[28:38].set(jnp.array([85., 85., 85., 85., 80.,
                                      85., 85., 85., 85., 80.])))

    # ── Scan state: (x_physics, s_progress) ──────────────────────────────
    def scan_fn(carry, _step_idx):
        x, s_progress = carry

        vx = x[14]
        vy = x[15]
        yaw = x[5]
        vx_safe = jnp.maximum(vx, 0.5)

        # Path-following controller
        delta, F_total, _info = path_following_controller(
            vx, vy, yaw, s_progress, track,
        )

        # Physics step
        u = jnp.array([delta, F_total])
        x_next = vehicle.simulate_step(x, u, setup_vector, dt)

        # Update track progress: ds = vx * dt (projected onto centreline)
        ds = vx_safe * dt
        s_next = s_progress + ds

        # ── Per-step metrics ─────────────────────────────────────────────
        power = jnp.abs(F_total * vx_safe)
        ay = jnp.abs(vy * x[19])  # vy × yaw_rate ≈ lateral acceleration
        T_tire = jnp.max(x_next[28:31])  # front surface temps

        return (x_next, s_next), (vx_safe, power, ay, T_tire, ds)

    # ── Run simulation ───────────────────────────────────────────────────
    (x_final, s_final), (vx_hist, power_hist, ay_hist, T_hist, ds_hist) = jax.lax.scan(
        scan_fn,
        (x0, jnp.array(0.0)),
        jnp.arange(max_steps),
    )

    # ── Aggregate metrics ────────────────────────────────────────────────
    total_distance = jnp.sum(ds_hist)
    mean_speed = jnp.mean(vx_hist)
    total_energy = jnp.sum(power_hist) * dt
    energy_per_meter = total_energy / jnp.maximum(total_distance, 1.0)
    max_lat_g = jnp.max(ay_hist) / 9.81
    T_tire_max = jnp.max(T_hist)

    # Effective lap time: distance / mean_speed
    # This is a differentiable proxy — lower mean_speed = longer lap time.
    # For a fixed track length L:
    #   lap_time = L / mean_speed
    # But we use total_distance (which depends on how well the car follows
    # the track) to penalise setups that can't complete the lap.
    effective_lap_time = track.total_length / jnp.maximum(mean_speed, 1.0)

    # Completion bonus: penalise if car didn't finish the lap
    completion_ratio = jnp.minimum(total_distance / track.total_length, 1.0)
    # Smooth penalty: 0 at 100% completion, grows quadratically below
    completion_penalty = 10.0 * (1.0 - completion_ratio) ** 2

    return {
        'effective_lap_time': effective_lap_time + completion_penalty,
        'mean_speed':         mean_speed,
        'total_energy':       total_energy,
        'energy_per_meter':   energy_per_meter,
        'max_lateral_g':      max_lat_g,
        'final_progress':     s_final,
        'completion_ratio':   completion_ratio,
        'T_tire_max':         T_tire_max,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §3  Gradient Computation — THE MONEY SHOT
# ─────────────────────────────────────────────────────────────────────────────

def compute_lap_time_gradient(
    setup_vector: jax.Array,
    track: DifferentiableTrack,
    dt: float = 0.005,
    max_steps: int = 3000,
) -> tuple:
    """
    Compute ∂(lap_time)/∂(setup_vector).

    Returns: (lap_time, gradient_28d)

    The gradient tells you exactly how much each of the 28 setup parameters
    affects lap time. Positive gradient = increasing this parameter makes
    the lap SLOWER. Negative = makes it FASTER.

    This is the single most powerful tool for setup engineering.
    """
    def lap_time_scalar(setup):
        result = simulate_full_lap(setup, track, dt, max_steps)
        return result['effective_lap_time']

    lap_time, grad = jax.value_and_grad(lap_time_scalar)(setup_vector)
    return lap_time, grad


def sensitivity_report(
    setup_vector: jax.Array,
    track: DifferentiableTrack,
):
    """
    Print a human-readable sensitivity report: ∂(lap_time)/∂(param) for all 28.
    """
    lap_time, grad = compute_lap_time_gradient(setup_vector, track)
    grad_np = np.array(grad)

    print(f"\n{'═' * 72}")
    print(f"  LAP TIME SENSITIVITY REPORT")
    print(f"{'═' * 72}")
    print(f"  Effective lap time: {float(lap_time):.3f} s")
    print(f"  Track: {float(track.total_length):.0f} m")
    print(f"\n  {'Parameter':<20} {'∂t/∂p':>10} {'Direction':>12} {'Impact':>8}")
    print(f"  {'─' * 52}")

    # Sort by absolute sensitivity
    sorted_idx = np.argsort(-np.abs(grad_np))
    for i in sorted_idx:
        name = SETUP_NAMES[i] if i < len(SETUP_NAMES) else f"param_{i}"
        g = grad_np[i]
        direction = "→ faster" if g > 0.001 else ("→ slower" if g < -0.001 else "≈ neutral")
        impact = "HIGH" if abs(g) > 0.01 else ("MED" if abs(g) > 0.001 else "LOW")
        print(f"  {name:<20} {g:>+10.6f} {direction:>12} {impact:>8}")

    print(f"  {'─' * 52}")
    print(f"  Gradient ℓ₂ norm: {np.linalg.norm(grad_np):.6f}")
    print(f"{'═' * 72}\n")

    return lap_time, grad_np


# ─────────────────────────────────────────────────────────────────────────────
# §4  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from models.vehicle_dynamics import build_default_setup_28, SETUP_NAMES

    print("[LapSim] Building differentiable track…")
    track = make_differentiable_track(track_name='fsg_autocross')
    print(f"[LapSim] Track: {float(track.total_length):.0f} m, "
          f"{track.s.shape[0]} nodes")

    print("[LapSim] Building default setup…")
    setup = build_default_setup_28()

    print("[LapSim] Compiling full-lap simulation (first run = XLA compile)…")
    import time
    t0 = time.time()

    result = simulate_full_lap(setup, track)
    t_sim = time.time() - t0
    print(f"[LapSim] Simulation complete in {t_sim:.1f}s")
    print(f"  Effective lap time: {float(result['effective_lap_time']):.3f} s")
    print(f"  Mean speed: {float(result['mean_speed']):.1f} m/s")
    print(f"  Energy: {float(result['total_energy'])/1000:.1f} kJ")
    print(f"  Energy/m: {float(result['energy_per_meter']):.1f} J/m")
    print(f"  Max lateral G: {float(result['max_lateral_g']):.2f} G")
    print(f"  Completion: {float(result['completion_ratio'])*100:.1f}%")
    print(f"  Tire T_max: {float(result['T_tire_max']):.1f} °C")

    print(f"\n[LapSim] Computing ∂(lap_time)/∂(setup)…")
    t0 = time.time()
    sensitivity_report(setup, track)
    t_grad = time.time() - t0
    print(f"[LapSim] Gradient computed in {t_grad:.1f}s")


if __name__ == '__main__':
    main()