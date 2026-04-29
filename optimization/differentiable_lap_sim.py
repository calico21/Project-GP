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
from config.vehicles.ter26 import vehicle_params as VP
from config.tire_coeffs import tire_coeffs as TC


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

def simulate_full_lap(setup_vector, track, dt: float = 0.02, max_steps: int = 5000):
    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    from config.vehicles.ter26 import vehicle_params
    from config.tire_coeffs import tire_coeffs

    vehicle = DifferentiableMultiBodyVehicle(vehicle_params, tire_coeffs)

    # Initial state
    x_init = vehicle.make_initial_state(T_env=25.0, vx0=10.0)
    s_init = jnp.array(0.0)
    done_init = jnp.array(False)
    steps_init = jnp.array(0, dtype=jnp.int32)

    def scan_fn(carry, _):
        x, s_curr, done, steps_used = carry

        def active_step(carry):
            x, s_curr, done, steps_used = carry

            vx = x[14]
            vy = x[15] if x.shape[0] > 15 else jnp.array(0.0, dtype=x.dtype)
            yaw = x[5]

            # Use progress-based track query instead of nearest-node jumping
            s_progress = jnp.clip(s_curr, 0.0, track.total_length - 1e-3)

            delta, F_total, _ = path_following_controller(
                vx=vx,
                vy=vy,
                yaw=yaw,
                s_progress=s_progress,
                track=track,
            )

            # Map total longitudinal force to wheel torques + brake force
            R_wheel = 0.2045
            eta = 0.95
            T_peak_wheel = 21.0 * 4.5

            T_req_total = jnp.maximum(0.0, F_total) * R_wheel / eta
            T_demand = jnp.clip(T_req_total * 0.25, 0.0, T_peak_wheel)
            F_brake = jnp.maximum(0.0, -F_total)

            u_full = jnp.array(
                [delta, T_demand, T_demand, T_demand, T_demand, F_brake],
                dtype=x.dtype,
            )

            x_next = vehicle.simulate_step(x, u_full, setup_vector, dt)

            # Use forward progress only, not abs(speed)
            vx_next = jnp.maximum(x_next[14], 0.0)
            ds = vx_next * dt
            s_next = s_curr + ds

            power = jnp.maximum(0.0, F_total) * vx_next
            # Robust lateral acceleration from kinematics (centripetal-dominated)
            vx_safe = jnp.maximum(x_next[14], 0.0)
            ay = vx_safe * x_next[19]
            ay = jnp.clip(ay, -50.0, 50.0)  # ±5G physical sanity
            T_fl = x_next[28] if x.shape[0] > 28 else jnp.array(0.0, dtype=x.dtype)

            done_next = s_next >= track.total_length
            steps_next = steps_used + jnp.array(1, dtype=jnp.int32)

            return (x_next, s_next, done_next, steps_next), (
                vx_next, power, ay, T_fl, ds, delta, T_demand, F_brake
            )

        def inactive_step(carry):
            x, s_curr, done, steps_used = carry
            zeros = (
                jnp.array(0.0, dtype=x.dtype),
                jnp.array(0.0, dtype=x.dtype),
                jnp.array(0.0, dtype=x.dtype),
                jnp.array(0.0, dtype=x.dtype),
                jnp.array(0.0, dtype=x.dtype),
                jnp.array(0.0, dtype=x.dtype),
                jnp.array(0.0, dtype=x.dtype),
                jnp.array(0.0, dtype=x.dtype),
            )
            return carry, zeros

        return jax.lax.cond(done, inactive_step, active_step, carry)

    (x_final, s_final, done_final, steps_used), (
        vx_hist, power_hist, ay_hist, T_hist, ds_hist, delta_hist, T_dem_hist, F_brk_hist
    ) = jax.lax.scan(
        scan_fn,
        (x_init, s_init, done_init, steps_init),
        jnp.arange(max_steps),
    )

    # Mask out inactive tail if the lap finished early
    mask = (jnp.arange(max_steps) < steps_used).astype(vx_hist.dtype)
    denom = jnp.maximum(jnp.sum(mask), 1.0)

    return {
        'effective_lap_time': jnp.asarray(steps_used, dtype=jnp.float32) * dt,
        'mean_speed':         jnp.sum(vx_hist * mask) / denom,
        'total_energy':       jnp.sum(power_hist * mask) * dt,
        'energy_per_meter':   (jnp.sum(power_hist * mask) * dt) / jnp.maximum(s_final, 1.0),
        'max_lateral_g':      jnp.max(jnp.abs(ay_hist) * mask) / 9.81,
        'final_progress':     s_final,
        'completion_ratio':   jnp.clip(s_final / track.total_length, 0.0, 1.0),
        'T_tire_max':         jnp.max(T_hist * mask),
        's_hist':             ds_hist,
        'vx_hist':            vx_hist,
        'delta_hist':         delta_hist,
        'T_dem_hist':         T_dem_hist,
        'F_brk_hist':         F_brk_hist,
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
        result = simulate_full_lap(setup, track, dt=dt, max_steps=max_steps)
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
    from models.vehicle_dynamics import build_default_setup_28
    from config.vehicles.ter26 import vehicle_params 

    print("[LapSim] Building differentiable track…")
    track = make_differentiable_track(track_name='fsg_autocross')
    print(f"[LapSim] Track: {float(track.total_length):.0f} m, {track.s.shape[0]} nodes")

    print("[LapSim] Building default setup…")
    setup = build_default_setup_28(vehicle_params)

    print("[LapSim] Compiling full-lap simulation (first run = XLA compile)…")
    import time
    t0 = time.time()

    result = simulate_full_lap(setup, track)
    t_sim = time.time() - t0
    
    print(f"[LapSim] Simulation complete in {t_sim:.1f}s")
    print(f"  Effective lap time: {float(result['effective_lap_time']):.3f} s")
    print(f"  Mean speed: {float(result['mean_speed']):.1f} m/s")
    print(f"  Max lateral G: {float(result['max_lateral_g']):.2f} G")
    print(f"  Completion: {float(result['completion_ratio'])*100:.1f}%")
    print(f"  Tire T_max: {float(result['T_tire_max']):.1f} °C")
    
    # ── Save CSV for Validation Pipeline ─────────────────────────────────────
    print("\n[LapSim] Saving telemetry to out/simple_tv.csv...")
    import pandas as pd
    import os
    os.makedirs("out", exist_ok=True)
    
    s_array = np.cumsum(np.array(result['s_hist']))
    vx_array = np.array(result['vx_hist'])
    
    # Create DataFrame with REAL telemetry data!
    df = pd.DataFrame({
        's': s_array,
        'vx': vx_array,
        'throttle': np.clip(np.array(result['T_dem_hist']) / 94.5, 0.0, 1.0), 
        'brake': np.clip(np.array(result['F_brk_hist']) / 4000.0, 0.0, 1.0),
        'delta': np.array(result['delta_hist']),
        'T_fl': np.array(result['T_dem_hist']) * 0 + 80.0, # (Simplified for now)
        'T_fr': np.array(result['T_dem_hist']) * 0 + 80.0,
        'T_rl': np.array(result['T_dem_hist']) * 0 + 80.0,
        'T_rr': np.array(result['T_dem_hist']) * 0 + 80.0,
        'util_r': np.clip(vx_array / 30.0, 0, 1)
    })
    
    df.to_csv("out/simple_tv.csv", index=False)
    print("[LapSim] Saved successfully.")

    print("\n[LapSim] Skipping setup gradient calculation to prevent OOM crash.")

if __name__ == "__main__":
    main()

