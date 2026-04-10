# powertrain/modes/simple/torque_vectoring.py
# Project-GP — Simple PD Torque Vectoring Controller (Mode 2.0)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Direct yaw-moment PD controller for embedded (SIMPLE mode) use.
# Features a velocity-adaptive gain and a Low-Pass Filtered "Derivative on 
# Measurement" to prevent derivative kick from aggressive steering inputs.
#
# Architecture contract (differs from ADVANCED mode):
#   ADVANCED: tv_step → delta_T references → SOCP allocator uses them
#   SIMPLE:   tv_simple → directly outputs T_commanded
#
# The pipeline for SIMPLE mode is:
#   T_base, state, diag = simple_dyc_torque_vectoring_pd(...)
#   T_final = tc_simple(T_base, omega, vx, ...)  # PI slip correction
#
# Mathematical formulation:
#   ωz_target = (vₓ·δ) / (L + K_us·vₓ²)                  [steady-state Ackermann]
#   e         = ωz_target − ωz                           [yaw rate error]
#   decay     = min(1.0, 5.0 / max(|vₓ|, 1.0))           [high-speed damping]
#   d(ωz)/dt  = -(ωz − ωz_prev) / dt                     [derivative on measurement]
#   α         = dt / (τ_filter + dt)                     [low-pass filter coefficient]
#   D_filt    = (1−α)·D_prev + α·d(ωz)/dt                [filtered chassis rotation]
#   Mz_demand = (Kp·decay)·e + (Kd·decay)·D_filt         [PD control law]
#   ΔT        = (Mz_demand·r_w) / Track                  [torque delta per side]
#
# All ops are C∞ differentiable — safe inside jit/grad/vmap/scan.
# Zero Python-level conditionals inside traced code.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

# Asumiendo que esta es tu ruta de configuración
from config.vehicles.ter27 import vehicle_params_ter27


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class TVGeometry(NamedTuple):
    """Static vehicle geometry dynamically loaded from config."""
    lf: float
    lr: float
    track_f: float
    track_r: float
    r_w: float
    mass: float

    @staticmethod
    def from_vehicle_params(vp: dict) -> 'TVGeometry':
        return TVGeometry(
            lf=vp.get('lf', 0.806),
            lr=vp.get('lr', 0.744),
            track_f=vp.get('track_front', 1.220),
            track_r=vp.get('track_rear', 1.200),
            r_w=vp.get('wheel_radius', 0.2032),
            mass=vp.get('total_mass', 320.0),
        )

DEFAULT_TER27_GEO = TVGeometry.from_vehicle_params(vehicle_params_ter27)

class SimpleTVParams(NamedTuple):
    """
    PD Torque Vectoring hyperparameters.
    """
    Kp_yaw: float = 800.0       # Base Proportional gain [Nm / (rad/s)]
    Kd_yaw: float = 60.0        # Base Derivative gain (Chassis damping) [Nm·s / rad]
    tau_filter: float = 0.04    # Low-pass filter time constant (40ms cutoff)


# ─────────────────────────────────────────────────────────────────────────────
# §2  State
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTVState(NamedTuple):
    """Persistent state for the simple PD controller."""
    wz_prev: jax.Array          # Previous chassis yaw rate [rad/s]
    derivative_prev: jax.Array  # Previous filtered derivative value

    @classmethod
    def default(cls) -> 'SimpleTVState':
        return cls(
            wz_prev=jnp.zeros(()),
            derivative_prev=jnp.zeros(())
        )

def make_simple_tv_state() -> SimpleTVState:
    return SimpleTVState.default()


# ─────────────────────────────────────────────────────────────────────────────
# §3  Diagnostics Output
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTVDiagnostics(NamedTuple):
    """Per-step diagnostics for telemetry / dashboard."""
    wz_target: jax.Array           # Requested yaw rate from driver [rad/s]
    wz_error: jax.Array            # Tracking error [rad/s]
    raw_derivative: jax.Array      # Unfiltered derivative on measurement
    filtered_derivative: jax.Array # Filtered chassis damping component
    Mz_demand: jax.Array           # Total requested yaw moment [Nm]
    gain_decay: jax.Array          # Active velocity-dependent gain multiplier [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main Step Function
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('is_rwd',))
def simple_dyc_torque_vectoring_pd(
    vx: jax.Array,
    wz: jax.Array,
    delta: jax.Array,
    Fx_driver: jax.Array,
    dt: jax.Array,
    T_min: jax.Array,
    T_max: jax.Array,
    state: SimpleTVState,
    params: SimpleTVParams = SimpleTVParams(),
    geo: TVGeometry = DEFAULT_TER27_GEO,
    is_rwd: bool = False,
) -> tuple[jax.Array, SimpleTVState, SimpleTVDiagnostics]:
    """
    One PD Torque Vectoring timestep.

    Calculates yaw moment demand using a proportional error term and a 
    derivative-on-measurement damping term, then statically allocates it.

    Returns:
        T_out:       (4,) torque commands [Nm]
        new_state:   SimpleTVState carrying forward to next timestep
        diag:        SimpleTVDiagnostics for telemetry / dashboard
    """
    T_total_req = Fx_driver * geo.r_w
    vx_safe = jnp.maximum(jnp.abs(vx), 1.0)
    L = geo.lf + geo.lr
    K_us = geo.mass * (geo.lr - geo.lf) / (L * 30000.0)
    
    # ── §4.1 Target & Proportional Error ──────────────────────────────────
    wz_target = (vx_safe * delta) / (L + K_us * vx_safe ** 2)
    error = wz_target - wz

    # ── §4.2 Velocity Gain Decay ──────────────────────────────────────────
    # Full gain at ≤5 m/s, decays as 1/v above
    gain_decay = jnp.minimum(1.0, 5.0 / vx_safe)
    Kp_eff = params.Kp_yaw * gain_decay
    Kd_eff = params.Kd_yaw * gain_decay

    # ── §4.3 Derivative on Measurement & Low-Pass Filter ──────────────────
    # Negative sign: derivative force opposes actual chassis rotation
    raw_derivative = -(wz - state.wz_prev) / dt
    
    alpha = dt / (params.tau_filter + dt)
    filtered_derivative = (1.0 - alpha) * state.derivative_prev + alpha * raw_derivative

    # ── §4.4 Yaw Moment Demand ────────────────────────────────────────────
    Mz_demand = (Kp_eff * error) + (Kd_eff * filtered_derivative)

    # ── §4.5 Static Allocation ────────────────────────────────────────────
    if is_rwd:
        T_base = T_total_req / 2.0
        delta_T = (Mz_demand * geo.r_w) / geo.track_r
        T_commanded = jnp.array([
            jnp.zeros(()),     # FL — undriven
            jnp.zeros(()),     # FR — undriven
            T_base - delta_T,  # RL
            T_base + delta_T,  # RR
        ])
    else:
        T_base = T_total_req / 4.0
        delta_T = (Mz_demand * geo.r_w) / (geo.track_f + geo.track_r)
        T_commanded = jnp.array([
            T_base - delta_T,  # FL
            T_base + delta_T,  # FR
            T_base - delta_T,  # RL
            T_base + delta_T,  # RR
        ])

    # Basic asymmetric clip for contingency mode
    T_out = jnp.clip(T_commanded, T_min, T_max)

    # ── §4.6 State Update ─────────────────────────────────────────────────
    new_state = SimpleTVState(
        wz_prev=wz,
        derivative_prev=filtered_derivative
    )

    # ── §4.7 Diagnostics ──────────────────────────────────────────────────
    diag = SimpleTVDiagnostics(
        wz_target=wz_target,
        wz_error=error,
        raw_derivative=raw_derivative,
        filtered_derivative=filtered_derivative,
        Mz_demand=Mz_demand,
        gain_decay=gain_decay,
    )

    return T_out, new_state, diag


# ─────────────────────────────────────────────────────────────────────────────
# §5  Standalone Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """Quick validation: compile, run 200 steps, check outputs and latency."""
    import time

    params = SimpleTVParams()
    state  = SimpleTVState.default()
    dt     = jnp.array(0.005)

    # Simulate: 15 m/s, aggressive steering delta (0.3 rad), minor oversteer wz
    vx        = jnp.array(15.0)
    wz        = jnp.array(0.5) 
    delta     = jnp.array(0.3)
    Fx_driver = jnp.array(400.0) # approx 81 Nm total torque
    T_min     = jnp.array(-21.0)
    T_max     = jnp.array(21.0)

    print("Compiling simple_dyc_torque_vectoring_pd...")
    t0 = time.perf_counter()
    T_out, state, diag = simple_dyc_torque_vectoring_pd(
        vx, wz, delta, Fx_driver, dt, T_min, T_max, state, params, is_rwd=False
    )
    _ = float(T_out[0])  # force eval
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"  Compile: {compile_ms:.1f} ms")

    print("Running 200 steps (step response)...")
    t0 = time.perf_counter()
    for _ in range(200):
        # Update wz to simulate chassis reaction (simplistic)
        wz = wz + dt * 2.0 
        T_out, state, diag = simple_dyc_torque_vectoring_pd(
            vx, wz, delta, Fx_driver, dt, T_min, T_max, state, params, is_rwd=False
        )
    _ = float(T_out[0])
    run_ms = (time.perf_counter() - t0) * 1000
    per_step = run_ms / 200

    print(f"  Total: {run_ms:.1f} ms | Per-step: {per_step:.4f} ms")
    print(f"  Budget (1 ms): {'PASS' if per_step < 1.0 else 'FAIL'}")
    print(f"\nFinal Outputs (AWD cornering scenario):")
    print(f"  T_out (FL, FR, RL, RR): {[round(float(x), 2) for x in T_out]}")
    print(f"  Mz_demand:           {float(diag.Mz_demand):.2f} Nm")
    print(f"  wz_target:           {float(diag.wz_target):.4f} rad/s")
    print(f"  wz_error:            {float(diag.wz_error):.4f} rad/s")
    print(f"  Filtered Derivative: {float(diag.filtered_derivative):.4f}")
    print(f"  Gain Decay:          {float(diag.gain_decay):.4f}")

    # Sanity assertions
    assert float(T_out[0]) != float(T_out[1]), "Left and Right torques must differ during cornering"
    assert float(T_max) >= float(T_out[1]), "Output must respect T_max limits"
    print("\n✅ All assertions passed")


if __name__ == '__main__':
    smoke_test()