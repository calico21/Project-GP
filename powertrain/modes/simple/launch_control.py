# powertrain/modes/simple/launch_control.py
# Project-GP — Simple Launch Sequencer (TC-Guarded Mode 2.0)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Design philosophy: dead-reliable, zero exotic dependencies, auditable.
# Acts exclusively as a Torque Profile Generator. Relies on the downstream 
# PID Traction Control (TC) to handle friction limits and slip targets.
#
# Architecture contract (TC-Guarded pipeline):
#   T_launch_base = simple_launch_step(...)      # Aggressive Quintic Ramp
#   T_commanded   = tc_simple(T_launch_base,...) # Slip capture & emergency cut
#
# Launch sequence (simple mode):
#   IDLE  ──(brake > 0.5 AND throttle > 0.5)──► ARMED
#   ARMED ──(brake < 0.3 AND throttle > 0.9)──► LAUNCH
#   LAUNCH──(brake > 0.5 OR throttle < 0.1)───► IDLE (Abort)
#
# Torque profile: quintic polynomial ramp (C² continuous, zero jerk at endpoints).
# Load Transfer Bias: Automatically biases torque rearward (e.g., 30/70) during 
# launch to maximize utilization of dynamic Fz shift.
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
SLC_IDLE   = 0
SLC_ARMED  = 1
SLC_LAUNCH = 2


# ─────────────────────────────────────────────────────────────────────────────
# §2  Configuration
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLCParams(NamedTuple):
    """
    Simple launch control tuning — all XLA compile-time constants.
    """
    # ── Arming thresholds ─────────────────────────────────────────────────
    brake_arm_threshold: float    = 0.50   # brake fraction to arm
    throttle_arm_threshold: float = 0.50   # throttle fraction to arm
    brake_launch_threshold: float = 0.30   # brake below this → launch
    throttle_launch_gate: float   = 0.90   # throttle above this required
    
    # ── Abort thresholds ──────────────────────────────────────────────────
    throttle_abort: float         = 0.10   # lift-off aborts launch
    brake_abort: float            = 0.20   # tapping brake aborts launch
    
    # ── Profile ──────────────────────────────────────────────────────────
    # T_peak specifies the absolute max torque requested per motor.
    # We set this high, letting the TC cut it down to reality.
    T_peak_wheel: float           = 30.0   # [Nm] per wheel (example limit)
    t_ramp_end: float             = 0.50   # [s] fast ramp to hit the TC wall quickly
    
    # ── Drive config (Load Transfer Bias) ────────────────────────────────
    # During extreme acceleration, ~70% of vehicle weight shifts to the rear.
    # 0.0 = RWD; 0.30 = 30% Front / 70% Rear (Optimal AWD Launch Bias)
    front_torque_fraction_launch: float = 0.30 


# ─────────────────────────────────────────────────────────────────────────────
# §3  State
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLCState(NamedTuple):
    phase: jax.Array         # scalar int32
    t_phase_start: jax.Array # scalar float: time current phase began [s]
    t_current: jax.Array     # scalar float: accumulated simulation time [s]

    @classmethod
    def default(cls) -> "SimpleLCState":
        return cls(
            phase=jnp.array(SLC_IDLE, dtype=jnp.int32),
            t_phase_start=jnp.array(0.0),
            t_current=jnp.array(0.0),
        )

def make_simple_lc_state() -> SimpleLCState:
    return SimpleLCState.default()


# ─────────────────────────────────────────────────────────────────────────────
# §4  Output
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLCOutput(NamedTuple):
    T_base_request: jax.Array   # (4,) wheel torques [Nm] (To feed into TC)
    phase: jax.Array            # scalar: current phase code
    is_launch_active: jax.Array # scalar float: 1.0 during LAUNCH
    profile_value: jax.Array    # scalar: ramp value [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# §5  Quintic Polynomial Ramp
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def _quintic_ramp(t: jax.Array, t_ramp: float) -> jax.Array:
    """
    Quintic Hermite from 0→1 over [0, t_ramp], plateau at 1.0 after.
    Boundary conditions: f(0)=0, f'(0)=0, f''(0)=0, f(1)=1, f'(1)=0, f''(1)=0.
    """
    s = jnp.clip(t / (t_ramp + 1e-6), 0.0, 1.0)
    return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3


# ─────────────────────────────────────────────────────────────────────────────
# §6  Torque Distribution (Dynamic Bias)
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def _distribute_torque_biased(
    profile_val: jax.Array,
    T_max_hw: jax.Array,
    params: SimpleLCParams,
    is_rwd: bool = False
) -> jax.Array:
    """
    Shifts torque rearward based on the front_torque_fraction_launch parameter
    to mathematically exploit dynamic load transfer during the launch phase.
    """
    front_frac = 0.0 if is_rwd else params.front_torque_fraction_launch
    rear_frac = 1.0 - front_frac
    
    # We calculate the total available torque limit, then partition it.
    T_total_max = params.T_peak_wheel * 4.0 
    
    T_front_axle = T_total_max * front_frac * profile_val
    T_rear_axle  = T_total_max * rear_frac * profile_val

    T_cmd = jnp.array([
        T_front_axle * 0.5, # FL
        T_front_axle * 0.5, # FR
        T_rear_axle  * 0.5, # RL
        T_rear_axle  * 0.5, # RR
    ])
    
    return jnp.minimum(T_cmd, T_max_hw)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Main Step
# ─────────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnames=('is_rwd',))
def simple_launch_step(
    throttle: jax.Array,         # [0, 1]
    brake: jax.Array,            # [0, 1]
    T_max_hw: jax.Array,         # (4,) hardware torque ceiling [Nm]
    lc_state: SimpleLCState,
    dt: jax.Array,
    params: SimpleLCParams = SimpleLCParams(),
    is_rwd: bool = False,
) -> Tuple[SimpleLCOutput, SimpleLCState]:
    """
    Single-step simple launch controller.
    
    Outputs T_base_request, which the Powertrain Manager must immediately pipe
    into the Traction Control node to form the final inverter command.
    """
    phase   = lc_state.phase
    t_start = lc_state.t_phase_start
    t       = lc_state.t_current

    # ── §7.1 Phase transitions ─────────────────────────────────────────────
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

    # Hard abort: heavy braking or lift-off throttle
    abort = (phase >= SLC_LAUNCH) & ((brake > params.brake_abort) | (throttle < params.throttle_abort))
    phase = jnp.where(abort, jnp.array(SLC_IDLE, dtype=jnp.int32), phase)

    # ── §7.2 Torque computation ────────────────────────────────────────────
    t_launch_elapsed = jnp.maximum(t - t_start, 0.0)

    profile_val = _quintic_ramp(t_launch_elapsed, params.t_ramp_end)
    T_launch_biased = _distribute_torque_biased(profile_val, T_max_hw, params, is_rwd)

    is_launch = (phase == SLC_LAUNCH).astype(jnp.float32)
    
    T_idle = jnp.zeros(4)
    T_out = jnp.where(is_launch > 0.5, T_launch_biased, T_idle)

    output = SimpleLCOutput(
        T_base_request=T_out,
        phase=phase,
        is_launch_active=is_launch,
        profile_value=profile_val,
    )
    
    new_state = SimpleLCState(
        phase=phase,
        t_phase_start=t_start,
        t_current=t + dt,
    )
    
    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §8  Standalone Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
def smoke_test():
    """Validates phase transitions and load transfer biased torque allocation."""
    import time

    params = SimpleLCParams(T_peak_wheel=30.0)
    state  = SimpleLCState.default()
    dt     = jnp.array(0.01)
    T_max  = jnp.array(21.0) # Hardware limit per motor
    T_max_arr = jnp.full(4, T_max)

    print("Compiling TC-Guarded Launch Control...")
    t0 = time.perf_counter()
    
    # Simulate ARMED phase
    out, state = simple_launch_step(
        throttle=jnp.array(1.0), brake=jnp.array(1.0), 
        T_max_hw=T_max_arr, lc_state=state, dt=dt, params=params
    )
    _ = float(out.phase)
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"  Compile: {compile_ms:.1f} ms")
    print(f"  Phase after brake+throttle: {int(out.phase)} (Expected: 1/ARMED)")

    # Simulate LAUNCH trigger (Brake release)
    print("\nSimulating Launch Release (0 to 0.6 seconds):")
    brake_release = jnp.array(0.0)
    throttle_full = jnp.array(1.0)
    
    for step in range(60):
        out, state = simple_launch_step(
            throttle=throttle_full, brake=brake_release, 
            T_max_hw=T_max_arr, lc_state=state, dt=dt, params=params
        )
        if step % 10 == 0:
            print(f"  t={step*0.01:.2f}s | Profile: {float(out.profile_value):.2f} | T_cmd: {[round(float(x), 1) for x in out.T_base_request]}")

    # Check the static 30/70 split at full profile
    front_requested = float(out.T_base_request[0] + out.T_base_request[1])
    rear_requested  = float(out.T_base_request[2] + out.T_base_request[3])
    total_requested = front_requested + rear_requested
    
    print(f"\nFinal Distribution Analysis (Profile = 1.0):")
    print(f"  Total Torque: {total_requested:.1f} Nm")
    print(f"  Front Axle:   {front_requested:.1f} Nm ({front_requested/total_requested*100:.1f}%)")
    print(f"  Rear Axle:    {rear_requested:.1f} Nm ({rear_requested/total_requested*100:.1f}%)")

    assert int(out.phase) == SLC_LAUNCH, "Should be in LAUNCH phase"
    assert round(front_requested/total_requested, 2) == 0.30, "Load transfer bias must match parameter (0.30)"

    print("\n✅ All assertions passed. Ready to feed into Traction Control.")

if __name__ == '__main__':
    smoke_test()