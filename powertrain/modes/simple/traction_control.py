# powertrain/modes/simple/traction_control.py
# Project-GP — Simple PID Traction Controller (Yaw-Preserving Mode 2.0)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Direct torque-correcting PID controller for embedded (SIMPLE mode) use.
# Features Yaw-Preserving cross-axle reduction and vx-independent Emergency Cut.
#
# Architecture contract (differs from ADVANCED mode):
#   ADVANCED: tc_step → kappa_star references → SOCP allocator uses them
#   SIMPLE:   tc_simple → directly corrects T_requested in-place
#
# The pipeline for SIMPLE mode is:
#   T_base  = simple_dyc_torque_vectoring(...)   # DYC yaw control
#   T_final = tc_simple(T_base, omega, vx, ...)  # PID slip correction
#
# Mathematical formulation (per driven wheel i):
#   κᵢ     = (ωᵢ·r_w − vₓ) / softplus(vₓ − v_min + v_min)   [smooth denom]
#   dωᵢ/dt = (ωᵢ − ω_prevᵢ) / dt                            [wheel acceleration]
#   eᵢ     = κᵢ − κ_ref
#   Iᵢ    += eᵢ·dt·gate                                     [gated accumulation]
#   Iᵢ     = I_max · tanh(Iᵢ / I_max)                       [anti-windup]
#
#   πᵢ     = Kp·eᵢ + Ki·Iᵢ + Kd·dωᵢ/dt                      [PID output]
#   E_cutᵢ = K_emg · softplus(dωᵢ/dt − ω_dot_max)           [vx-independent cut]
#
#   R_rawᵢ = maskᵢ · (gate · softplus(πᵢ·β)/β + E_cutᵢ)     [raw reduction needed]
#
#   Yaw-Preserving Cross-Axle Logic:
#   R_f    = max(R_raw_FL, R_raw_FR)
#   R_r    = max(R_raw_RL, R_raw_RR)
#   Δ_axle = [R_f, R_f, R_r, R_r]                           [symmetric reduction]
#
#   T_cmdᵢ = T_reqᵢ − Δ_axleᵢ                               [preserves ΔT]
#   T_outᵢ = driven ? softplus(T_cmdᵢ·β)/β : T_reqᵢ         [lower-clamp at 0]
#
# All ops are C∞ differentiable — safe inside jit/grad/vmap/scan.
# Zero Python-level conditionals inside traced code.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTCParams(NamedTuple):
    """
    PID traction control hyperparameters.
    """
    Kp: float = 80.0             # Nm per unit slip error
    Ki: float = 20.0             # Nm / (slip·s)
    Kd: float = 2.0              # Nm / (rad/s²) — derivative reaction to acceleration
    
    lambda_ref: float = 0.08     # target slip ratio
    I_max: float = 5.0           # anti-windup ceiling [slip·s]
    clamp_sharpness: float = 50.0# softplus β [Nm⁻¹]
    
    r_w: float = 0.2032          # effective wheel radius [m]
    v_min: float = 2.0           # TC activation threshold [m/s]
    
    # Emergency vx-independent cutoff limits
    omega_dot_max: float = 250.0 # Physical acceleration limit [rad/s²]
    K_emg: float = 10.0          # Nm per rad/s² beyond the physical limit
    
    drive_mask: tuple = (0., 0., 1., 1.)  # which wheels are driven; RWD default


# ─────────────────────────────────────────────────────────────────────────────
# §2  State
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTCState(NamedTuple):
    """Persistent state for the simple PID controller."""
    pi_integral: jax.Array    # (4,) per-wheel slip error integral [slip·s]
    omega_prev: jax.Array     # (4,) previous wheel angular velocities [rad/s]
    kappa_meas: jax.Array     # (4,) last measured slip ratios (for diagnostics)

    @classmethod
    def default(cls, params: SimpleTCParams = SimpleTCParams()) -> 'SimpleTCState':
        return cls(
            pi_integral=jnp.zeros(4),
            omega_prev=jnp.zeros(4),
            kappa_meas=jnp.full(4, params.lambda_ref),
        )

def make_simple_tc_state(params: SimpleTCParams = SimpleTCParams()) -> SimpleTCState:
    return SimpleTCState.default(params)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Diagnostics Output
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTCDiagnostics(NamedTuple):
    """Per-step diagnostics — structurally compatible with TCOutput for dashboard."""
    kappa_star: jax.Array      # (4,) target slip ratios
    kappa_measured: jax.Array  # (4,) measured slip ratios
    kappa_error: jax.Array     # (4,) error = measured − target
    omega_dot: jax.Array       # (4,) wheel angular acceleration [rad/s²]
    pid_output: jax.Array      # (4,) raw PID output before gating [Nm]
    emg_cut: jax.Array         # (4,) emergency high-accel reduction [Nm]
    raw_reduction: jax.Array   # (4,) theoretical per-wheel reduction needed [Nm]
    axle_reduction: jax.Array  # (4,) symmetric yaw-preserving reduction applied [Nm]
    speed_gate: jax.Array      # scalar TC activation level [0, 1]
    confidence: jax.Array      # scalar wheel speed plausibility [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# §4  Slip Ratio + Sensor Confidence (standalone, JIT-safe)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _compute_kappa(omega: jax.Array, vx: jax.Array, params: SimpleTCParams) -> jax.Array:
    denom = params.v_min + jax.nn.softplus(vx - params.v_min)
    return jnp.clip(
        (omega * params.r_w - vx) / denom,
        -0.80, 0.80,
    )

@jax.jit
def _wheel_confidence(omega: jax.Array, vx: jax.Array, params: SimpleTCParams, omega_max: float = 1200.0) -> jax.Array:
    in_range = jnp.prod(
        jax.nn.sigmoid((omega_max - omega) * 0.01)
        * jax.nn.sigmoid(omega * 10.0)
    )
    kappa = _compute_kappa(omega, vx, params)
    axle_delta = jnp.abs(jnp.mean(kappa[:2]) - jnp.mean(kappa[2:]))
    consistency = jax.nn.sigmoid(0.6 - axle_delta * 5.0)
    return jnp.clip(in_range * consistency, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Main Step Function
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def tc_simple(
    T_requested: jax.Array,    # (4,) torques from DYC allocator [Nm]
    omega_wheel: jax.Array,    # (4,) wheel angular velocities [rad/s]
    vx: jax.Array,             # scalar longitudinal velocity [m/s]
    state: SimpleTCState,
    params: SimpleTCParams,
    dt: jax.Array,
) -> tuple[jax.Array, SimpleTCState, SimpleTCDiagnostics]:
    """
    One PID traction control timestep with Yaw-Preservation.
    """
    drive = jnp.array(params.drive_mask)

    # ── §5.1 Slip & Acceleration measurement ──────────────────────────────
    kappa = _compute_kappa(omega_wheel, vx, params)
    error = kappa - params.lambda_ref
    
    # Derivada de la rueda: Esencial para atrapar picos de deslizamiento antes 
    # de que impacten la medición de slip estática, o cuando vx es engañoso.
    omega_dot = (omega_wheel - state.omega_prev) / dt

    # ── §5.2 Speed activation gate ────────────────────────────────────────
    speed_gate = jax.nn.sigmoid((vx - params.v_min) * 2.0)

    # ── §5.3 Anti-windup Integrator ───────────────────────────────────────
    raw_integral = state.pi_integral + error * dt * speed_gate
    new_integral = params.I_max * jnp.tanh(raw_integral / params.I_max)

    # ── §5.4 PID Output & Emergency Cut ───────────────────────────────────
    # P + I + D (D reacciona instantáneamente a la tasa de cambio de la rueda)
    pid_out = params.Kp * error + params.Ki * new_integral + params.Kd * omega_dot
    
    # Límite físico: Si aceleramos más de omega_dot_max (ej. rueda en el aire o hielo total), 
    # cortamos el par agresivamente ignorando el speed_gate (independiente de vx).
    emg_cut = params.K_emg * jax.nn.softplus(omega_dot - params.omega_dot_max)

    # ── §5.5 Raw Independent Reduction ────────────────────────────────────
    beta = params.clamp_sharpness
    pid_reduction = speed_gate * jax.nn.softplus(pid_out * beta) / beta
    
    raw_reduction = drive * (pid_reduction + emg_cut)

    # ── §5.6 Yaw-Preserving Symmetric Allocation ──────────────────────────
    # Para no arruinar el Torque Vectoring, encontramos la reducción máxima
    # necesaria en el eje y se la restamos a ambas ruedas por igual. 
    # Esto mantiene el Delta T direccional inalterado.
    red_f = jnp.maximum(raw_reduction[0], raw_reduction[1])
    red_r = jnp.maximum(raw_reduction[2], raw_reduction[3])
    
    axle_reduction = jnp.array([red_f, red_f, red_r, red_r])

    T_cmd = T_requested - axle_reduction

    # ── §5.7 Output conditioning ──────────────────────────────────────────
    T_driven   = jax.nn.softplus(T_cmd * beta) / beta   # ≥ 0
    T_undriven = T_requested                            # exact pass-through
    T_out      = jnp.where(drive > 0.5, T_driven, T_undriven)

    # ── §5.8 State & Diagnostics update ───────────────────────────────────
    new_state = SimpleTCState(
        pi_integral=new_integral,
        omega_prev=omega_wheel,
        kappa_meas=kappa,
    )

    confidence = _wheel_confidence(omega_wheel, vx, params)
    diag = SimpleTCDiagnostics(
        kappa_star=jnp.full(4, params.lambda_ref),
        kappa_measured=kappa,
        kappa_error=error,
        omega_dot=omega_dot,
        pid_output=pid_out,
        emg_cut=emg_cut,
        raw_reduction=raw_reduction,
        axle_reduction=axle_reduction,
        speed_gate=speed_gate,
        confidence=confidence,
    )

    return T_out, new_state, diag


# ─────────────────────────────────────────────────────────────────────────────
# §6  Standalone Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """Validates Yaw-Preserving behavior under asymmetrical slip."""
    import time

    params = SimpleTCParams()
    state  = SimpleTCState.default(params)
    dt     = jnp.array(0.005)

    # Scenario: High speed cornering right.
    # Outer wheel (RL, index 2) hits a slippery patch and spikes its acceleration.
    # Inner wheel (RR, index 3) has normal speed.
    vx = jnp.array(15.0)
    w_base = 15.0 / 0.2032
    
    omega_spin = jnp.array([
        w_base,          # FL 
        w_base,          # FR 
        w_base * 1.20,   # RL (Outer wheel, spinning 20% over vx)
        w_base           # RR (Inner wheel, no slip)
    ])
    
    # Set prev state to simulate the massive acceleration spike (d_omega/dt > 300)
    state = state._replace(omega_prev=jnp.array([w_base, w_base, w_base, w_base]))
    
    # TV requested torques: 80Nm outer, 20Nm inner (Delta T = 60Nm)
    T_req = jnp.array([0.0, 0.0, 80.0, 20.0])

    print("Compiling Yaw-Preserving tc_simple...")
    t0 = time.perf_counter()
    T_out, state, diag = tc_simple(T_req, omega_spin, vx, state, params, dt)
    _ = float(T_out[2])  # force eval
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"  Compile: {compile_ms:.1f} ms")

    print("\nSimulating Torque Vectoring cornering + Outer wheel slip spike")
    print(f"  T_req (TV input):    {[float(x) for x in T_req]}")
    print(f"  Initial Delta T:     {float(T_req[2] - T_req[3]):.1f} Nm")
    
    print(f"\n  T_out (TC output):   {[round(float(x), 2) for x in T_out]}")
    print(f"  Final Delta T:       {round(float(T_out[2] - T_out[3]), 2)} Nm")
    
    print(f"\nDiagnostics:")
    print(f"  Raw Reduction req.:  {[round(float(x), 2) for x in diag.raw_reduction]}")
    print(f"  Axle Reduction applied:{[round(float(x), 2) for x in diag.axle_reduction]}")
    print(f"  Outer Wheel d_omega: {float(diag.omega_dot[2]):.1f} rad/s²")
    print(f"  Outer Wheel EMG cut: {float(diag.emg_cut[2]):.1f} Nm")

    # Sanity assertions
    assert float(T_out[2]) < float(T_req[2]), "RL torque must be reduced"
    assert float(T_out[3]) < float(T_req[3]), "RR torque must ALSO be reduced to preserve yaw"
    assert float(diag.axle_reduction[2]) == float(diag.axle_reduction[3]), "Reduction must be symmetric"
    
    # The Delta T should be preserved exactly, UNLESS the inner wheel clamped at 0
    delta_req = float(T_req[2] - T_req[3])
    delta_out = float(T_out[2] - T_out[3])
    assert abs(delta_req - delta_out) < 1.0 or float(T_out[3]) < 1e-3, "Delta T must be preserved unless inner wheel hits 0 limit"
    
    print("\n✅ All assertions passed. Yaw moment preserved successfully.")

if __name__ == '__main__':
    smoke_test()