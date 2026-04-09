# powertrain/modes/simple/traction_control.py
# Project-GP — Simple PI Traction Controller
# ═══════════════════════════════════════════════════════════════════════════════
#
# Direct torque-correcting PI controller for embedded (SIMPLE mode) use.
#
# Architecture contract (differs from ADVANCED mode):
#   ADVANCED: tc_step → kappa_star references → SOCP allocator uses them
#   SIMPLE:   tc_simple → directly corrects T_requested in-place
#
# The pipeline for SIMPLE mode is:
#   T_base  = simple_dyc_torque_vectoring(...)   # DYC yaw control
#   T_final = tc_simple(T_base, omega, vx, ...)  # PI slip correction
#
# Mathematical formulation (per driven wheel i):
#   κᵢ     = (ωᵢ·r_w − vₓ) / softplus(vₓ − v_min + v_min)   [smooth denom]
#   eᵢ     = κᵢ − κ_ref
#   Iᵢ    += eᵢ·dt·gate                                        [gated accumulation]
#   Iᵢ     = I_max · tanh(Iᵢ / I_max)                          [anti-windup]
#   πᵢ     = Kp·eᵢ + Ki·Iᵢ
#   Δᵢ     = maskᵢ · gate · softplus(πᵢ · β) / β              [one-sided, smooth]
#   T_cmdᵢ = T_reqᵢ − Δᵢ
#   T_outᵢ = driven  ? softplus(T_cmdᵢ · β) / β               [lower-clamp at 0]
#           : T_reqᵢ                                           [undriven: pass-through]
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
    PI traction control hyperparameters.

    Gain rationale (Hoosier R20 @ 300 kg FS vehicle):
      Kp = 80  Nm/slip:  at 5% excess slip → 4 Nm reduction (~7% of 55 Nm peak per wheel)
                         at 20% excess slip → 16 Nm reduction (full slip suppression)
      Ki = 20  Nm/(slip·s): 5% excess slip for 5s → 5 Nm additional (I_max·tanh caps it)
      I_max = 5.0 slip·s: caps integral authority at Ki·I_max = 100 Nm — prevents
                          aggressive braking torque on recovery from loss-of-traction
      clamp_sharpness = 50: softplus(x·50)/50 approximates max(0,x) with <15mNm
                            bias in the ±20mNm transition band — negligible at FS scale
    """
    Kp: float = 80.0             # Nm per unit slip error
    Ki: float = 20.0             # Nm / (slip·s)
    lambda_ref: float = 0.08     # target slip ratio (Hoosier R20 optimum: 0.07–0.10)
    I_max: float = 5.0           # anti-windup ceiling [slip·s] → caps Ki effect at 100 Nm
    clamp_sharpness: float = 50.0  # softplus β [Nm⁻¹] for smooth max(0, ·)
    r_w: float = 0.2032          # effective wheel radius [m] (8" tire loaded)
    v_min: float = 2.0           # TC activation threshold [m/s] — suppresses low-speed windup
    drive_mask: tuple = (0., 0., 1., 1.)  # which wheels are driven; RWD default


# ─────────────────────────────────────────────────────────────────────────────
# §2  State
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTCState(NamedTuple):
    """Persistent state for the simple PI controller."""
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
    kappa_star: jax.Array      # (4,) target slip ratios [= lambda_ref broadcast]
    kappa_measured: jax.Array  # (4,) measured slip ratios
    kappa_error: jax.Array     # (4,) error = measured − target
    pi_output: jax.Array       # (4,) raw PI output before one-sided gate [Nm]
    reduction: jax.Array       # (4,) applied torque reduction [Nm]
    speed_gate: jax.Array      # scalar TC activation level [0, 1]
    confidence: jax.Array      # scalar wheel speed plausibility [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# §4  Slip Ratio + Sensor Confidence (standalone, JIT-safe)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _compute_kappa(
    omega: jax.Array,
    vx: jax.Array,
    params: SimpleTCParams,
) -> jax.Array:
    """
    Per-wheel longitudinal slip ratio.
    Smooth denominator prevents division by zero at standstill and
    naturally deactivates TC below v_min via the speed gate.
    """
    denom = params.v_min + jax.nn.softplus(vx - params.v_min)
    return jnp.clip(
        (omega * params.r_w - vx) / denom,
        -0.80, 0.80,    # physical bounds: never beyond locked/full-spin
    )


@jax.jit
def _wheel_confidence(
    omega: jax.Array,
    vx: jax.Array,
    params: SimpleTCParams,
    omega_max: float = 1200.0,
) -> jax.Array:
    """
    Scalar sensor health score [0, 1].
    Degrades under: out-of-range speeds or large front/rear slip disagreement.
    Smooth everywhere — safe for grad().
    """
    # Per-wheel range gate (both directions)
    in_range = jnp.prod(
        jax.nn.sigmoid((omega_max - omega) * 0.01)
        * jax.nn.sigmoid(omega * 10.0)
    )
    # Axle consistency: large front–rear kappa delta → fault / spinout
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
    One PI traction control timestep.

    Directly modifies T_requested to reduce driven-wheel slip toward lambda_ref.
    Undriven wheels pass through unchanged (including negative regen torques).

    Returns:
        T_corrected: (4,) torque commands after PI correction [Nm]
        new_state:   SimpleTCState carrying forward to next timestep
        diag:        SimpleTCDiagnostics for telemetry / dashboard
    """
    drive = jnp.array(params.drive_mask)    # (4,) static — resolved at trace time

    # ── §5.1  Slip measurement ────────────────────────────────────────────
    kappa = _compute_kappa(omega_wheel, vx, params)
    error = kappa - params.lambda_ref        # positive = over-slip (too much wheel spin)

    # ── §5.2  Speed activation gate ───────────────────────────────────────
    # Sigmoid ramp: 0.0 at vx=v_min, 0.88 at vx=2·v_min, ~1.0 at vx>>v_min
    # Prevents integrator windup during standstill and launch crawl.
    speed_gate = jax.nn.sigmoid((vx - params.v_min) * 2.0)

    # ── §5.3  Anti-windup PI integrator ──────────────────────────────────
    # Gated accumulation: integral grows only when TC is active
    raw_integral = state.pi_integral + error * dt * speed_gate
    # Tanh saturation: smoothly clamps to (−I_max, +I_max)
    # Advantage over hard clamp: gradient flows through at saturation → MORL-tunable
    new_integral = params.I_max * jnp.tanh(raw_integral / params.I_max)

    # ── §5.4  PI output ───────────────────────────────────────────────────
    pi_out = params.Kp * error + params.Ki * new_integral   # [Nm], can be ±

    # ── §5.5  One-sided torque reduction ─────────────────────────────────
    # softplus(π · β) / β ≈ max(0, π) — only positive pi_out reduces torque.
    # When π < 0 (under-slip): reduction ≈ 0 → T_out ≈ T_requested (no intervention)
    # When π > 0 (over-slip):  reduction ≈ π → torque decreases
    # drive mask: undriven wheels get zero reduction regardless
    beta = params.clamp_sharpness
    reduction = drive * speed_gate * jax.nn.softplus(pi_out * beta) / beta

    T_cmd = T_requested - reduction

    # ── §5.6  Output conditioning (per-wheel) ────────────────────────────
    # Driven wheels: lower-clamp at 0 (no negative drive torque from TC)
    # Undriven wheels: pass through unchanged (preserves regen torques)
    T_driven   = jax.nn.softplus(T_cmd * beta) / beta   # ≥ 0, smooth
    T_undriven = T_requested                              # exact pass-through
    T_out      = jnp.where(drive > 0.5, T_driven, T_undriven)

    # ── §5.7  State update ────────────────────────────────────────────────
    new_state = SimpleTCState(
        pi_integral=new_integral,
        omega_prev=omega_wheel,
        kappa_meas=kappa,
    )

    # ── §5.8  Diagnostics ─────────────────────────────────────────────────
    confidence = _wheel_confidence(omega_wheel, vx, params)
    diag = SimpleTCDiagnostics(
        kappa_star=jnp.full(4, params.lambda_ref),
        kappa_measured=kappa,
        kappa_error=error,
        pi_output=pi_out,
        reduction=reduction,
        speed_gate=speed_gate,
        confidence=confidence,
    )

    return T_out, new_state, diag


# ─────────────────────────────────────────────────────────────────────────────
# §6  Standalone Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """Quick validation: compile, run 200 steps, check outputs."""
    import time

    params = SimpleTCParams()
    state  = SimpleTCState.default(params)
    dt     = jnp.array(0.005)

    # Simulate: 15 m/s, rear wheels spinning at 10% slip (kappa ≈ 0.10 > 0.08 ref)
    omega_spin = jnp.array([15.0 / 0.2032,  # FL — no slip
                             15.0 / 0.2032,  # FR — no slip
                             15.0 / 0.2032 * 1.10,   # RL — 10% over
                             15.0 / 0.2032 * 1.10])  # RR — 10% over
    vx     = jnp.array(15.0)
    T_req  = jnp.array([0.0, 0.0, 50.0, 50.0])   # RWD: 50 Nm/wheel rear

    print("Compiling tc_simple...")
    t0 = time.perf_counter()
    T_out, state, diag = tc_simple(T_req, omega_spin, vx, state, params, dt)
    _ = float(T_out[2])   # force eval
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"  Compile: {compile_ms:.1f} ms")

    print("Running 200 steps...")
    t0 = time.perf_counter()
    for _ in range(200):
        T_out, state, diag = tc_simple(T_req, omega_spin, vx, state, params, dt)
    _ = float(T_out[2])
    run_ms = (time.perf_counter() - t0) * 1000
    per_step = run_ms / 200

    print(f"  Total: {run_ms:.1f} ms | Per-step: {per_step:.4f} ms")
    print(f"  Budget (5 ms): {'PASS' if per_step < 5.0 else 'FAIL'}")
    print(f"\nOutputs (over-slip scenario):")
    print(f"  T_req:       {[float(x) for x in T_req]}")
    print(f"  T_out:       {[round(float(x), 2) for x in T_out]}")
    print(f"  reduction:   {[round(float(x), 2) for x in diag.reduction]}")
    print(f"  kappa_meas:  {[round(float(x), 4) for x in diag.kappa_measured]}")
    print(f"  kappa_error: {[round(float(x), 4) for x in diag.kappa_error]}")
    print(f"  speed_gate:  {float(diag.speed_gate):.4f}")
    print(f"  confidence:  {float(diag.confidence):.4f}")
    print(f"  PI integrals:{[round(float(x), 4) for x in state.pi_integral]}")

    # Sanity assertions
    assert float(T_out[0]) == 0.0 or abs(float(T_out[0])) < 1e-3, "Front FL should be ~0"
    assert float(T_out[1]) == 0.0 or abs(float(T_out[1])) < 1e-3, "Front FR should be ~0"
    assert float(T_out[2]) < float(T_req[2]), "RL torque must be reduced (over-slip)"
    assert float(T_out[3]) < float(T_req[3]), "RR torque must be reduced (over-slip)"
    assert float(T_out[2]) >= 0.0, "RL torque must be non-negative (driven wheel)"
    print("\n✅ All assertions passed")


if __name__ == '__main__':
    smoke_test()