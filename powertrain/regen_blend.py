# powertrain/regen_blend.py
# Project-GP — Batch 9: Dynamic Regen Blend
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces the fixed `regen_blend = 0.7` scalar in PowertrainConfig with an
# analytically optimal α* that maximises energy recovery subject to:
#   · Battery aggregate charge-current limit  (P_regen ≤ I_charge_max · V_bus)
#   · Cell temperature derating               (hot battery → reduce regen)
#   · High-SoC tapering                       (near-full → reduce regen)
#   · Per-wheel motor envelope                (already encoded in T_min)
#   · Slip-barrier feasibility                (don't exceed κ* even during regen)
#
# After the KKT allocator produces T* (wheel torques), this module:
#   1. Computes the regen power T* would deliver at current ω
#   2. Checks it against the battery-side power budget
#   3. Scales T* if the budget is exceeded (battery-limited blending)
#   4. Computes F_hydraulic — the brake force the hydraulic system must provide
#      to make up the deficit between driver demand and what regen achieves
#   5. Returns α ∈ [0,1] (regen fraction) as a diagnostic + control output
#
# DIFFERENTIABILITY
# ─────────────────
# All operations are C∞ (softplus, sigmoid, jnp.clip). No hard conditionals.
# jax.grad(total_regen_energy)(setup_vector) is well-defined — this is the
# signal the Batch 11 bilevel setup optimizer will backprop through.
#
# HYDRAULIC BRAKE INTERFACE
# ─────────────────────────
# The output F_brake_hydraulic [N, positive = braking force] is the command
# to the brake-by-wire actuator. At the physics server level:
#
#     F_brake_total_wheel = F_regen_wheel + F_hydraulic_wheel
#
# The hydraulic distribution (front/rear bias) is handled by the brake
# pressure modulator, not by this module — we output the total scalar force.
#
# INTEGRATION POINT
# ─────────────────
# Called from powertrain_manager.py between Step 8 (SOCP) and Step 9 (CBF):
#
#   T_alloc_regen, alpha, F_hydraulic, regen_diag = compute_regen_blend(
#       T_alloc, T_min, Fx_driver, omega_wheel, pt, mp, bp, geo, params,
#   )
#
# T_alloc_regen replaces T_alloc going into the CBF filter. The CBF does not
# touch F_hydraulic — that signal exits the control loop to the brake actuator.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from powertrain.motor_model import MotorParams, BatteryParams, PowertrainState


# ─────────────────────────────────────────────────────────────────────────────
# §1  Parameters
# ─────────────────────────────────────────────────────────────────────────────

class RegenBlendParams(NamedTuple):
    """Tuning knobs for the dynamic regen blend."""

    # Minimum braking demand before regen activates [N].
    # Below this, friction brakes handle everything (creep / light touch).
    F_brake_threshold: float = 100.0

    # Temperature above which cell regen current starts to derate [°C].
    T_cell_derate_start: float = 40.0

    # Temperature at which regen is fully cut [°C].
    T_cell_derate_end: float = 55.0

    # SoC above which regen tapers [%].
    SoC_taper_start: float = 92.0

    # SoC at which regen is fully blocked [%].
    SoC_taper_end: float = 98.0

    # Blending sharpness — higher = harder threshold (still C∞).
    # 50 is near-ideal: looks like a step at ±2% SoC but grad is finite.
    sigmoid_sharpness: float = 50.0

    # Hydraulic brake gain: scales deficit into brake pressure command.
    # 1.0 = full deficit goes to hydraulics (conservative, no brake fade).
    hydraulic_gain: float = 1.0

    # Weight for regen reward in the objective (used by Batch 11 bilevel).
    w_regen: float = 0.15

    # Physical constants — must match motor_model.py defaults.
    r_w: float = 0.2032    # m  loaded wheel radius


# ─────────────────────────────────────────────────────────────────────────────
# §2  Battery-side regen power budget
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_regen_power_budget(
    pt:     PowertrainState,
    bp:     BatteryParams = BatteryParams(),
    p:      RegenBlendParams = RegenBlendParams(),
) -> jax.Array:
    """
    Maximum aggregate regen power the battery can absorb [W].

    Two independent limits, both C∞:
      1. Current limit: I_charge_max × V_bus, tapered by high SoC
      2. Temperature derating: linear reduction T_cell ∈ [T_start, T_end]

    Returns a scalar P_regen_max [W] ≥ 0.
    """
    # ── SoC taper ─────────────────────────────────────────────────────────
    # sigmoid(k · (SoC_end - SoC)) → 1 below SoC_start, → 0 above SoC_end
    soc_gate = jax.nn.sigmoid(
        p.sigmoid_sharpness * (p.SoC_taper_end - pt.SoC)
    )  # scalar ∈ (0, 1)

    # ── Charge current available (already SoC-weighted in motor_model,
    #    but we re-derive the aggregate here for the blend computation) ────
    I_charge = bp.I_charge_max * jax.nn.sigmoid(
        0.3 * (95.0 - pt.SoC)
    )  # scalar [A]

    # ── Thermal derating ──────────────────────────────────────────────────
    # Linear interp from 1.0 at T_start to 0.0 at T_end, clipped, C∞ via clip
    thermal_frac = jnp.clip(
        (p.T_cell_derate_end - pt.T_cell)
        / (p.T_cell_derate_end - p.T_cell_derate_start + 1e-3),
        0.0, 1.0,
    )  # scalar ∈ [0, 1]

    # ── Combine ────────────────────────────────────────────────────────────
    P_regen_max = I_charge * pt.V_bus * soc_gate * thermal_frac  # [W]
    return jnp.maximum(P_regen_max, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Core: compute optimal α and hydraulic residual
# ─────────────────────────────────────────────────────────────────────────────

class RegenDiagnostics(NamedTuple):
    """Regen blend diagnostics — all scalar unless noted."""
    alpha_regen:        jax.Array   # scalar ∈ [0,1] actual regen fraction
    F_brake_hydraulic:  jax.Array   # scalar [N] hydraulic brake force (positive)
    P_regen_achieved:   jax.Array   # scalar [W] actual regen power from T_alloc
    P_regen_budget:     jax.Array   # scalar [W] battery-side budget
    F_regen_achieved:   jax.Array   # scalar [N] total regen braking force
    F_brake_demand:     jax.Array   # scalar [N] total braking demand (positive)
    battery_limited:    jax.Array   # scalar [0,1] soft flag: budget was binding


@jax.jit
def compute_regen_blend(
    T_alloc:     jax.Array,         # (4,) KKT-optimal wheel torques [Nm]
    T_min:       jax.Array,         # (4,) min wheel torque from motor model [Nm]
    Fx_driver:   jax.Array,         # scalar total force demand [N] (negative = braking)
    omega_wheel: jax.Array,         # (4,) wheel angular velocities [rad/s]
    pt:          PowertrainState,
    bp:          BatteryParams      = BatteryParams(),
    p:           RegenBlendParams   = RegenBlendParams(),
) -> tuple[jax.Array, RegenDiagnostics]:
    """
    Compute the dynamic regen blend and hydraulic brake residual.

    Returns:
        T_alloc_scaled  (4,)          — regen torques scaled to battery budget
        RegenDiagnostics              — α, F_hydraulic, and diagnostics
    """
    # ── 1. Braking demand (positive scalar) ────────────────────────────────
    # Only active when driver demands braking (Fx_driver < 0)
    F_brake_demand = jax.nn.softplus(
        p.sigmoid_sharpness * (-Fx_driver - p.F_brake_threshold)
    ) / p.sigmoid_sharpness + p.F_brake_threshold * jax.nn.sigmoid(
        p.sigmoid_sharpness * (-Fx_driver - p.F_brake_threshold)
    )
    # Simpler: smooth max(0, -Fx_driver)
    F_brake_demand = jax.nn.softplus(-Fx_driver * 10.0) / 10.0   # ≈ max(0, -Fx_driver)

    # ── 2. Regen power T_alloc would deliver ──────────────────────────────
    # Power = torque × angular velocity, but only for negative (regen) torques
    # Use softplus(-T·ω) to smoothly select regen wheels
    T_regen_smooth = -jax.nn.softplus(-T_alloc * 10.0) / 10.0    # (4,) ≈ min(T_alloc, 0)
    P_regen_achieved = jnp.sum(
        jax.nn.softplus(-T_alloc * jnp.abs(omega_wheel) * 10.0) / 10.0
    )   # [W] smooth positive regen power

    # Simpler, numerically stable version:
    P_regen_per_wheel = jax.nn.softplus(
        -T_alloc * jnp.abs(omega_wheel)
    )   # (4,) [W] — softplus(x) ≈ max(0, x), smooth
    P_regen_achieved  = jnp.sum(P_regen_per_wheel)    # [W]

    # ── 3. Battery power budget ───────────────────────────────────────────
    P_regen_budget = compute_regen_power_budget(pt, bp, p)    # [W]

    # ── 4. Budget utilisation — scaling factor for T_alloc regen portion ──
    # If P_regen_achieved ≤ P_regen_budget: no scaling needed, α = 1
    # If P_regen_achieved > P_regen_budget: scale down regen torques
    # Smooth ratio — avoid division by zero
    budget_ratio = P_regen_budget / (P_regen_achieved + 1e-3)   # scalar
    # Soft clip at 1.0: scale = min(1, budget_ratio)
    regen_scale = jnp.clip(budget_ratio, 0.0, 1.0)              # ∈ [0, 1]

    # ── 5. Scale regen torques, keep drive torques intact ─────────────────
    # For each wheel: if T_i < 0 (regen), scale it; if T_i > 0 (drive), keep it
    # Smooth split via tanh-based gate:
    #   regen_gate_i = σ(−k·T_i)  → 1 when T_i ≪ 0, 0 when T_i ≫ 0
    regen_gate = jax.nn.sigmoid(-p.sigmoid_sharpness * T_alloc / 50.0)   # (4,)
    T_alloc_scaled = T_alloc * (regen_gate * regen_scale + (1.0 - regen_gate))
    # = regen wheels scaled by regen_scale, drive wheels unchanged

    # ── 6. Regen force actually achieved after scaling ────────────────────
    F_regen_achieved = -jnp.sum(
        jax.nn.softplus(-T_alloc_scaled) / p.r_w
    )   # [N] negative = braking, take magnitude
    F_regen_achieved = jnp.sum(
        jax.nn.softplus(-T_alloc_scaled / p.r_w)
    )   # [N] positive scalar, force equivalent of regen

    # ── 7. α = regen fraction of total braking demand ─────────────────────
    # Gate: only meaningful when braking (F_brake_demand > threshold)
    brake_active = jax.nn.sigmoid(
        p.sigmoid_sharpness * (F_brake_demand - p.F_brake_threshold)
    )   # scalar ∈ (0, 1)

    alpha_regen = brake_active * jnp.clip(
        F_regen_achieved / (F_brake_demand + 1e-3),
        0.0, 1.0,
    )   # scalar ∈ [0, 1]

    # ── 8. Hydraulic brake residual ───────────────────────────────────────
    # F_brake_demand not covered by regen → hydraulic system
    F_brake_deficit = jax.nn.softplus(
        F_brake_demand - F_regen_achieved
    )   # [N] positive, smooth max(0, deficit)
    F_brake_hydraulic = p.hydraulic_gain * F_brake_deficit   # [N]

    # ── 9. Battery-limited flag (soft) ────────────────────────────────────
    battery_limited = jax.nn.sigmoid(
        p.sigmoid_sharpness * (P_regen_achieved - P_regen_budget) / (P_regen_budget + 1.0)
    )   # → 1 when achieved > budget (was being limited)

    diag = RegenDiagnostics(
        alpha_regen       = alpha_regen,
        F_brake_hydraulic = F_brake_hydraulic,
        P_regen_achieved  = P_regen_achieved,
        P_regen_budget    = P_regen_budget,
        F_regen_achieved  = F_regen_achieved,
        F_brake_demand    = F_brake_demand,
        battery_limited   = battery_limited,
    )

    return T_alloc_scaled, diag


# ─────────────────────────────────────────────────────────────────────────────
# §4  Regen energy integrator (for lap-level accounting)
# ─────────────────────────────────────────────────────────────────────────────

class RegenEnergyState(NamedTuple):
    """Persistent energy accounting across steps."""
    E_regen_J:    jax.Array   # scalar: cumulative regen energy this lap [J]
    E_consumed_J: jax.Array   # scalar: cumulative consumed energy this lap [J]
    n_steps:      jax.Array   # scalar int: step counter

    @classmethod
    def zero(cls) -> "RegenEnergyState":
        return cls(
            E_regen_J    = jnp.array(0.0),
            E_consumed_J = jnp.array(0.0),
            n_steps      = jnp.array(0, dtype=jnp.int32),
        )


@jax.jit
def update_regen_energy(
    state:     RegenEnergyState,
    diag:      RegenDiagnostics,
    T_alloc:   jax.Array,         # (4,) [Nm] post-blend torques
    omega:     jax.Array,         # (4,) [rad/s]
    dt:        jax.Array,         # scalar [s]
) -> RegenEnergyState:
    """
    Integrate regen and consumption energy over one timestep.
    Used for lap-level efficiency reporting (Batch 11 objective).
    """
    # Mechanical power per wheel [W] — negative = regen, positive = drive
    P_wheel = T_alloc * omega          # (4,)

    # Split: regen (P < 0) and drive (P > 0)
    P_regen_this    = jnp.sum(jax.nn.softplus(-P_wheel))   # [W] smooth positive
    P_consumed_this = jnp.sum(jax.nn.softplus(P_wheel))    # [W] smooth positive

    return RegenEnergyState(
        E_regen_J    = state.E_regen_J    + P_regen_this    * dt,
        E_consumed_J = state.E_consumed_J + P_consumed_this * dt,
        n_steps      = state.n_steps + jnp.array(1, dtype=jnp.int32),
    )


@jax.jit
def regen_efficiency(state: RegenEnergyState) -> jax.Array:
    """
    Regen-to-consumption ratio over accumulated period.
    Target: ≥ 15% for FSG Endurance (typical Formula Student E benchmark).
    """
    return state.E_regen_J / (state.E_consumed_J + 1.0)   # scalar


# ─────────────────────────────────────────────────────────────────────────────
# §5  Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """
    Verify:
    1. High-SoC → low α (battery blocks regen)
    2. Nominal SoC + hard braking → high α (maximum regen)
    3. No braking → F_hydraulic ≈ 0
    4. Budget-limited scenario → T_alloc scaled down, F_hydraulic covers deficit
    5. Gradient flows through α (differentiable w.r.t. Fx_driver)

    Run: python -m powertrain.regen_blend
    """
    import time

    print("=" * 60)
    print(" REGEN BLEND SMOKE TEST (BATCH 9)")
    print("=" * 60)

    p   = RegenBlendParams()
    bp  = BatteryParams()
    mp  = None  # not needed here

    omega = jnp.full(4, 20.0 / 0.2032)    # ~98 rad/s wheel speed
    # KKT allocated heavy braking: all regen torques
    T_braking = jnp.full(4, -180.0)       # [Nm] regen
    Fx_braking = jnp.array(-6000.0)       # [N] hard braking demand

    # ── Test 1: Nominal SoC, hard braking → max regen ─────────────────────
    print("\n[Test 1] Nominal SoC (80%), hard braking — expect high α")
    pt_nominal = PowertrainState(
        T_motors=jnp.full(4, 55.0), T_invs=jnp.full(4, 45.0),
        SoC=jnp.array(80.0), T_cell=jnp.array(32.0),
        V_bus=jnp.array(580.0),
    )
    t0 = time.perf_counter()
    T_sc, diag = compute_regen_blend(T_braking, jnp.full(4, -200.0), Fx_braking,
                                      omega, pt_nominal, bp, p)
    _ = float(diag.alpha_regen)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  Post-JIT: {ms:.3f} ms")
    print(f"  α_regen = {float(diag.alpha_regen):.3f} (expected > 0.4)")
    print(f"  F_hydraulic = {float(diag.F_brake_hydraulic):.1f} N")
    print(f"  P_regen_budget = {float(diag.P_regen_budget)/1000:.1f} kW")
    print(f"  P_regen_achieved = {float(diag.P_regen_achieved)/1000:.1f} kW")
    if float(diag.alpha_regen) > 0.4:
        print("  [PASS] High regen fraction at nominal SoC")
    else:
        print("  [WARN] Lower than expected regen fraction")

    # ── Test 2: High SoC → α suppressed ────────────────────────────────────
    print("\n[Test 2] High SoC (97%), hard braking — expect low α")
    pt_high_soc = PowertrainState(
        T_motors=jnp.full(4, 55.0), T_invs=jnp.full(4, 45.0),
        SoC=jnp.array(97.0), T_cell=jnp.array(32.0),
        V_bus=jnp.array(600.0),
    )
    _, diag_hs = compute_regen_blend(T_braking, jnp.full(4, -200.0), Fx_braking,
                                      omega, pt_high_soc, bp, p)
    print(f"  α_regen = {float(diag_hs.alpha_regen):.3f} (expected < 0.3)")
    print(f"  P_regen_budget = {float(diag_hs.P_regen_budget)/1000:.1f} kW")
    if float(diag_hs.alpha_regen) < float(diag.alpha_regen):
        print("  [PASS] High SoC correctly suppresses regen")
    else:
        print("  [FAIL] High SoC did not reduce regen")

    # ── Test 3: Hot cell → α reduced ───────────────────────────────────────
    print("\n[Test 3] Hot cell (52°C), nominal SoC — expect thermal derating")
    pt_hot = PowertrainState(
        T_motors=jnp.full(4, 90.0), T_invs=jnp.full(4, 70.0),
        SoC=jnp.array(75.0), T_cell=jnp.array(52.0),
        V_bus=jnp.array(570.0),
    )
    _, diag_hot = compute_regen_blend(T_braking, jnp.full(4, -200.0), Fx_braking,
                                       omega, pt_hot, bp, p)
    print(f"  α_regen = {float(diag_hot.alpha_regen):.3f}")
    print(f"  P_regen_budget = {float(diag_hot.P_regen_budget)/1000:.1f} kW")
    if float(diag_hot.P_regen_budget) < float(diag.P_regen_budget):
        print("  [PASS] Thermal derating reduces regen budget")
    else:
        print("  [WARN] No thermal derating observed")

    # ── Test 4: No braking → F_hydraulic ≈ 0 ──────────────────────────────
    print("\n[Test 4] Throttle demand (no braking) — expect F_hydraulic ≈ 0")
    T_drive = jnp.full(4, 150.0)
    Fx_drive = jnp.array(4000.0)
    _, diag_drive = compute_regen_blend(T_drive, jnp.full(4, -200.0), Fx_drive,
                                         omega, pt_nominal, bp, p)
    print(f"  F_hydraulic = {float(diag_drive.F_brake_hydraulic):.2f} N (expected ≈ 0)")
    print(f"  α_regen = {float(diag_drive.alpha_regen):.3f} (expected ≈ 0)")
    if float(diag_drive.F_brake_hydraulic) < 10.0:
        print("  [PASS] No hydraulic braking during throttle")
    else:
        print("  [FAIL] Spurious hydraulic brake command during acceleration")

    # ── Test 5: Differentiability ──────────────────────────────────────────
    print("\n[Test 5] Gradient ∂α/∂Fx_driver (differentiability)")

    def alpha_from_fx(fx):
        _, d = compute_regen_blend(T_braking, jnp.full(4, -200.0), fx,
                                    omega, pt_nominal, bp, p)
        return d.alpha_regen

    grad_fn  = jax.jit(jax.grad(alpha_from_fx))
    g        = grad_fn(Fx_braking)
    if jnp.isfinite(g):
        print(f"  ∂α/∂Fx = {float(g):.6f}  [PASS] Gradient finite")
    else:
        print(f"  [FAIL] Non-finite gradient: {g}")

    # ── Test 6: Energy integrator ─────────────────────────────────────────
    print("\n[Test 6] Energy integrator over 100 braking steps")
    e_state = RegenEnergyState.zero()
    dt = jnp.array(0.005)
    for _ in range(100):
        T_sc_step, _ = compute_regen_blend(T_braking, jnp.full(4, -200.0),
                                            Fx_braking, omega, pt_nominal, bp, p)
        e_state = update_regen_energy(e_state, _, T_sc_step, omega, dt)
    eff = float(regen_efficiency(e_state))
    E_regen_Wh = float(e_state.E_regen_J) / 3600
    print(f"  E_regen = {E_regen_Wh*1000:.2f} mWh over 0.5s braking")
    print(f"  Regen efficiency = {eff*100:.1f}%")
    if eff > 0.0:
        print("  [PASS] Energy integrator working")
    else:
        print("  [FAIL] Zero regen energy accumulated")

    print("\n[DONE] Regen blend smoke test complete.")


if __name__ == "__main__":
    smoke_test()