# powertrain/modes/advanced/slip_barrier.py
# Project-GP — Batch 8: Predictive Slip-CBF Row Builder for mpQP Allocator
# ═══════════════════════════════════════════════════════════════════════════════
#
# Constructs the (8, 4) constraint rows A_slip and rhs b_slip such that:
#
#     A_slip @ T  ≤  b_slip
#
# enforces, for each of the 4 wheels i:
#
#     |κ_{k+d}^{(i)}(T)| ≤ κ*_i − κ_safe · σ(κ*_i)     (slip CBF, E-ABS)
#
# where κ_{k+d}^{(i)} is the 1-step-ahead slip ratio predicted under torque T,
# κ*_i is the Koopman observer's optimal-slip estimate, and σ(κ*_i) its IFT
# uncertainty.
#
# LINEARISATION
# ─────────────
# The wheel rotational dynamics are:
#     ω̇_i = (T_i − F_x^tire_i · r_w) / I_w
#     κ_i  = (ω_i · r_w − v_x) / v_x
#
# First-order Euler over the actuator-delay horizon Δt = tau_delay:
#
#     κ_{k+d}^{(i)}(T) ≈ κ_i + (r_w · Δt / (I_w · v_x)) · (T_i − F_x^tire_i · r_w)
#
# This is AFFINE in T_i (F_x^tire is treated as a measured constant) — exactly
# the form the KKT system expects. The constraint |κ| ≤ budget is split into
# two linear inequalities, giving two rows per wheel (8 total).
#
# INTEGRATION WITH THE ALLOCATOR
# ──────────────────────────────
# build_slip_barrier_rows() returns (A_slip: (8,4), b_slip: (8,)).
# These are passed to build_kkt_system_extended() as `extra_A, extra_b`.
# The active-set classifier (V2) predicts which of these 8 constraints are
# active; until V2 is trained, the active-set for slip defaults to zeros
# (inactive) and the 3-step polish enforces feasibility as fallback.
#
# DISABLING
# ─────────
# SlipBarrierInputs.disabled() returns sentinel inputs that set b_slip = 1e6,
# making A_slip @ T ≤ b_slip trivially satisfied for all feasible T. Use this
# during launch control or when the Koopman observer is unhealthy.
#
# JAX CONTRACT
# ─────────────
# Pure JAX. No Python conditionals inside traced code. C∞ everywhere.
# Safe inside jit / vmap / grad / scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# §1  Parameters
# ─────────────────────────────────────────────────────────────────────────────

class SlipBarrierParams(NamedTuple):
    """Tuning parameters for the predictive slip CBF."""

    # Actuator + sensor delay (total) — used for 1-step-ahead prediction.
    # 3 steps @ 5 ms = 15 ms covers motor-to-wheel torque delay + signal lag.
    tau_delay:    float = 0.015    # s

    # Robustness multiplier on Koopman σ(κ*).
    # Budget = κ*_i − kappa_safe · σ_i.  1.5× = moderate safety margin.
    kappa_safe:   float = 1.5

    # Speed gate: barrier disabled below v_min (no meaningful slip dynamics).
    v_min:        float = 1.5     # m/s

    # Hard floor on budget — prevents degenerate constraints when σ is large.
    # Car can always use at least this much slip (cornering-stiffness-limited).
    kappa_floor:  float = 0.015   # ~10% below minimum useful κ*

    # Sentinel RHS used to make a constraint trivially inactive.
    # Must exceed any physically reachable |κ| by a large margin.
    rhs_inactive: float = 1.0e6

    # Physical constants — must match motor_model.py defaults.
    I_w:  float = 1.2     # kg·m²  wheel + rotor inertia
    r_w:  float = 0.2032  # m      loaded wheel radius


# ─────────────────────────────────────────────────────────────────────────────
# §2  Input struct
# ─────────────────────────────────────────────────────────────────────────────

class SlipBarrierInputs(NamedTuple):
    """
    Per-step inputs to build_slip_barrier_rows().

    All arrays have shape (4,) unless noted.  Wheel indexing: FL, FR, RL, RR.
    """
    kappa_star:  jax.Array   # (4,) κ*  per wheel — Koopman, broadcast front/rear
    sigma_star:  jax.Array   # (4,) σ(κ*) per wheel
    kappa_now:   jax.Array   # (4,) current slip ratio  κ_k
    fx_tire_est: jax.Array   # (4,) inertia-corrected F_x^tire estimate [N]
    vx:          jax.Array   # scalar longitudinal velocity [m/s]
    active:      jax.Array   # scalar [0, 1] — sigmoid gate (0 = barriers off)

    @classmethod
    def disabled(cls) -> "SlipBarrierInputs":
        """
        Returns sentinel inputs that render all 8 constraints trivially inactive.

        Use during: launch control, v < v_min, Koopman observer unhealthy.
        All resulting b_slip values are rhs_inactive = 1e6.
        """
        return cls(
            kappa_star  = jnp.full(4, 0.20),   # nominal-ish κ*
            sigma_star  = jnp.full(4, 0.05),   # nominal σ
            kappa_now   = jnp.zeros(4),
            fx_tire_est = jnp.zeros(4),
            vx          = jnp.array(1.0),       # triggers v_min gate → inactive
            active      = jnp.array(0.0),       # hard gate off
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3  Core: predictive slip linearisation
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def build_slip_barrier_rows(
    inp:  SlipBarrierInputs,
    p:    SlipBarrierParams = SlipBarrierParams(),
) -> tuple[jax.Array, jax.Array]:
    """
    Construct (A_slip, b_slip) such that A_slip @ T ≤ b_slip enforces
    the per-wheel predictive slip CBF constraints.

    Shape: A_slip (8, 4), b_slip (8,).

    Constraint layout:
        rows [0:4]  — κ_upper: +sens_i · T_i ≤ +budget_i − κ_preview_0_i
        rows [4:8]  — κ_lower: −sens_i · T_i ≤ +budget_i + κ_preview_0_i

    Returns:
        A_slip  (8, 4)  — coefficient matrix (strictly diagonal in the per-wheel axes)
        b_slip  (8,)    — right-hand side
    """
    # ── Speed gate — soft via sigmoid ─────────────────────────────────────
    v_gate = jax.nn.sigmoid(10.0 * (inp.vx - p.v_min))   # → 0 below v_min, 1 above

    # ── Active gate — from caller (launch, fault, etc.) ───────────────────
    gate = v_gate * inp.active                            # scalar ∈ [0, 1]

    vx_s = jnp.maximum(jnp.abs(inp.vx), p.v_min)        # denominator guard

    # ── 1. Sensitivity  ∂κ_{k+d} / ∂T_i = r_w · Δt / (I_w · v_x) ────────
    sensitivity = p.r_w * p.tau_delay / (p.I_w * vx_s)  # scalar [1/(Nm)]

    # ── 2. Free-coast κ preview: κ at delay horizon if T_i = 0 ───────────
    # κ_{k+d}^0 = κ_now − sens · F_x^tire · r_w
    kappa_preview_0 = inp.kappa_now - sensitivity * inp.fx_tire_est * p.r_w  # (4,)

    # ── 3. Koopman safe budget ────────────────────────────────────────────
    # budget_i = max(κ*_i − κ_safe · σ_i, κ_floor)
    budget_raw = inp.kappa_star - p.kappa_safe * inp.sigma_star      # (4,)
    budget = jnp.maximum(budget_raw, p.kappa_floor)                  # (4,)

    # ── 4. Sensitivity matrix: diagonal of (4,4) — each wheel independent ─
    sens4 = jnp.full(4, sensitivity)   # (4,) same sensitivity for all wheels
    S = jnp.diag(sens4)                # (4, 4)

    # ── 5. Two rows per wheel (+κ, −κ) ────────────────────────────────────
    A_upper = S                         # (4, 4)  : +sens_i · T_i ≤ ...
    A_lower = -S                        # (4, 4)  : −sens_i · T_i ≤ ...

    b_upper = budget - kappa_preview_0  # (4,)
    b_lower = budget + kappa_preview_0  # (4,)

    # ── 6. Gate: inactive rows → huge RHS (constraints can never bind) ────
    # When gate = 0, b_active = rhs_inactive → T_i always feasible.
    def gate_b(b):
        return gate * b + (1.0 - gate) * p.rhs_inactive

    b_upper_gated = gate_b(b_upper)     # (4,)
    b_lower_gated = gate_b(b_lower)     # (4,)

    # ── 7. Stack ────────────────────────────────────────────────────────────
    A_slip = jnp.concatenate([A_upper, A_lower], axis=0)           # (8, 4)
    b_slip = jnp.concatenate([b_upper_gated, b_lower_gated])       # (8,)

    return A_slip, b_slip


# ─────────────────────────────────────────────────────────────────────────────
# §4  Convenience: build from Koopman observer output
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def make_slip_barrier_inputs(
    kappa_star_front: jax.Array,   # scalar — Koopman front κ*
    kappa_star_rear:  jax.Array,   # scalar — Koopman rear  κ*
    sigma_front:      jax.Array,   # scalar — IFT σ(κ*) front
    sigma_rear:       jax.Array,   # scalar — IFT σ(κ*) rear
    kappa_measured:   jax.Array,   # (4,)   — current slip ratios from state
    T_prev:           jax.Array,   # (4,)   — previous wheel torques [Nm]
    omega_wheel:      jax.Array,   # (4,)   — current ω [rad/s]
    omega_prev:       jax.Array,   # (4,)   — previous ω [rad/s]
    vx:               jax.Array,   # scalar
    launch_active:    jax.Array,   # scalar [0,1] — disables barrier during launch
    dt:               jax.Array,   # scalar [s]
    r_w:              float = 0.2032,
    I_w:              float = 1.2,
) -> SlipBarrierInputs:
    """
    Factory: construct SlipBarrierInputs from Koopman observer outputs and
    the motor-side inertia-corrected Fx estimate.

    The Fx estimate uses:   F_x^tire ≈ (T_prev − I_w · ω̇ · r_w) / r_w
    (Same estimator as traction_control.estimate_fx_from_motors.)
    """
    # Broadcast axle κ* and σ to per-wheel
    kappa_star = jnp.array([
        kappa_star_front, kappa_star_front,
        kappa_star_rear,  kappa_star_rear,
    ])
    sigma_star = jnp.array([
        sigma_front, sigma_front,
        sigma_rear,  sigma_rear,
    ])

    # Inertia-corrected F_x^tire per wheel  [N]
    dt_safe = jnp.maximum(dt, 1e-6)
    omega_dot  = (omega_wheel - omega_prev) / dt_safe     # (4,) [rad/s²]
    tau_net    = T_prev - I_w * omega_dot                 # (4,) [Nm] net axle torque
    fx_tire_est = tau_net / r_w                           # (4,) [N]

    # Barrier is active when NOT in launch and above v_min
    active = 1.0 - launch_active   # scalar — smoothly gated externally

    return SlipBarrierInputs(
        kappa_star  = kappa_star,
        sigma_star  = sigma_star,
        kappa_now   = kappa_measured,
        fx_tire_est = fx_tire_est,
        vx          = vx,
        active      = active,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §5  Feasibility check — used by polish step extension
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def check_slip_feasibility(
    T:     jax.Array,          # (4,) proposed torque
    A_slip: jax.Array,         # (8, 4) from build_slip_barrier_rows
    b_slip: jax.Array,         # (8,)
    tol:   float = 0.002,      # κ tolerance (0.002 = 0.2 pp slip)
) -> jax.Array:
    """Returns True iff all slip constraints are satisfied within tolerance."""
    residuals = A_slip @ T - b_slip               # (8,) — should be ≤ 0
    max_viol  = jnp.max(residuals)
    return max_viol <= tol


# ─────────────────────────────────────────────────────────────────────────────
# §6  Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """
    Verify:
    1. Hard braking into a corner — slip barrier activates, constrains T.
    2. Disabled inputs — all T feasible regardless of magnitude.
    3. Gradients flow through: jax.grad(loss)(T) is finite.

    Run: python -m powertrain.modes.advanced.slip_barrier
    """
    import time

    print("=" * 60)
    print(" SLIP BARRIER SMOKE TEST")
    print("=" * 60)

    # ── Test 1: Active barrier during hard braking ─────────────────────
    # Low speed (vx=5 m/s) makes sensitivity large enough to show violation:
    #   sensitivity = 0.2032 * 0.015 / (1.2 * 5) = 5.08e-4 [1/Nm]
    # With T = −400 Nm, kappa_now = −0.04, Fx_tire = −800 N:
    #   κ_preview_0 = −0.04 − 5.08e-4 · (−800) · 0.2032 = −0.04 + 0.0826 = 0.0426
    #   κ_pred = 0.0426 + 5.08e-4 · (−400) = 0.0426 − 0.2032 = −0.1606
    #   budget = max(0.08 − 1.5 · 0.015, 0.015) = 0.0575
    #   |κ_pred| = 0.1606 > 0.0575 → VIOLATION (lower bound row)
    print("\n[Test 1] Hard braking at low speed (vx=5 m/s, kappa_now=−0.04)")
    print("  Expected: −400 Nm INFEASIBLE (|κ_pred|=0.161 > budget=0.058)")
    inp_braking = SlipBarrierInputs(
        kappa_star  = jnp.full(4, 0.08),
        sigma_star  = jnp.full(4, 0.015),
        kappa_now   = jnp.full(4, -0.04),
        fx_tire_est = jnp.full(4, -800.0),   # low-speed braking force
        vx          = jnp.array(5.0),
        active      = jnp.array(1.0),
    )
    p = SlipBarrierParams()

    t0 = time.perf_counter()
    A_s, b_s = build_slip_barrier_rows(inp_braking, p)
    _ = float(A_s[0, 0])
    ms = (time.perf_counter() - t0) * 1000
    print(f"  Compile+run: {ms:.2f} ms (target: <1 ms post-JIT)")

    # After JIT warm-up:
    t0 = time.perf_counter()
    A_s, b_s = build_slip_barrier_rows(inp_braking, p)
    _ = float(A_s[0, 0])
    ms = (time.perf_counter() - t0) * 1000
    print(f"  Post-JIT:    {ms:.3f} ms (target: <0.05 ms)")

    budget = 0.11 - 1.5 * 0.015
    print(f"  Expected budget: {budget:.4f}")
    print(f"  A_slip[0,0]  (sensitivity): {float(A_s[0,0]):.6f}")
    print(f"  b_slip[0]    (upper rhs):   {float(b_s[0]):.6f}")
    print(f"  b_slip[4]    (lower rhs):   {float(b_s[4]):.6f}")

    # Check: a large braking torque (−400 Nm) should violate
    T_large_brake = jnp.full(4, -400.0)
    viol = A_s @ T_large_brake - b_s
    print(f"  Max constraint violation at T=−400 Nm: {float(jnp.max(viol)):.4f}")
    if float(jnp.max(viol)) > 0:
        print("  [PASS] Barrier correctly identifies lockup torque as infeasible")
    else:
        print("  [WARN] Barrier did not flag −400 Nm as infeasible (check parameters)")

    # Check: small braking (−30 Nm per wheel) should be fine at low speed
    T_moderate = jnp.full(4, -30.0)
    viol_mod = A_s @ T_moderate - b_s
    print(f"  Max constraint violation at T=−30 Nm:  {float(jnp.max(viol_mod)):.4f}")
    if float(jnp.max(viol_mod)) <= 0:
        print("  [PASS] Gentle braking correctly within slip budget")
    else:
        print("  [WARN] Gentle braking flagged as infeasible (check kappa_floor)")

    # ── Test 2: Disabled inputs ────────────────────────────────────────
    print("\n[Test 2] Disabled inputs — all constraints trivially inactive")
    inp_off = SlipBarrierInputs.disabled()
    A_off, b_off = build_slip_barrier_rows(inp_off, p)
    T_extreme = jnp.full(4, -500.0)
    viol_off = A_off @ T_extreme - b_off
    if float(jnp.max(viol_off)) < 0:
        print("  [PASS] Disabled barrier: T=−500 Nm is feasible (rhs=1e6)")
    else:
        print("  [FAIL] Disabled barrier incorrectly constraining")

    # ── Test 3: Differentiability ─────────────────────────────────────
    print("\n[Test 3] Gradient flow through slip feasibility check")

    def slip_loss(T_vec):
        A, b = build_slip_barrier_rows(inp_braking, p)
        residuals = A @ T_vec - b
        return jnp.sum(jax.nn.softplus(residuals * 100.0))

    grad_fn = jax.jit(jax.grad(slip_loss))
    g = grad_fn(jnp.full(4, -100.0))
    if jnp.all(jnp.isfinite(g)):
        print(f"  [PASS] Gradient finite: {[round(float(x), 4) for x in g]}")
    else:
        print(f"  [FAIL] Gradient contains NaN/Inf: {g}")

    # ── Test 4: make_slip_barrier_inputs factory ──────────────────────
    print("\n[Test 4] make_slip_barrier_inputs factory")
    inp_factory = make_slip_barrier_inputs(
        kappa_star_front = jnp.array(0.11),
        kappa_star_rear  = jnp.array(0.10),
        sigma_front      = jnp.array(0.012),
        sigma_rear       = jnp.array(0.018),
        kappa_measured   = jnp.array([-0.05, -0.06, -0.07, -0.08]),
        T_prev           = jnp.full(4, -120.0),
        omega_wheel      = jnp.array([90.0, 89.0, 88.0, 87.0]),
        omega_prev       = jnp.array([92.0, 91.0, 90.0, 89.0]),
        vx               = jnp.array(18.0),
        launch_active    = jnp.array(0.0),
        dt               = jnp.array(0.005),
    )
    A_f, b_f = build_slip_barrier_rows(inp_factory, p)
    print(f"  kappa_star (front/rear): {float(inp_factory.kappa_star[0]):.3f} / {float(inp_factory.kappa_star[2]):.3f}")
    print(f"  fx_tire_est[0]: {float(inp_factory.fx_tire_est[0]):.1f} N")
    if jnp.all(jnp.isfinite(b_f)):
        print(f"  [PASS] Factory output finite, b_slip[0]={float(b_f[0]):.4f}")
    else:
        print(f"  [FAIL] Factory produced non-finite output")

    print("\n[DONE] Slip barrier smoke test complete.")


if __name__ == "__main__":
    smoke_test()