#!/usr/bin/env python3
# experiments/glrk_convergence.py
# Project-GP — GLRK-4 Convergence Verification (Phase 1.5 · A1)
# ═══════════════════════════════════════════════════════════════════════════════
"""
3-tier convergence study for the 2-stage Gauss–Legendre RK4 integrator.

  Tier A  Harmonic oscillator (analytic reference)   → must give p ≈ 4.0
  Tier B  Mechanical subsystem [0:28] only           → isolates mech ODE
  Tier C  Full 108-DOF system                        → diagnostic

Uses float64 throughout and extended refinement h → h/32 with Richardson
self-convergence in addition to reference comparison.

Outputs:
    results/glrk_convergence.csv
    figs/glrk_convergence.png
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# §0  Pure GLRK-4 integrator (Butcher tableau only, no physics, no clips)
# ═══════════════════════════════════════════════════════════════════════════════

def _glrk4_pure_step(f, x, dt, n_newton=8, return_residual=False):
    """
    Single GLRK-4 step for dx/dt = f(x).

    2-stage Gauss–Legendre with Butcher coefficients:
        c  = [1/2 − √3/6,  1/2 + √3/6]
        A  = [[1/4,            1/4 − √3/6],
              [1/4 + √3/6,     1/4        ]]
        b  = [1/2, 1/2]

    Uses fixed-point (Picard) iteration — sufficient for
    convergence studies where dt is small relative to the Lipschitz constant.
    """
    sqrt3 = jnp.sqrt(3.0)
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    b1, b2 = 0.5, 0.5

    # Initial guess: explicit Euler slopes
    k1 = f(x)
    k2 = k1

    def iterate(carry, _):
        k1_, k2_ = carry
        x1 = x + dt * (a11 * k1_ + a12 * k2_)
        x2 = x + dt * (a21 * k1_ + a22 * k2_)
        return (f(x1), f(x2)), None

    (k1, k2), _ = jax.lax.scan(iterate, (k1, k2), None, length=n_newton)
    x1 = x + dt * (a11 * k1 + a12 * k2)
    x2 = x + dt * (a21 * k1 + a22 * k2)
    residual = jnp.linalg.norm(jnp.concatenate((k1 - f(x1), k2 - f(x2))))
    x_next = x + dt * (b1 * k1 + b2 * k2)
    return (x_next, residual) if return_residual else x_next


def _glrk4_rollout(f, x0, dt, n_steps, n_newton=8, diagnostics=False):
    """Roll out n_steps of GLRK-4."""
    if not diagnostics:
        def body(x, _):
            return _glrk4_pure_step(f, x, dt, n_newton), None
        x_final, _ = jax.lax.scan(body, x0, None, length=n_steps)
        return x_final

    def body(x, _):
        x_next, residual = _glrk4_pure_step(f, x, dt, n_newton, return_residual=True)
        return x_next, residual
    x_final, residuals = jax.lax.scan(body, x0, None, length=n_steps)
    return x_final, jnp.max(residuals), jnp.mean(residuals)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Tier A — Harmonic Oscillator (analytic reference)
# ═══════════════════════════════════════════════════════════════════════════════

def _harmonic_rhs(x):
    """dx/dt for simple harmonic oscillator: dq/dt = p, dp/dt = -q."""
    q, p = x[0], x[1]
    return jnp.array([p, -q])


def _harmonic_exact(x0, t):
    """Exact solution of simple harmonic oscillator."""
    q0, p0 = x0[0], x0[1]
    q = q0 * jnp.cos(t) + p0 * jnp.sin(t)
    p = -q0 * jnp.sin(t) + p0 * jnp.cos(t)
    return jnp.array([q, p])


def run_tier_a(
    T_final: float = 2.0,
    h_base: float = 0.1,
    refinement_factors: tuple = (1, 2, 4, 8, 16, 32),
) -> dict:
    """Tier A: harmonic oscillator with analytic reference."""
    print("\n" + "─" * 60)
    print("  TIER A — Harmonic Oscillator (analytic exact solution)")
    print("─" * 60)

    x0 = jnp.array([1.0, 0.0], dtype=jnp.float64)  # q=1, p=0
    x_exact = _harmonic_exact(x0, T_final)

    results = {"step_sizes": [], "errors": []}

    for factor in refinement_factors:
        dt = h_base / factor
        n_steps = int(round(T_final / dt))
        x_num = _glrk4_rollout(_harmonic_rhs, x0, dt, n_steps)
        err = float(jnp.linalg.norm(x_num - x_exact))
        results["step_sizes"].append(float(dt))
        results["errors"].append(err)
        print(f"    h = {dt:.6f}  ({n_steps:6d} steps)  error = {err:.6e}")

    # Convergence orders from consecutive pairs
    orders = _compute_orders(results["step_sizes"], results["errors"])
    results["orders"] = orders
    results["richardson_orders"] = _richardson_orders(results["errors"])

    _print_orders("Tier A", orders, results["richardson_orders"])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Tier B — Mechanical Subsystem Only (28 DOF)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier_b(
    T_final: float = 0.5,
    h_base: float = 0.01,
    refinement_factors: tuple = (1, 2, 4, 8, 16, 32),
) -> dict:
    """Tier B: vehicle mechanical subsystem [0:28], no aux states."""
    print("\n" + "─" * 60)
    print("  TIER B — Mechanical Subsystem (28 DOF, no aux)")
    print("─" * 60)

    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from config.tire_coeffs import tire_coeffs as TC

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)

    x0_full = vehicle.make_initial_state(vx0=15.0).astype(jnp.float64)
    setup = (jnp.ones(28) * 0.5).astype(jnp.float64)
    u = jnp.array([0.02, 8.0, 8.0, 8.0, 8.0, 0.0], dtype=jnp.float64)

    # RHS that only uses mechanical DOFs — create full 108-state with
    # zeroed aux and extract first 28 derivatives
    def mech_rhs(x28):
        x108 = jnp.zeros(108, dtype=jnp.float64).at[:28].set(x28)
        # Copy equilibrium aux states
        x108 = x108.at[28:56].set(x0_full[28:56])
        x108 = x108.at[56:72].set(x0_full[56:72])
        x108 = x108.at[72:84].set(x0_full[72:84])
        x108 = x108.at[84:108].set(x0_full[84:108])
        dx = vehicle._compute_derivatives(x108, u, setup)
        return dx[:28]

    x0_mech = x0_full[:28]

    # Reference solution at finest resolution
    h_ref = h_base / refinement_factors[-1]
    n_ref = int(round(T_final / h_ref))
    print(f"    Computing reference at h/{refinement_factors[-1]} ({n_ref} steps)...")
    x_ref = _glrk4_rollout(mech_rhs, x0_mech, h_ref, n_ref, n_newton=6)

    results = {"step_sizes": [], "errors": []}

    for factor in refinement_factors[:-1]:
        dt = h_base / factor
        n_steps = int(round(T_final / dt))
        x_num = _glrk4_rollout(mech_rhs, x0_mech, dt, n_steps, n_newton=6)
        err = float(jnp.linalg.norm(x_num - x_ref))
        results["step_sizes"].append(float(dt))
        results["errors"].append(err)
        print(f"    h = {dt:.6f}  ({n_steps:6d} steps)  error = {err:.6e}")

    orders = _compute_orders(results["step_sizes"], results["errors"])
    results["orders"] = orders
    results["richardson_orders"] = _richardson_orders(results["errors"])

    _print_orders("Tier B", orders, results["richardson_orders"])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Tier C — Full 108-DOF System (diagnostic)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier_c(
    T_final: float = 0.2,
    h_base: float = 0.005,
    refinement_factors: tuple = (1, 2, 4, 8, 16, 32),
) -> dict:
    """Tier C: full 108-DOF system — diagnostic, documents order limitation."""
    print("\n" + "─" * 60)
    print("  TIER C — Full 108-DOF System (diagnostic)")
    print("─" * 60)

    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from config.tire_coeffs import tire_coeffs as TC

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)

    x0 = vehicle.make_initial_state(vx0=15.0).astype(jnp.float64)
    setup = (jnp.ones(28) * 0.5).astype(jnp.float64)
    u = jnp.array([0.02, 8.0, 8.0, 8.0, 8.0, 0.0], dtype=jnp.float64)

    def full_rhs(x108):
        return vehicle._compute_derivatives(x108, u, setup)

    # Reference solution at finest resolution
    h_ref = h_base / refinement_factors[-1]
    n_ref = int(round(T_final / h_ref))
    print(f"    Computing reference at h/{refinement_factors[-1]} ({n_ref} steps)...")
    x_ref = _glrk4_rollout(full_rhs, x0, h_ref, n_ref, n_newton=8)

    results = {"step_sizes": [], "errors": [], "errors_mech": [], "errors_aux": [],
               "max_stage_residual": [], "mean_stage_residual": [], "picard_iterations": 8}

    for factor in refinement_factors[:-1]:
        dt = h_base / factor
        n_steps = int(round(T_final / dt))
        x_num, max_residual, mean_residual = _glrk4_rollout(
            full_rhs, x0, dt, n_steps, n_newton=8, diagnostics=True)

        err_total = float(jnp.linalg.norm(x_num - x_ref))
        err_mech  = float(jnp.linalg.norm(x_num[:28] - x_ref[:28]))
        err_aux   = float(jnp.linalg.norm(x_num[28:] - x_ref[28:]))

        results["step_sizes"].append(float(dt))
        results["errors"].append(err_total)
        results["errors_mech"].append(err_mech)
        results["errors_aux"].append(err_aux)
        results["max_stage_residual"].append(float(max_residual))
        results["mean_stage_residual"].append(float(mean_residual))
        print(f"    h = {dt:.6f}  ({n_steps:6d} steps)"
              f"  err_total = {err_total:.6e}"
              f"  err_mech = {err_mech:.6e}"
              f"  err_aux = {err_aux:.6e}"
              f"  stage_res(max/mean)={float(max_residual):.2e}/{float(mean_residual):.2e}")

    orders = _compute_orders(results["step_sizes"], results["errors"])
    orders_mech = _compute_orders(results["step_sizes"], results["errors_mech"])
    orders_aux  = _compute_orders(results["step_sizes"], results["errors_aux"])
    results["orders"] = orders
    results["orders_mech"] = orders_mech
    results["orders_aux"] = orders_aux
    results["richardson_orders"] = _richardson_orders(results["errors"])

    _print_orders("Tier C total", orders, results["richardson_orders"])
    print(f"    Mech orders: {[f'{o:.2f}' for o in orders_mech]}")
    print(f"    Aux orders:  {[f'{o:.2f}' for o in orders_aux]}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_orders(step_sizes, errors):
    """Standard convergence order from consecutive pairs."""
    orders = []
    for i in range(len(errors) - 1):
        e1, e2 = errors[i], errors[i + 1]
        h1, h2 = step_sizes[i], step_sizes[i + 1]
        if e1 > 1e-15 and e2 > 1e-15:
            p = np.log(e1 / e2) / np.log(h1 / h2)
        else:
            p = float("nan")
        orders.append(p)
    return orders


def _richardson_orders(errors):
    """
    Self-convergence (Richardson) order: assumes halving.
    p = log₂(|e_i − e_{i+1}| / |e_{i+1} − e_{i+2}|)
    """
    orders = []
    for i in range(len(errors) - 2):
        num = abs(errors[i] - errors[i + 1])
        den = abs(errors[i + 1] - errors[i + 2])
        if num > 1e-15 and den > 1e-15:
            orders.append(np.log2(num / den))
        else:
            orders.append(float("nan"))
    return orders


def _print_orders(label, orders, richardson_orders):
    mean_o = np.nanmean(orders) if orders else float("nan")
    mean_r = np.nanmean(richardson_orders) if richardson_orders else float("nan")
    print(f"\n    {label}:")
    print(f"      Pair orders:       {[f'{o:.2f}' for o in orders]}")
    print(f"      Richardson orders: {[f'{o:.2f}' for o in richardson_orders]}")
    print(f"      Mean pair order:   {mean_o:.2f}")
    print(f"      Mean Richardson:   {mean_r:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Save results
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(tier_a, tier_b, tier_c, out_dir=None):
    if out_dir is None:
        out_dir = _ROOT
    results_dir = out_dir / "results"
    figs_dir = out_dir / "figs"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    # ── CSV ──
    csv_path = results_dir / "glrk_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "step_size", "error", "convergence_order",
                         "max_stage_residual", "mean_stage_residual", "picard_iterations"])
        for label, res in [("A_harmonic", tier_a), ("B_mech28", tier_b), ("C_full108", tier_c)]:
            for i, (h, e) in enumerate(zip(res["step_sizes"], res["errors"])):
                order = res["orders"][i - 1] if i > 0 and i <= len(res["orders"]) else ""
                writer.writerow([label, h, e, order,
                                 res.get("max_stage_residual", [""] * len(res["errors"]))[i],
                                 res.get("mean_stage_residual", [""] * len(res["errors"]))[i],
                                 res.get("picard_iterations", "")])
    print(f"\n  Saved: {csv_path}")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (label, res, color) in zip(axes, [
            ("Tier A: Harmonic", tier_a, "#e74c3c"),
            ("Tier B: Mech-28", tier_b, "#3498db"),
            ("Tier C: Full-108", tier_c, "#2ecc71"),
        ]):
            hs = np.array(res["step_sizes"])
            es = np.array(res["errors"])

            ax.loglog(hs, es, "o-", color=color, linewidth=2, markersize=8,
                      label="Measured error")

            # Reference O(h⁴) slope
            if len(hs) >= 2 and es[0] > 0:
                h_line = np.linspace(hs.min() * 0.7, hs.max() * 1.3, 50)
                e_line = es[0] * (h_line / hs[0]) ** 4
                ax.loglog(h_line, e_line, "--", color="#95a5a6", linewidth=1.5,
                          label="O(h⁴) ref")

            # Reference O(h²) slope
            if len(hs) >= 2 and es[0] > 0:
                e_line2 = es[0] * (h_line / hs[0]) ** 2
                ax.loglog(h_line, e_line2, ":", color="#bdc3c7", linewidth=1.2,
                          label="O(h²) ref")

            # Annotate orders
            for i, order in enumerate(res["orders"]):
                if not np.isnan(order) and i < len(hs) - 1:
                    h_mid = np.sqrt(hs[i] * hs[i + 1])
                    e_mid = np.sqrt(es[i] * es[i + 1])
                    ax.annotate(f"p={order:.2f}", (h_mid, e_mid),
                               fontsize=9, fontweight="bold", color="#2c3e50",
                               ha="center", va="bottom")

            ax.set_xlabel("Step size h [s]")
            ax.set_ylabel("‖x − x_ref‖")
            ax.set_title(label, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Project-GP · GLRK-4 Convergence Verification (Phase 1.5)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figs_dir / f"glrk_convergence.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figs_dir / 'glrk_convergence.png'}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping figure")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  PROJECT-GP  ·  GLRK-4 Convergence (Phase 1.5 · 3-tier)")
    print("=" * 72)

    tier_a = run_tier_a()
    tier_b = run_tier_b()
    tier_c = run_tier_c()

    save_results(tier_a, tier_b, tier_c)

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    for label, res, target in [
        ("Tier A (harmonic)", tier_a, 4.0),
        ("Tier B (mech-28)",  tier_b, 3.5),
        ("Tier C (full-108)", tier_c, None),
    ]:
        mean_o = np.nanmean(res["orders"]) if res["orders"] else float("nan")
        if target is not None:
            status = "PASS" if abs(mean_o - target) < 0.5 else "INVESTIGATE"
            print(f"  {label}: mean order = {mean_o:.2f}  (target ≥ {target:.1f}) → {status}")
        else:
            print(f"  {label}: mean order = {mean_o:.2f}  (diagnostic)")
    print("=" * 72)
