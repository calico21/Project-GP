#!/usr/bin/env python3
# experiments/glrk_convergence.py
# Project-GP — GLRK-4 Convergence Verification
# ═══════════════════════════════════════════════════════════════════════════════
"""
Verifies 4th-order convergence of the 2-stage Gauss-Legendre RK4 integrator.

Method:
    1. Pick a reference state and compute solutions at h, h/2, h/4, h/8
    2. Use h/16 as "truth" reference
    3. Compute E(h) = ‖x_h - x_ref‖ for each h
    4. Compute convergence order p = log(E₁/E₂) / log(h₁/h₂)
    5. Target: p ≈ 4

Outputs:
    results/glrk_convergence.csv
    figs/glrk_convergence.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np


def run_convergence_study(
    n_steps: int = 200,
    h_base: float = 0.01,
    refinement_factors: tuple = (1, 2, 4, 8),
    ref_factor: int = 16,
    seed: int = 42,
) -> dict:
    """
    Run the GLRK-4 convergence study.

    Returns:
        dict with step_sizes, errors, convergence_orders
    """
    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from config.tire_coeffs import tire_coeffs as TC

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)

    # ── Reference initial state ──────────────────────────────────────────────
    x0 = vehicle.make_initial_state(vx0=20.0)
    setup = jnp.ones(28) * 0.5  # Mid-range setup

    # Constant control input (mild steer + moderate throttle)
    u = jnp.array([0.03, 10.0, 10.0, 10.0, 10.0, 0.0])

    # ── Compute reference solution at h/16 ──────────────────────────────────
    h_ref = h_base / ref_factor
    n_ref = n_steps * ref_factor

    print(f"  Computing reference solution (h = {h_ref:.5f}, {n_ref} steps)...")
    x_ref = _rollout(vehicle, x0, u, setup, h_ref, n_ref)

    # ── Compute solutions at each refinement level ───────────────────────────
    results = {"step_sizes": [], "errors": [], "states_final": []}

    for factor in refinement_factors:
        h = h_base / factor
        n = n_steps * factor
        print(f"  Computing h = {h:.5f} ({n} steps)...")

        x_final = _rollout(vehicle, x0, u, setup, h, n)
        error = float(jnp.linalg.norm(x_final[:28] - x_ref[:28]))

        results["step_sizes"].append(h)
        results["errors"].append(error)
        results["states_final"].append(x_final)

    # ── Compute convergence orders ───────────────────────────────────────────
    orders = []
    for i in range(len(results["errors"]) - 1):
        e1 = results["errors"][i]
        e2 = results["errors"][i + 1]
        h1 = results["step_sizes"][i]
        h2 = results["step_sizes"][i + 1]
        if e1 > 1e-15 and e2 > 1e-15:
            p = np.log(e1 / e2) / np.log(h1 / h2)
        else:
            p = float("nan")
        orders.append(p)
    results["convergence_orders"] = orders

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n  Step size     Error            Order")
    print("  " + "─" * 50)
    for i, (h, e) in enumerate(zip(results["step_sizes"], results["errors"])):
        order_str = f"{orders[i-1]:.2f}" if i > 0 and i <= len(orders) else "  —"
        print(f"  {h:10.5f}    {e:.6e}    {order_str}")

    mean_order = np.nanmean(orders) if orders else float("nan")
    print(f"\n  Mean convergence order: {mean_order:.2f}  (target: 4.00)")

    return results


def _rollout(vehicle, x0, u, setup, dt, n_steps):
    """Roll out n_steps with given dt."""
    x = x0
    for _ in range(n_steps):
        x = vehicle.simulate_step(x, u, setup, dt=dt, n_substeps=1)
    return x


def save_results(results: dict, out_dir: Path = None):
    """Save convergence data to CSV and generate figure."""
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
        writer.writerow(["step_size", "error", "convergence_order"])
        for i, (h, e) in enumerate(zip(results["step_sizes"], results["errors"])):
            order = results["convergence_orders"][i - 1] if i > 0 else ""
            writer.writerow([h, e, order])
    print(f"  Saved: {csv_path}")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        hs = np.array(results["step_sizes"])
        es = np.array(results["errors"])

        ax.loglog(hs, es, "o-", color="#e74c3c", linewidth=2, markersize=8,
                  label="GLRK-4 error")

        # Reference 4th-order slope
        if len(hs) >= 2 and es[0] > 0:
            h_ref_line = np.linspace(hs.min() * 0.8, hs.max() * 1.2, 50)
            e_ref_line = es[0] * (h_ref_line / hs[0]) ** 4
            ax.loglog(h_ref_line, e_ref_line, "--", color="#95a5a6", linewidth=1.5,
                      label="O(h⁴) reference")

        ax.set_xlabel("Step size h [s]", fontsize=12)
        ax.set_ylabel("‖x_h - x_ref‖", fontsize=12)
        ax.set_title("GLRK-4 Convergence Verification", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Annotate convergence orders
        for i, order in enumerate(results["convergence_orders"]):
            if not np.isnan(order):
                h_mid = np.sqrt(hs[i] * hs[i + 1])
                e_mid = np.sqrt(es[i] * es[i + 1])
                ax.annotate(f"p={order:.2f}", (h_mid, e_mid),
                           fontsize=10, fontweight="bold", color="#2c3e50",
                           ha="center", va="bottom")

        plt.tight_layout()
        fig.savefig(figs_dir / "glrk_convergence.png", dpi=150, bbox_inches="tight")
        fig.savefig(figs_dir / "glrk_convergence.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figs_dir / 'glrk_convergence.png'}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping figure")


if __name__ == "__main__":
    print("=" * 72)
    print("  PROJECT-GP  ·  GLRK-4 Convergence Verification")
    print("=" * 72)

    results = run_convergence_study()
    save_results(results)

    mean_order = np.nanmean(results["convergence_orders"])
    if abs(mean_order - 4.0) < 1.0:
        print(f"\n  ✓ Convergence order {mean_order:.2f} ≈ 4  — PASS")
    else:
        print(f"\n  ✗ Convergence order {mean_order:.2f} ≠ 4  — INVESTIGATE")
