#!/usr/bin/env python3
# experiments/gradient_accuracy.py
# Project-GP — IFT Gradient Validation (Phase 1.5 · A2)
# ═══════════════════════════════════════════════════════════════════════════════
"""
Two-problem gradient validation:

  Problem 1  Toy implicit root with ANALYTIC gradient    → validates IFT concept
  Problem 2  Vehicle 1-step with FD plateau analysis     → validates production

Compares: analytic (P1), FD sweep, autodiff, explicit IFT.

Outputs:
    results/gradient_accuracy.csv
    figs/gradient_accuracy.png
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Problem 1 — Toy Implicit Root
# ═══════════════════════════════════════════════════════════════════════════════
#
#   F(z, θ) = z³ + θ·z − 1 = 0   defines z*(θ) implicitly.
#   Objective: J(θ) = z*(θ)²
#
#   IFT:  dz/dθ = −F_θ / F_z = −z / (3z² + θ)
#   Analytic: dJ/dθ = 2·z·(dz/dθ) = −2z² / (3z² + θ)
#

def _solve_cubic_root(theta, z_init=1.0, n_iter=50):
    """Solve z³ + θ·z − 1 = 0 via Newton."""
    z = jnp.array(z_init, dtype=jnp.float64)
    def body(z, _):
        F = z**3 + theta * z - 1.0
        Fz = 3.0 * z**2 + theta
        return z - F / Fz, None
    z, _ = jax.lax.scan(body, z, None, length=n_iter)
    return z


def _analytic_grad_toy(theta):
    """Analytic dJ/dθ for the toy problem."""
    z = _solve_cubic_root(theta)
    dz_dtheta = -z / (3.0 * z**2 + theta)
    return 2.0 * z * dz_dtheta


def _fd_grad_toy(theta, eps):
    """Central FD gradient for toy problem."""
    z_plus  = _solve_cubic_root(theta + eps)
    z_minus = _solve_cubic_root(theta - eps)
    J_plus  = z_plus**2
    J_minus = z_minus**2
    return (J_plus - J_minus) / (2.0 * eps)


def _autodiff_grad_toy(theta):
    """jax.grad through the solver (unrolled Newton)."""
    def J(th):
        z = _solve_cubic_root(th)
        return z**2
    return jax.grad(J)(theta)


def _explicit_ift_grad_toy(theta):
    """Explicit IFT: dJ/dθ = (∂J/∂z)·(−F_z⁻¹·F_θ)."""
    z = _solve_cubic_root(theta)
    # Partial derivatives of F(z,θ) = z³ + θ·z − 1
    F_z = 3.0 * z**2 + theta
    F_theta = z
    dz_dtheta = -F_theta / F_z
    # dJ/dz = 2z
    return 2.0 * z * dz_dtheta


def run_problem_1() -> dict:
    """Problem 1: toy implicit root gradient validation."""
    print("\n" + "─" * 60)
    print("  PROBLEM 1 — Toy Implicit Root (analytic gradient)")
    print("─" * 60)

    theta_vals = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float64)
    eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    results = {"theta": [], "analytic": [], "autodiff": [], "ift": [],
               "fd_eps": {eps: [] for eps in eps_values}}

    for theta in theta_vals:
        g_analytic = float(_analytic_grad_toy(theta))
        g_autodiff = float(_autodiff_grad_toy(theta))
        g_ift      = float(_explicit_ift_grad_toy(theta))

        results["theta"].append(float(theta))
        results["analytic"].append(g_analytic)
        results["autodiff"].append(g_autodiff)
        results["ift"].append(g_ift)

        err_auto = abs(g_autodiff - g_analytic) / (abs(g_analytic) + 1e-15)
        err_ift  = abs(g_ift - g_analytic) / (abs(g_analytic) + 1e-15)

        print(f"\n    θ = {float(theta):.1f}:")
        print(f"      Analytic:  {g_analytic:+.10e}")
        print(f"      Autodiff:  {g_autodiff:+.10e}  (rel err = {err_auto:.2e})")
        print(f"      Expl IFT:  {g_ift:+.10e}  (rel err = {err_ift:.2e})")

        for eps in eps_values:
            g_fd = float(_fd_grad_toy(theta, eps))
            results["fd_eps"][eps].append(g_fd)
            err_fd = abs(g_fd - g_analytic) / (abs(g_analytic) + 1e-15)
            print(f"      FD(ε={eps:.0e}): {g_fd:+.10e}  (rel err = {err_fd:.2e})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Problem 2 — Vehicle 1-step FD Plateau
# ═══════════════════════════════════════════════════════════════════════════════

def _fd_gradient_vehicl(f, x, eps):
    """Central FD gradient in float64."""
    x = jnp.asarray(x, dtype=jnp.float64)
    n = x.shape[0]
    grad = jnp.zeros(n, dtype=jnp.float64)
    for i in range(n):
        e = jnp.zeros_like(x).at[i].set(eps)
        fp = f(x + e)
        fm = f(x - e)
        grad = grad.at[i].set((fp - fm) / (2.0 * eps))
    return grad


def _vehicle_stage_residual(vehicle, x, u, setup, dt, stages):
    """Actual 216-variable GLRK stage equation F(z, theta)=0.

    This deliberately calls the same clipped RHS as the production step.  It
    therefore validates the implemented numerical map, rather than an ideal
    mechanical surrogate with different nonsmoothness or auxiliary dynamics.
    """
    sqrt3 = jnp.sqrt(3.0)
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    k1, k2 = stages.reshape(2, 108)
    x1 = x + dt * (a11 * k1 + a12 * k2)
    x2 = x + dt * (a21 * k1 + a22 * k2)
    f1 = jnp.clip(vehicle._compute_derivatives(x1, u, setup), -500.0, 500.0)
    f2 = jnp.clip(vehicle._compute_derivatives(x2, u, setup), -500.0, 500.0)
    return stages - jnp.concatenate((f1, f2))


def _solve_vehicle_stages(vehicle, x, u, setup, dt, n_iter=32, return_history=False):
    """The production Picard stage solve, exposed for residual/IFT checks."""
    z0 = jnp.tile(jnp.clip(vehicle._compute_derivatives(x, u, setup), -500., 500.), 2)

    def body(z, _):
        # z <- z - F(z), the same fixed-point update used in _glrk4_step.
        z_next = z - _vehicle_stage_residual(vehicle, x, u, setup, dt, z)
        residual = _vehicle_stage_residual(vehicle, x, u, setup, dt, z_next)
        return z_next, jnp.array([jnp.linalg.norm(residual), jnp.max(jnp.abs(residual))])
    z, history = jax.lax.scan(body, z0, None, length=n_iter)
    return (z, history) if return_history else z


def _vehicle_stage_output(x, stages, dt):
    """Exact output reconstruction used by _glrk4_step, including aux clip."""
    k1, k2 = stages.reshape(2, 108)
    x_next = x + 0.5 * dt * (k1 + k2)
    return x_next.at[28:108].set(jnp.clip(x_next[28:108], -1000.0, 1000.0))


def _stage_solution_diagnostics(vehicle, x, u, setup, dt):
    """Forward-only branch diagnostic for nominal and FD-perturbed solves."""
    stages, history = _solve_vehicle_stages(
        vehicle, x, u, setup, dt, return_history=True)
    residual = _vehicle_stage_residual(vehicle, x, u, setup, dt, stages)
    return {
        "stages": stages,
        "state": _vehicle_stage_output(x, stages, dt),
        "residual_2": float(jnp.linalg.norm(residual)),
        "residual_inf": float(jnp.max(jnp.abs(residual))),
        "history": history,
    }


def _explicit_vehicle_ift_gradient(vehicle, x, u, setup, n_setup_params, dt, x_target):
    """IFT derivative for one full 108-state vehicle step.

    The solve is intentionally explicit: F_z and F_theta are differentiated
    independently, then dz/dtheta = -F_z^{-1} F_theta is formed.  This is a
    diagnostic reference; it is only meaningful when the reported residual is
    small and F_z is reasonably conditioned.
    """
    theta0 = setup[:n_setup_params]

    def full_setup(theta):
        return setup.at[:n_setup_params].set(theta)

    def residual(z, theta):
        return _vehicle_stage_residual(vehicle, x, u, full_setup(theta), dt, z)

    z, residual_history = _solve_vehicle_stages(
        vehicle, x, u, full_setup(theta0), dt, return_history=True)
    Fz = jax.jacrev(residual, 0)(z, theta0)
    Ftheta = jax.jacrev(residual, 1)(z, theta0)
    dz_dtheta = jnp.linalg.solve(Fz, -Ftheta)

    b_stages = jnp.array([0.5, 0.5], dtype=x.dtype)
    dx_dtheta = dt * (b_stages[0] * dz_dtheta[:108] + b_stages[1] * dz_dtheta[108:])
    x_stage = _vehicle_stage_output(x, z, dt)
    grad_x = 2.0 * (x_stage[:28] - x_target)
    gradient = grad_x @ dx_dtheta[:28]
    cond = jnp.linalg.cond(Fz)
    residual_norm = jnp.linalg.norm(residual(z, theta0))
    return gradient, residual_norm, cond, z, residual_history


def run_problem_2(
    n_setup_params: int = 6,
    horizons: tuple = (1, 2, 5),
    eps_values: tuple = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    seed: int = 42,
) -> dict:
    """Problem 2: vehicle gradient with FD plateau analysis."""
    print("\n" + "─" * 60)
    print("  PROBLEM 2 — Vehicle Gradient (FD plateau + autodiff)")
    print("─" * 60)

    from models.vehicle_dynamics import (DifferentiableMultiBodyVehicle,
                                         DEFAULT_SETUP, SETUP_LB)
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from config.tire_coeffs import tire_coeffs as TC

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)
    dt = 0.005

    x0 = vehicle.make_initial_state(vx0=20.0).astype(jnp.float64)
    # Setup entries are physical SI quantities, not unit-cube optimisation
    # coordinates.  This deterministic, in-bounds point is also in the Picard
    # convergence regime; a random [0,1]^28 vector is physically invalid.
    setup = (0.5 * (DEFAULT_SETUP + SETUP_LB)).astype(jnp.float64)
    u = jnp.array([0.03, 10.0, 10.0, 10.0, 10.0, 0.0], dtype=jnp.float64)
    x_target = vehicle.make_initial_state(vx0=22.0).astype(jnp.float64)[:28]

    all_results = {}

    for horizon in horizons:
        print(f"\n  ── Horizon: {horizon} step(s) ({horizon * dt:.3f}s) ──")

        def objective(s_partial):
            s_full = setup.at[:n_setup_params].set(s_partial)
            x = x0
            for _ in range(horizon):
                x = vehicle.simulate_step(x, u, s_full, dt=dt, n_substeps=1)
            return jnp.sum((x[:28] - x_target) ** 2)

        s_test = setup[:n_setup_params]

        # ── FD plateau ──
        print("\n    FD plateau analysis:")
        fd_grads = {}
        fd_norms = {}
        for eps in eps_values:
            t0 = time.time()
            g = _fd_gradient_vehicl(objective, s_test, eps)
            elapsed = time.time() - t0
            fd_grads[eps] = g
            fd_norms[eps] = float(jnp.linalg.norm(g))
            print(f"      ε = {eps:.0e}: ‖g‖ = {fd_norms[eps]:.6e}  ({elapsed:.1f}s)")

        # ── Autodiff (production) ──
        t0 = time.time()
        g_auto = jax.grad(objective)(s_test)
        g_auto.block_until_ready()
        t_auto = time.time() - t0
        norm_auto = float(jnp.linalg.norm(g_auto))
        print(f"\n    Production autodiff: ‖g‖ = {norm_auto:.6e}  ({t_auto:.1f}s)")

        # Explicit IFT is intentionally limited to one step.  Multi-step IFT
        # requires chaining each step's tangent map; production autodiff and
        # FD below provide that rollout comparison at 2 and 5 steps.
        if horizon == 1:
            t0 = time.time()
            g_ift, stage_residual, stage_cond, stages, residual_history = _explicit_vehicle_ift_gradient(
                vehicle, x0, u, setup, n_setup_params, dt, x_target)
            g_ift.block_until_ready()
            ift_time = time.time() - t0
            ift_rel = float(jnp.linalg.norm(g_auto - g_ift) /
                            (jnp.linalg.norm(g_ift) + 1e-15))
            print(f"    Explicit stage IFT: ‖g‖ = {float(jnp.linalg.norm(g_ift)):.6e}"
                  f"  ({ift_time:.1f}s)")
            print(f"      stage residual = {float(stage_residual):.3e}; "
                  f"cond(F_z) = {float(stage_cond):.3e}; "
                  f"production-vs-IFT rel err = {ift_rel:.3e}")
            x_production = vehicle.simulate_step(x0, u, setup, dt=dt, n_substeps=1)
            x_stage = _vehicle_stage_output(x0, stages, dt)
            state_error = x_production - x_stage
            print("      production-stage state error "
                  f"(q/v/thermal/slip/damper/elastokin) = "
                  + "/".join(f"{float(jnp.max(jnp.abs(state_error[a:b]))):.2e}"
                             for a, b in ((0, 14), (14, 28), (28, 56), (56, 72),
                                          (72, 84), (84, 108))))
            print("      Picard residual history (iter: ||F||∞): " + ", ".join(
                f"{i + 1}:{float(residual_history[i, 1]):.2e}"
                for i in (0, 1, 3, 7, 15, 23, 31)))
            print("      FD branch diagnostics (ε=1e-4; ||F||∞, stage distance):")
            for i in range(n_setup_params):
                e = jnp.zeros_like(s_test).at[i].set(1e-4)
                plus = _stage_solution_diagnostics(vehicle, x0, u, setup.at[:n_setup_params].set(s_test + e), dt)
                minus = _stage_solution_diagnostics(vehicle, x0, u, setup.at[:n_setup_params].set(s_test - e), dt)
                print(f"        s[{i}]: + {plus['residual_inf']:.2e}, "
                      f"- {minus['residual_inf']:.2e}, "
                      f"||z+−z-||={float(jnp.linalg.norm(plus['stages'] - minus['stages'])):.2e}")
        else:
            g_ift = None
            stage_residual = None
            stage_cond = None
            ift_rel = None

        # ── Per-component comparison (best FD vs autodiff) ──
        best_eps = eps_values[len(eps_values) // 2]  # mid-range
        g_fd_best = fd_grads[best_eps]
        cos_sim = float(jnp.dot(g_auto, g_fd_best) /
                        (jnp.linalg.norm(g_auto) * jnp.linalg.norm(g_fd_best) + 1e-15))
        rel_err = float(jnp.linalg.norm(g_auto - g_fd_best) /
                        (jnp.linalg.norm(g_fd_best) + 1e-15))

        print(f"    vs FD(ε={best_eps:.0e}):")
        print(f"      Cosine similarity: {cos_sim:.6f}")
        print(f"      Relative error:    {rel_err:.6e}")

        print(f"\n    Componentwise gradients (FD(ε={best_eps:.0e}), explicit IFT, unrolled/production AD):")
        for i in range(n_setup_params):
            if g_ift is None:
                print(f"      s[{i}]: FD={float(g_fd_best[i]):+.4e}  AD={float(g_auto[i]):+.4e}")
            else:
                print(f"      s[{i}]: FD={float(g_fd_best[i]):+.4e}  "
                      f"IFT={float(g_ift[i]):+.4e}  AD={float(g_auto[i]):+.4e}")

        all_results[horizon] = {
            "fd_norms": fd_norms,
            "auto_norm": norm_auto,
            "cosine": cos_sim,
            "rel_error": rel_err,
            "auto_time": t_auto,
            "ift_norm": None if g_ift is None else float(jnp.linalg.norm(g_ift)),
            "ift_rel_error": ift_rel,
            "stage_residual": None if stage_residual is None else float(stage_residual),
            "stage_condition": None if stage_cond is None else float(stage_cond),
        }

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Save results
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(p1_results, p2_results, out_dir=None):
    if out_dir is None:
        out_dir = _ROOT
    results_dir = out_dir / "results"
    figs_dir = out_dir / "figs"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    # ── CSV (Problem 1) ──
    csv1 = results_dir / "gradient_accuracy_toy.csv"
    with open(csv1, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta", "analytic", "autodiff", "explicit_ift"] +
                        [f"fd_eps_{e}" for e in p1_results["fd_eps"]])
        for i, theta in enumerate(p1_results["theta"]):
            row = [theta, p1_results["analytic"][i], p1_results["autodiff"][i],
                   p1_results["ift"][i]]
            for eps_list in p1_results["fd_eps"].values():
                row.append(eps_list[i])
            writer.writerow(row)
    print(f"\n  Saved: {csv1}")

    # ── CSV (Problem 2) ──
    csv2 = results_dir / "gradient_accuracy_vehicle.csv"
    with open(csv2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon", "eps", "fd_norm", "auto_norm", "ift_norm",
                         "production_ift_rel_error", "stage_residual", "stage_condition",
                         "cosine", "rel_error"])
        for horizon, data in p2_results.items():
            for eps, norm in data["fd_norms"].items():
                writer.writerow([horizon, eps, norm, data["auto_norm"], data["ift_norm"],
                                 data["ift_rel_error"], data["stage_residual"],
                                 data["stage_condition"], data["cosine"], data["rel_error"]])
    print(f"  Saved: {csv2}")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Problem 1: error vs theta
        thetas = p1_results["theta"]
        for label, key, color, marker in [
            ("Autodiff", "autodiff", "#e74c3c", "o"),
            ("Explicit IFT", "ift", "#3498db", "s"),
        ]:
            errors = [abs(p1_results[key][i] - p1_results["analytic"][i]) /
                      (abs(p1_results["analytic"][i]) + 1e-15) for i in range(len(thetas))]
            ax1.semilogy(thetas, errors, f"{marker}-", color=color, label=label)

        # FD at different eps
        for eps, grads in p1_results["fd_eps"].items():
            errors = [abs(grads[i] - p1_results["analytic"][i]) /
                      (abs(p1_results["analytic"][i]) + 1e-15) for i in range(len(thetas))]
            ax1.semilogy(thetas, errors, "x--", alpha=0.5, label=f"FD(ε={eps:.0e})")

        ax1.set_xlabel("θ")
        ax1.set_ylabel("Relative error vs analytic")
        ax1.set_title("Problem 1: Toy Implicit Root", fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Problem 2: FD plateau
        for horizon, data in p2_results.items():
            eps_list = sorted(data["fd_norms"].keys())
            norms = [data["fd_norms"][e] for e in eps_list]
            ax2.semilogx(eps_list, norms, "o-", label=f"FD ‖g‖ ({horizon}-step)")
            ax2.axhline(data["auto_norm"], linestyle="--", alpha=0.5,
                        label=f"Autodiff ‖g‖ ({horizon}-step)")

        ax2.set_xlabel("Perturbation ε")
        ax2.set_ylabel("‖gradient‖")
        ax2.set_title("Problem 2: Vehicle FD Plateau", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()

        fig.suptitle("Project-GP · Gradient Validation (Phase 1.5)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figs_dir / f"gradient_accuracy.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figs_dir / 'gradient_accuracy.png'}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping figure")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  PROJECT-GP  ·  Gradient Validation (Phase 1.5)")
    print("=" * 72)

    p1 = run_problem_1()
    p2 = run_problem_2()
    save_results(p1, p2)

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Problem 1 check
    max_err_auto = max(abs(p1["autodiff"][i] - p1["analytic"][i]) /
                       (abs(p1["analytic"][i]) + 1e-15) for i in range(len(p1["theta"])))
    max_err_ift = max(abs(p1["ift"][i] - p1["analytic"][i]) /
                      (abs(p1["analytic"][i]) + 1e-15) for i in range(len(p1["theta"])))
    print(f"  Problem 1 (toy):")
    print(f"    Autodiff max rel err:     {max_err_auto:.2e}"
          f"  {'PASS' if max_err_auto < 1e-6 else 'INVESTIGATE'}")
    print(f"    Explicit IFT max rel err: {max_err_ift:.2e}"
          f"  {'PASS' if max_err_ift < 1e-6 else 'INVESTIGATE'}")

    print(f"  Problem 2 (vehicle):")
    for horizon, data in p2.items():
        print(f"    {horizon}-step: cos={data['cosine']:.4f}  "
              f"rel_err={data['rel_error']:.4e}"
              f"  {'OK' if data['cosine'] > 0.9 else 'INVESTIGATE'}")
    print("=" * 72)
