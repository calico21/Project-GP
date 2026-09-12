#!/usr/bin/env python3
# experiments/energy_lab.py
# Project-GP — Energy Conservation Proof (Phase 1.5 · A3)
# ═══════════════════════════════════════════════════════════════════════════════
"""
Two-part energy verification:

  Part 1  Port-Hamiltonian toy benchmark (controlled J, R, G)
          → proves ΔH = W_ext − W_diss analytically

  Part 2  Vehicle energy diagnostic (observational)
          → reports energy balance for the full vehicle model

Uses the same GLRK-4 Butcher tableau as the vehicle integrator.

Outputs:
    results/energy_lab.csv
    figs/energy_lab.png
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
# §0  GLRK-4 integrator (reused from glrk_convergence.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _glrk4_step_forced(system, x, u1, u2, dt, n_newton=10):
    """GLRK-4 step and its collocation states for a two-stage forced PH ODE."""
    sqrt3 = jnp.sqrt(3.0)
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    b1, b2 = 0.5, 0.5

    k1 = system.rhs(x, u1)
    k2 = k1

    def iterate(carry, _):
        k1_, k2_ = carry
        x1 = x + dt * (a11 * k1_ + a12 * k2_)
        x2 = x + dt * (a21 * k1_ + a22 * k2_)
        return (system.rhs(x1, u1), system.rhs(x2, u2)), (x1, x2)

    (k1, k2), stages = jax.lax.scan(iterate, (k1, k2), None, length=n_newton)
    # The final scan output is the collocation state associated with the
    # converged slopes.  Reconstructing it avoids an endpoint/trapezoid mix.
    x1 = x + dt * (a11 * k1 + a12 * k2)
    x2 = x + dt * (a21 * k1 + a22 * k2)
    return x + dt * (b1 * k1 + b2 * k2), x1, x2


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Part 1 — Port-Hamiltonian Toy Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ẋ = (J − R) ∇H + G·u     (port-Hamiltonian ODE)
#   y = Gᵀ ∇H                 (output)
#   H(x) = ½ xᵀ Q x           (quadratic Hamiltonian)
#
#   Energy identity:  dH/dt = yᵀu − xᵀQ R Q x
#                            = W_supply − W_diss
#

class PHSystem:
    """Simple linear port-Hamiltonian system for energy verification."""

    def __init__(self, Q, J, R, G):
        """
        Args:
            Q: (n,n) symmetric positive definite — Hamiltonian quadratic form
            J: (n,n) skew-symmetric — interconnection
            R: (n,n) symmetric PSD — dissipation
            G: (n,m) — input/output port matrix
        """
        self.Q = jnp.asarray(Q, dtype=jnp.float64)
        self.J = jnp.asarray(J, dtype=jnp.float64)
        self.R = jnp.asarray(R, dtype=jnp.float64)
        self.G = jnp.asarray(G, dtype=jnp.float64)
        self.n = self.Q.shape[0]
        self.m = self.G.shape[1]

    def H(self, x):
        """Hamiltonian: H(x) = ½ xᵀ Q x."""
        return 0.5 * x @ self.Q @ x

    def grad_H(self, x):
        """∇H = Q x."""
        return self.Q @ x

    def output(self, x):
        """y = Gᵀ ∇H."""
        return self.G.T @ self.grad_H(x)

    def rhs(self, x, u):
        """ẋ = (J − R) ∇H + G u."""
        return (self.J - self.R) @ self.grad_H(x) + self.G @ u

    def dissipation_rate(self, x):
        """Instantaneous dissipation: (∇H)ᵀ R (∇H) ≥ 0."""
        gH = self.grad_H(x)
        return gH @ self.R @ gH

    def supply_rate(self, x, u):
        """Instantaneous supply: yᵀ u."""
        return self.output(x) @ u


def _run_ph_scenario(system: PHSystem, x0, u_fn, dt, n_steps, name, expected):
    """Run one PH scenario and compute energy balance."""
    print(f"\n    Scenario: {name}")
    print(f"    Expected: {expected}")

    H_traj = [float(system.H(x0))]
    W_supply_acc = 0.0
    W_diss_acc = 0.0

    x = x0
    for step in range(n_steps):
        # Use precisely the Gauss collocation nodes for both the integration
        # and power quadrature.  Endpoint trapezoidal quadrature is only
        # second order and was the source of the previous balance residual.
        t = step * dt
        c1, c2 = 0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0
        u1, u2 = u_fn(t + c1 * dt), u_fn(t + c2 * dt)
        x_next, x1, x2 = _glrk4_step_forced(system, x, u1, u2, dt)
        W_supply_acc += 0.5 * dt * float(
            system.supply_rate(x1, u1) + system.supply_rate(x2, u2))
        W_diss_acc += 0.5 * dt * float(
            system.dissipation_rate(x1) + system.dissipation_rate(x2))

        x = x_next
        H_traj.append(float(system.H(x)))

    H0 = H_traj[0]
    H_final = H_traj[-1]
    delta_H = H_final - H0
    balance_residual = delta_H - W_supply_acc + W_diss_acc

    print(f"    H(0) = {H0:.10e}")
    print(f"    H(T) = {H_final:.10e}")
    print(f"    ΔH   = {delta_H:+.10e}")
    print(f"    W_supply = {W_supply_acc:+.10e}")
    print(f"    W_diss   = {W_diss_acc:+.10e}")
    print(f"    Balance residual |ΔH − W_supply + W_diss| = {abs(balance_residual):.6e}")

    return {
        "name": name,
        "expected": expected,
        "H_trajectory": np.array(H_traj),
        "H0": H0,
        "H_final": H_final,
        "delta_H": delta_H,
        "W_supply": W_supply_acc,
        "W_diss": W_diss_acc,
        "balance_residual": balance_residual,
        "dt": dt,
    }


def run_part_1(T_final: float = 5.0, dt: float = 0.01) -> list[dict]:
    """Part 1: Port-Hamiltonian toy benchmark with 4 scenarios."""
    print("\n" + "─" * 60)
    print("  PART 1 — Port-Hamiltonian Toy Benchmark")
    print("─" * 60)

    n_steps = int(round(T_final / dt))

    # System: 4D port-Hamiltonian
    # H(x) = ½ xᵀ Q x with Q = diag(2, 3, 1, 4)
    Q = jnp.diag(jnp.array([2.0, 3.0, 1.0, 4.0], dtype=jnp.float64))

    # Skew-symmetric J (interconnection)
    J = jnp.array([
        [ 0.0,  1.0, -0.5,  0.0],
        [-1.0,  0.0,  0.0,  0.3],
        [ 0.5,  0.0,  0.0, -1.0],
        [ 0.0, -0.3,  1.0,  0.0],
    ], dtype=jnp.float64)

    # Symmetric PSD dissipation
    R_pos = jnp.diag(jnp.array([0.5, 0.3, 0.0, 0.8], dtype=jnp.float64))

    # Input matrix
    G = jnp.array([
        [1.0, 0.0],
        [0.0, 0.5],
        [0.0, 0.0],
        [0.0, 1.0],
    ], dtype=jnp.float64)

    x0 = jnp.array([1.0, 0.5, -0.3, 0.8], dtype=jnp.float64)

    results = []

    # ── Scenario 1: Conservative (R=0, u=0) → ΔH = 0 ──
    sys_cons = PHSystem(Q, J, jnp.zeros_like(R_pos), G)
    u_zero = lambda t: jnp.zeros(2, dtype=jnp.float64)
    results.append(_run_ph_scenario(
        sys_cons, x0, u_zero, dt, n_steps,
        "Conservative (R=0, u=0)", "ΔH = 0"
    ))

    # ── Scenario 2: Dissipative (R≥0, u=0) → ΔH ≤ 0 ──
    sys_diss = PHSystem(Q, J, R_pos, G)
    results.append(_run_ph_scenario(
        sys_diss, x0, u_zero, dt, n_steps,
        "Dissipative (R≥0, u=0)", "ΔH ≤ 0"
    ))

    # ── Scenario 3: Forced, no dissipation (R=0, u≠0) → ΔH = W_supply ──
    sys_forced = PHSystem(Q, J, jnp.zeros_like(R_pos), G)
    u_sine = lambda t: jnp.array([
        0.3 * jnp.sin(2.0 * jnp.pi * t / 2.0),
        0.2 * jnp.cos(2.0 * jnp.pi * t / 3.0),
    ], dtype=jnp.float64)
    results.append(_run_ph_scenario(
        sys_forced, x0, u_sine, dt, n_steps,
        "Forced (R=0, u≠0)", "ΔH = W_supply"
    ))

    # ── Scenario 4: Forced + dissipative → ΔH = W_supply − W_diss ──
    sys_full = PHSystem(Q, J, R_pos, G)
    results.append(_run_ph_scenario(
        sys_full, x0, u_sine, dt, n_steps,
        "Forced + dissipative", "ΔH = W_supply − W_diss"
    ))

    # A balance residual must converge under refinement; a single tolerance is
    # not evidence for the discrete identity.
    for factor in (2, 4):
        refined_dt = dt / factor
        refined_steps = int(round(T_final / refined_dt))
        for base, system, u_fn in zip(
            results, (sys_cons, sys_diss, sys_forced, sys_full),
            (u_zero, u_zero, u_sine, u_sine),
        ):
            refined = _run_ph_scenario(system, x0, u_fn, refined_dt, refined_steps,
                                       f"{base['name']} [dt/{factor}]", base["expected"])
            base.setdefault("balance_refinement", []).append(
                (refined_dt, refined["balance_residual"]))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Part 2 — Vehicle Energy Diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_2(
    duration: float = 2.0,
    dt: float = 0.005,
    vx0: float = 20.0,
) -> list[dict]:
    """Part 2: vehicle energy diagnostic — observational, not proof."""
    print("\n" + "─" * 60)
    print("  PART 2 — Vehicle Energy Diagnostic (observational)")
    print("─" * 60)
    print("  NOTE: This is a diagnostic, NOT a proof of passivity.")
    print("  It tracks physical energy, not just H_net.")

    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from config.tire_coeffs import tire_coeffs as TC

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)

    n_steps = int(duration / dt)
    setup = jnp.ones(28, dtype=jnp.float64) * 0.5

    scenarios = [
        ("coasting", "u=0, observe energy evolution",
         lambda t, x: jnp.zeros(6, dtype=jnp.float64)),
        ("constant_drive", "constant torque, observe energy growth",
         lambda t, x: jnp.array([0.0, 10.0, 10.0, 10.0, 10.0, 0.0], dtype=jnp.float64)),
        ("steer_oscillation", "sinusoidal steering, observe lateral dynamics",
         lambda t, x: jnp.array([
             0.05 * jnp.sin(2.0 * jnp.pi * t / 2.0),
             5.0, 5.0, 5.0, 5.0, 0.0
         ], dtype=jnp.float64)),
    ]

    results = []

    for name, desc, u_fn in scenarios:
        print(f"\n    Scenario: {name} — {desc}")
        x0 = vehicle.make_initial_state(vx0=vx0).astype(jnp.float64)
        x = x0

        KE_traj = []
        PE_traj = []

        for step in range(n_steps):
            t = step * dt
            v = x[14:28]
            q = x[0:14]

            # Physical kinetic energy: ½ M_diag v²
            KE = float(0.5 * jnp.sum(vehicle.M_diag * v**2))

            # Physical potential (spring): ½ k (z - z_eq)²
            z_eq = jnp.array([0.0128, 0.0128, 0.0142, 0.0142], dtype=jnp.float64)
            z_susp = q[6:10]
            k_approx = jnp.array([35000.0, 35000.0, 38000.0, 38000.0], dtype=jnp.float64)
            PE = float(0.5 * jnp.sum(k_approx * (z_susp - z_eq)**2))

            KE_traj.append(KE)
            PE_traj.append(PE)

            u = u_fn(t, x)
            x = vehicle.simulate_step(x, u, setup, dt=dt, n_substeps=1)

        # Final state energy
        v = x[14:28]
        q = x[0:14]
        KE_traj.append(float(0.5 * jnp.sum(vehicle.M_diag * v**2)))
        z_susp = q[6:10]
        PE_traj.append(float(0.5 * jnp.sum(k_approx * (z_susp - z_eq)**2)))

        KE_arr = np.array(KE_traj)
        PE_arr = np.array(PE_traj)
        E_total = KE_arr + PE_arr

        drift = (E_total[-1] - E_total[0]) / (duration + 1e-15)

        print(f"      KE: {KE_traj[0]:.1f} → {KE_traj[-1]:.1f} J")
        print(f"      PE: {PE_traj[0]:.4f} → {PE_traj[-1]:.4f} J")
        print(f"      E_total: {E_total[0]:.1f} → {E_total[-1]:.1f} J")
        print(f"      Energy drift rate: {drift:.2f} J/s")

        results.append({
            "name": name,
            "description": desc,
            "KE": KE_arr,
            "PE": PE_arr,
            "E_total": E_total,
            "drift_rate": drift,
            "dt": dt,
        })

    return results


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

    # ── CSV ──
    csv_path = results_dir / "energy_lab.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "scenario", "H0", "H_final", "delta_H",
                         "W_supply", "W_diss", "balance_residual", "expected"])
        for r in p1_results:
            writer.writerow(["PH_toy", r["name"], r["H0"], r["H_final"],
                             r["delta_H"], r["W_supply"], r["W_diss"],
                             r["balance_residual"], r["expected"]])
        for r in p2_results:
            writer.writerow(["vehicle", r["name"], r["E_total"][0], r["E_total"][-1],
                             r["E_total"][-1] - r["E_total"][0], "", "",
                             r["drift_rate"], r["description"]])
    print(f"\n  Saved: {csv_path}")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        # Part 1: PH toy scenarios
        colors_p1 = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
        for i, (r, color) in enumerate(zip(p1_results, colors_p1)):
            ax = axes[0, i]
            H = r["H_trajectory"]
            t = np.arange(len(H)) * r["dt"]
            ax.plot(t, H, color=color, linewidth=1.5)
            ax.axhline(y=H[0], color="gray", linestyle="--", alpha=0.4, label=f"H(0)={H[0]:.2f}")
            ax.set_ylabel("H [J]")
            ax.set_xlabel("Time [s]")
            short_name = r["name"].split("(")[0].strip()
            ax.set_title(short_name, fontsize=10, fontweight="bold")
            res_str = f"|res|={abs(r['balance_residual']):.1e}"
            ax.legend([f"H(t)", f"H(0)", res_str], fontsize=8)
            ax.grid(True, alpha=0.3)

        # Part 2: Vehicle diagnostics
        colors_p2 = ["#9b59b6", "#1abc9c", "#f39c12"]
        for i, (r, color) in enumerate(zip(p2_results, colors_p2)):
            ax = axes[1, i]
            t = np.arange(len(r["E_total"])) * r["dt"]
            ax.plot(t, r["KE"] / 1e3, color=color, linewidth=1.5, label="KE")
            ax.plot(t, r["PE"], color=color, linewidth=1, linestyle="--", label="PE")
            ax.set_ylabel("Energy [kJ / J]")
            ax.set_xlabel("Time [s]")
            ax.set_title(f"Vehicle: {r['name']}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        axes[1, 3].set_visible(False)

        fig.suptitle("Project-GP · Energy/Passivity Laboratory (Phase 1.5)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figs_dir / f"energy_lab.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figs_dir / 'energy_lab.png'}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping figure")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  PROJECT-GP  ·  Energy / Passivity Laboratory (Phase 1.5)")
    print("=" * 72)

    p1 = run_part_1()
    p2 = run_part_2()
    save_results(p1, p2)

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  SUMMARY — Part 1 (PH Toy)")
    print("=" * 72)

    all_pass = True
    for r in p1:
        res = abs(r["balance_residual"])
        if "Conservative" in r["name"]:
            ok = abs(r["delta_H"]) < 1e-8
            print(f"  {r['name']}: |ΔH| = {abs(r['delta_H']):.2e}"
                  f"  {'PASS' if ok else 'FAIL'}")
        elif "Dissipative" in r["name"] and "Forced" not in r["name"]:
            ok = r["delta_H"] <= 1e-10
            print(f"  {r['name']}: ΔH = {r['delta_H']:.2e}"
                  f"  {'PASS (≤0)' if ok else 'FAIL'}")
        else:
            ok = res < 1e-6
            print(f"  {r['name']}: |residual| = {res:.2e}"
                  f"  {'PASS' if ok else 'INVESTIGATE'}")
        if not ok:
            all_pass = False

    print(f"\n  Part 1 overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    print("\n  SUMMARY — Part 2 (Vehicle Diagnostic)")
    print("  " + "─" * 50)
    for r in p2:
        print(f"  {r['name']}: E drift = {r['drift_rate']:+.1f} J/s")
    print("  (Part 2 is diagnostic — no pass/fail criteria)")
    print("=" * 72)
