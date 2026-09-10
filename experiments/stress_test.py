#!/usr/bin/env python3
# experiments/stress_test.py
# Project-GP — Numerical Stress Test Suite
# ═══════════════════════════════════════════════════════════════════════════════
"""
Tests numerical robustness under 12 extreme scenarios:

Extreme velocities:
  1. Near-zero  vx = 0.5 m/s
  2. Very high  vx = 50 m/s
  3. Reverse    vx = -5 m/s

Extreme loads:
  4. Full lateral  ay ≈ 2g
  5. Combined 1.5g lateral + 1g longitudinal
  6. Extreme weight transfer

Extreme steering:
  7. Full lock  δ = ±0.15 rad
  8. Rapid step δ: 0 → 0.15 in 0.01s

Extreme timesteps:
  9.  dt = 0.001 (5× nominal)
  10. dt = 0.020 (4× nominal — near CFL boundary)
  11. dt = 0.050 (10× nominal — beyond CFL)

Extreme parameters:
  12. Corner-case setup (max stiffness + min damping)

Records: NaN, Inf, solver failure, energy explosion, gradient explosion/vanishing.

Outputs:
    results/robustness.csv
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class StressScenario:
    name: str
    description: str
    vx0: float = 20.0
    vy0: float = 0.0
    wz0: float = 0.0
    delta: float = 0.0
    torque: float = 10.0
    brake: float = 0.0
    dt: float = 0.005
    n_steps: int = 200
    setup_override: dict | None = None


STRESS_SCENARIOS = [
    # Extreme velocities
    StressScenario("near_zero_vx", "vx = 0.5 m/s", vx0=0.5),
    StressScenario("high_vx", "vx = 50 m/s", vx0=50.0),
    StressScenario("reverse", "vx = -5 m/s", vx0=-5.0),

    # Extreme loads
    StressScenario("full_lateral", "ay ≈ 2g", vx0=25.0, wz0=0.8,
                   delta=0.08),
    StressScenario("combined_load", "1.5g lat + 1g lon", vx0=25.0,
                   wz0=0.6, delta=0.05, torque=21.0, brake=1500.0),
    StressScenario("weight_transfer", "Full braking from 40 m/s", vx0=40.0,
                   brake=2000.0, torque=0.0),

    # Extreme steering
    StressScenario("full_lock", "δ = 0.15 rad (full lock)", delta=0.15),
    StressScenario("rapid_step", "Step steer 0→0.15 in 2 steps", delta=0.15,
                   n_steps=100),

    # Extreme timesteps
    StressScenario("fine_dt", "dt = 0.001s (5× nominal)", dt=0.001, n_steps=1000),
    StressScenario("coarse_dt", "dt = 0.020s (4× nominal)", dt=0.020, n_steps=50),
    StressScenario("extreme_dt", "dt = 0.050s (10× nominal)", dt=0.050, n_steps=20),

    # Extreme parameters
    StressScenario("corner_setup", "Max stiffness + min damping",
                   setup_override={"k_f": 60000., "k_r": 60000.,
                                   "c_lo_f": 2000., "c_lo_r": 2000.}),
]


def run_stress_tests(seed: int = 42) -> list[dict]:
    """
    Run all stress test scenarios.

    Returns:
        list of result dicts, one per scenario
    """
    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from config.tire_coeffs import tire_coeffs as TC

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)

    results = []
    setup_base = jnp.ones(28) * 0.5

    for scenario in STRESS_SCENARIOS:
        print(f"\n  ── {scenario.name}: {scenario.description} ──")

        x0 = vehicle.make_initial_state(vx0=scenario.vx0)
        x0 = x0.at[15].set(scenario.vy0)  # vy
        x0 = x0.at[19].set(scenario.wz0)  # wz

        u = jnp.array([scenario.delta, scenario.torque, scenario.torque,
                        scenario.torque, scenario.torque, scenario.brake])

        setup = setup_base.copy()
        if scenario.setup_override:
            param_indices = {"k_f": 0, "k_r": 1, "arb_f": 2, "arb_r": 3,
                             "c_lo_f": 4, "c_lo_r": 5}
            for name, val in scenario.setup_override.items():
                if name in param_indices:
                    setup = setup.at[param_indices[name]].set(val)

        # ── Run simulation ──
        t0 = time.time()
        x = x0
        trajectory = [x]
        solver_failures = 0
        energy_explosion = False

        try:
            for step in range(scenario.n_steps):
                x_next = vehicle.simulate_step(
                    x, u, setup, dt=scenario.dt, n_substeps=1)
                trajectory.append(x_next)

                # Check for divergence
                if jnp.any(jnp.isnan(x_next)) or jnp.any(jnp.isinf(x_next)):
                    solver_failures += 1
                    break

                # Check energy
                if jnp.any(jnp.abs(x_next[:28]) > 1e6):
                    energy_explosion = True
                    break

                x = x_next

        except Exception as e:
            solver_failures += 1
            print(f"    Exception: {e}")

        elapsed = time.time() - t0
        traj = jnp.stack(trajectory)

        # ── Analyse ──
        has_nan = bool(jnp.any(jnp.isnan(traj)))
        has_inf = bool(jnp.any(jnp.isinf(traj)))
        completed_steps = len(trajectory) - 1
        max_state = float(jnp.max(jnp.abs(traj[:, :28])))

        # Gradient check (can the model differentiate through?)
        gradient_ok = True
        try:
            def test_grad(s):
                x_ = x0
                for _ in range(min(5, scenario.n_steps)):
                    x_ = vehicle.simulate_step(x_, u, s, dt=scenario.dt,
                                                n_substeps=1)
                return jnp.sum(x_[:28] ** 2)
            g = jax.grad(test_grad)(setup)
            has_grad_nan = bool(jnp.any(jnp.isnan(g)))
            has_grad_inf = bool(jnp.any(jnp.isinf(g)))
            grad_norm = float(jnp.linalg.norm(g))
            gradient_ok = not has_grad_nan and not has_grad_inf
        except Exception:
            has_grad_nan = True
            has_grad_inf = False
            grad_norm = float("nan")
            gradient_ok = False

        result = {
            "scenario": scenario.name,
            "description": scenario.description,
            "completed_steps": completed_steps,
            "target_steps": scenario.n_steps,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "solver_failures": solver_failures,
            "energy_explosion": energy_explosion,
            "max_state_value": max_state,
            "gradient_ok": gradient_ok,
            "grad_norm": grad_norm,
            "has_grad_nan": has_grad_nan if 'has_grad_nan' in dir() else None,
            "wall_time_s": elapsed,
            "dt": scenario.dt,
            "pass": (not has_nan and not has_inf and not energy_explosion
                     and completed_steps == scenario.n_steps),
        }

        results.append(result)

        status = "✓ PASS" if result["pass"] else "✗ FAIL"
        print(f"    Steps: {completed_steps}/{scenario.n_steps}")
        print(f"    NaN: {has_nan}  Inf: {has_inf}  Explosion: {energy_explosion}")
        print(f"    Gradient: {'✓' if gradient_ok else '✗'} (‖g‖={grad_norm:.3e})")
        print(f"    {status}  ({elapsed:.2f}s)")

    return results


def save_results(results: list[dict], out_dir: Path = None):
    if out_dir is None:
        out_dir = _ROOT
    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "robustness.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["scenario", "description", "completed_steps", "target_steps",
                      "has_nan", "has_inf", "solver_failures", "energy_explosion",
                      "max_state_value", "gradient_ok", "grad_norm", "wall_time_s",
                      "dt", "pass"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    print("=" * 72)
    print("  PROJECT-GP  ·  Numerical Stress Test Suite")
    print("=" * 72)

    results = run_stress_tests()
    save_results(results)

    n_pass = sum(1 for r in results if r["pass"])
    n_total = len(results)

    print("\n" + "=" * 72)
    print(f"  RESULTS: {n_pass}/{n_total} scenarios passed")
    if n_pass < n_total:
        print("  Failed scenarios:")
        for r in results:
            if not r["pass"]:
                print(f"    ✗ {r['scenario']}: {r['description']}")
    print("=" * 72)
