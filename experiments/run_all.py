#!/usr/bin/env python3
# experiments/run_all.py
# Project-GP — Master Experiment Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
"""
Runs ALL Project-GP experiments for full reproducibility.

Usage:
    python experiments/run_all.py [--skip-benchmark] [--seed 42]

This script is the single entry point for reproducing all research results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main(skip_benchmark: bool = False, seed: int = 42):
    start = time.time()

    print("╔" + "═" * 70 + "╗")
    print("║   PROJECT-GP  ·  Full Reproducibility Suite                       ║")
    print("╚" + "═" * 70 + "╝")

    # Ensure output directories
    (_ROOT / "results").mkdir(exist_ok=True)
    (_ROOT / "figs").mkdir(exist_ok=True)

    results = {}

    # ── 1. Passivity Verification ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  [1/5]  Passivity Verification (P1–P6)")
    print("=" * 72)
    try:
        from physics.passivity_verification import run_verification
        results["passivity"] = run_verification()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["passivity"] = False

    # ── 2. GLRK-4 Convergence ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  [2/5]  GLRK-4 Convergence Verification")
    print("=" * 72)
    try:
        from experiments.glrk_convergence import run_convergence_study, save_results
        conv_results = run_convergence_study()
        save_results(conv_results)
        results["glrk_convergence"] = conv_results
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["glrk_convergence"] = None

    # ── 3. Gradient Accuracy ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  [3/5]  Gradient Accuracy (IFT vs FD)")
    print("=" * 72)
    try:
        from experiments.gradient_accuracy import run_gradient_comparison
        from experiments.gradient_accuracy import save_results as save_grad
        grad_results = run_gradient_comparison(seed=seed)
        save_grad(grad_results)
        results["gradient_accuracy"] = grad_results
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["gradient_accuracy"] = None

    # ── 4. Energy/Passivity Laboratory ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("  [4/5]  Energy / Passivity Laboratory")
    print("=" * 72)
    try:
        from experiments.energy_lab import run_energy_lab
        from experiments.energy_lab import save_results as save_energy
        energy_results = run_energy_lab(seed=seed)
        save_energy(energy_results)
        results["energy_lab"] = energy_results
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["energy_lab"] = None

    # ── 5. Stress Tests ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  [5/5]  Numerical Stress Tests")
    print("=" * 72)
    try:
        from experiments.stress_test import run_stress_tests
        from experiments.stress_test import save_results as save_stress
        stress_results = run_stress_tests(seed=seed)
        save_stress(stress_results)
        results["stress_tests"] = stress_results
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results["stress_tests"] = None

    # ── Optional: Full Benchmark ─────────────────────────────────────────────
    if not skip_benchmark:
        print("\n" + "=" * 72)
        print("  [BONUS]  Full Benchmark Suite (5 models)")
        print("=" * 72)
        try:
            from benchmarks.run_benchmark import run_benchmark
            bench_results = run_benchmark(seed=seed)
            results["benchmark"] = bench_results
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results["benchmark"] = None

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║   REPRODUCIBILITY SUITE COMPLETE                                 ║")
    print("╚" + "═" * 70 + "╝")
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n  Output files:")
    for p in sorted((_ROOT / "results").glob("*.csv")):
        print(f"    {p.relative_to(_ROOT)}")
    print(f"\n  Figures:")
    for p in sorted((_ROOT / "figs").glob("*")):
        print(f"    {p.relative_to(_ROOT)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project-GP Full Reproducibility")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip the full model benchmark (saves time)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(skip_benchmark=args.skip_benchmark, seed=args.seed)
