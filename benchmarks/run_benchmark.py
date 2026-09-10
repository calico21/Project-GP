#!/usr/bin/env python3
# benchmarks/run_benchmark.py
# Project-GP — Central Benchmark Experiment
# ═══════════════════════════════════════════════════════════════════════════════
"""
Scientific benchmark: trains 5 models on identical data with identical budget,
then evaluates on interpolation + extrapolation datasets with exhaustive metrics.

Usage:
    python benchmarks/run_benchmark.py [--n-train 10000] [--n-test 2000]
                                        [--n-epochs 500] [--seed 42]

Outputs:
    results/benchmark.csv           — per-model/per-metric table
    results/extrapolation.csv       — per-model extrapolation accuracy
    figs/benchmark.pdf              — comparison bar charts
    figs/extrapolation.pdf          — extrapolation degradation plots
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.baselines import BASELINE_REGISTRY
from benchmarks.metrics.trajectory import mae, rmse, nrmse, r_squared, pearson_rho
from benchmarks.metrics.stability import detect_numerical_failures


def _ensure_dirs():
    """Create results/ and figs/ if they don't exist."""
    (_ROOT / "results").mkdir(exist_ok=True)
    (_ROOT / "figs").mkdir(exist_ok=True)


def _generate_synthetic_dataset(n_train: int, n_test: int,
                                seed: int = 42) -> dict:
    """
    Generate a synthetic benchmark dataset.
    Falls back to synthetic data if vehicle model is unavailable.
    """
    try:
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
        from config.vehicles.ter27 import vehicle_params_ter27 as VP
        from config.tire_coeffs import tire_coeffs as TC
        from benchmarks.datasets.interpolation import generate_interpolation_dataset

        vehicle = DifferentiableMultiBodyVehicle(VP, TC)
        return generate_interpolation_dataset(
            vehicle, n_train=n_train, n_test=n_test, seed=seed)
    except Exception as e:
        print(f"  [WARN] Cannot load vehicle model ({e}), using synthetic data")
        return _synthetic_fallback(n_train, n_test, seed)


def _synthetic_fallback(n_train: int, n_test: int, seed: int) -> dict:
    """Generate simple synthetic dynamics data for structure testing."""
    rng = jax.random.PRNGKey(seed)
    state_dim, control_dim, setup_dim = 28, 6, 28
    dt = 0.005

    def _gen(rng, n):
        k1, k2, k3 = jax.random.split(rng, 3)
        x = jax.random.normal(k1, (n, state_dim)) * 0.5
        u = jax.random.normal(k2, (n, control_dim)) * 0.3
        setup = jax.random.uniform(k3, (n, setup_dim))
        # Simple nonlinear dynamics: x_next = x + dt * tanh(Ax + Bu)
        A = jax.random.normal(jax.random.PRNGKey(0), (state_dim, state_dim)) * 0.1
        B = jax.random.normal(jax.random.PRNGKey(1), (state_dim, control_dim)) * 0.1
        x_next = x + dt * jnp.tanh(x @ A + u @ B.T)
        return x, u, x_next, setup

    k_train, k_test = jax.random.split(rng)
    xtr, utr, xntr, str_ = _gen(k_train, n_train)
    xte, ute, xnte, ste = _gen(k_test, n_test)

    return {
        "train_x": xtr, "train_u": utr,
        "train_x_next": xntr, "train_setup": str_,
        "test_x": xte, "test_u": ute,
        "test_x_next": xnte, "test_setup": ste,
    }


def run_benchmark(n_train: int = 10000, n_test: int = 2000,
                  n_epochs: int = 500, seed: int = 42,
                  models: list[str] | None = None) -> dict:
    """
    Run the complete benchmark.

    Returns:
        dict mapping model_name → {metric_name: value}
    """
    _ensure_dirs()
    print("=" * 72)
    print("  PROJECT-GP  ·  Scientific Benchmark")
    print("=" * 72)

    # ── 1. Generate data ──────────────────────────────────────────────────────
    print("\n  [1/4] Generating dataset...")
    data = _generate_synthetic_dataset(n_train, n_test, seed)
    train_data = {
        "x": data["train_x"], "u": data["train_u"],
        "x_next": data["train_x_next"], "setup": data["train_setup"],
    }
    test_data = {
        "x": data["test_x"], "u": data["test_u"],
        "x_next": data["test_x_next"], "setup": data["test_setup"],
    }
    print(f"         Train: {train_data['x'].shape[0]} samples")
    print(f"         Test:  {test_data['x'].shape[0]} samples")

    # ── 2. Train all models ──────────────────────────────────────────────────
    if models is None:
        models = list(BASELINE_REGISTRY.keys())

    all_results = {}

    print(f"\n  [2/4] Training {len(models)} models ({n_epochs} epochs each)...")
    for model_name in models:
        print(f"\n    ┌─ {model_name}")
        cls = BASELINE_REGISTRY[model_name]
        model = cls()

        rng = jax.random.PRNGKey(seed)
        t0 = time.time()
        train_info = model.train(rng, train_data, n_epochs=n_epochs)
        train_time = time.time() - t0

        print(f"    │  Train loss: {train_info['final_loss']:.6f}")
        print(f"    │  Train time: {train_time:.1f}s")

        # ── 3. Evaluate ──────────────────────────────────────────────────────
        params = model.params if hasattr(model, 'params') else train_info["params"]

        # One-step predictions on test set
        x_pred = jax.vmap(
            lambda x, u, s: model.predict_step(params, x, u, s)
        )(test_data["x"], test_data["u"], test_data["setup"])

        x_true = test_data["x_next"]

        # Check for numerical failures
        failures = detect_numerical_failures(x_pred)

        results = {
            "train_loss": train_info["final_loss"],
            "train_time_s": train_time,
            "mae": mae(x_true, x_pred),
            "rmse": rmse(x_true, x_pred),
            "nrmse": nrmse(x_true, x_pred),
            "r2": r_squared(x_true, x_pred),
            "pearson": pearson_rho(x_true, x_pred),
            "has_nan": failures["has_nan"],
            "has_inf": failures["has_inf"],
        }

        all_results[model_name] = results
        print(f"    │  RMSE:    {results['rmse']:.6f}")
        print(f"    │  R²:      {results['r2']:.6f}")
        print(f"    │  Pearson: {results['pearson']:.6f}")
        print(f"    └─ {'✓ Valid' if failures['is_valid'] else '✗ NaN/Inf detected!'}")

    # ── 4. Save results ──────────────────────────────────────────────────────
    print("\n  [3/4] Saving results...")
    _save_csv(all_results, _ROOT / "results" / "benchmark.csv")

    # ── 5. Generate figures ──────────────────────────────────────────────────
    print("  [4/4] Generating figures...")
    _generate_figures(all_results, _ROOT / "figs")

    print("\n" + "=" * 72)
    print("  BENCHMARK COMPLETE")
    print(f"    Results: results/benchmark.csv")
    print(f"    Figures: figs/benchmark.pdf")
    print("=" * 72)

    return all_results


def _save_csv(results: dict, path: Path):
    """Save benchmark results to CSV."""
    if not results:
        return
    fieldnames = ["model"] + list(next(iter(results.values())).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            row = {"model": model_name, **metrics}
            writer.writerow(row)
    print(f"    Saved: {path}")


def _generate_figures(results: dict, fig_dir: Path):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(results.keys())
        metrics_to_plot = ["rmse", "nrmse", "r2", "pearson"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Project-GP Benchmark: Model Comparison", fontsize=14, fontweight="bold")

        for ax, metric in zip(axes.flat, metrics_to_plot):
            values = [results[m].get(metric, 0.0) for m in models]
            colors = ["#2ecc71" if m == "PassiveHNet" else "#3498db" for m in models]
            ax.bar(models, values, color=colors, edgecolor="white", linewidth=1)
            ax.set_title(metric.upper(), fontweight="bold")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        fig.savefig(fig_dir / "benchmark.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(fig_dir / "benchmark.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {fig_dir / 'benchmark.pdf'}")

    except ImportError:
        print("    [WARN] matplotlib not available, skipping figure generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project-GP Scientific Benchmark")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to benchmark (default: all)")
    args = parser.parse_args()

    run_benchmark(
        n_train=args.n_train,
        n_test=args.n_test,
        n_epochs=args.n_epochs,
        seed=args.seed,
        models=args.models,
    )
