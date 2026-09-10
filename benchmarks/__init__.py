# benchmarks/__init__.py
# Project-GP — Benchmark Framework
# ═══════════════════════════════════════════════════════════════════════════════
"""
Scientific benchmark framework for comparing PassiveHNet against baselines.

Usage:
    python benchmarks/run_benchmark.py
"""

from benchmarks.metrics.trajectory import (
    mae, rmse, nrmse, r_squared, pearson_rho, per_state_errors,
)
from benchmarks.metrics.energy import energy_drift, energy_balance_error
from benchmarks.metrics.gradients import gradient_relative_error
from benchmarks.metrics.stability import estimate_divergence_time

__all__ = [
    "mae", "rmse", "nrmse", "r_squared", "pearson_rho", "per_state_errors",
    "energy_drift", "energy_balance_error",
    "gradient_relative_error",
    "estimate_divergence_time",
]
