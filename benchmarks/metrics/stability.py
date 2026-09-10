# benchmarks/metrics/stability.py
# Project-GP — Stability and Robustness Metrics
# ═══════════════════════════════════════════════════════════════════════════════
"""
Metrics for evaluating model stability:
- Divergence time estimation
- Lyapunov exponent approximation
- Finite-time instability detection
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def estimate_divergence_time(
    traj_pred: jnp.ndarray,
    traj_true: jnp.ndarray,
    dt: float = 0.005,
    threshold: float = 0.5,
) -> dict:
    """
    Estimate when model predictions diverge from ground truth.

    Args:
        traj_pred: (T, D) predicted trajectory
        traj_true: (T, D) ground truth trajectory
        dt: timestep
        threshold: relative error threshold for divergence

    Returns:
        dict with divergence_time, divergence_step, trajectory_length
    """
    errors = jnp.sqrt(jnp.mean((traj_pred - traj_true) ** 2, axis=-1))
    true_scale = jnp.sqrt(jnp.mean(traj_true ** 2, axis=-1))
    rel_errors = errors / jnp.maximum(true_scale, 1e-8)

    # First timestep where relative error exceeds threshold
    diverged = rel_errors > threshold
    # Use argmax to find first True (returns 0 if none True)
    first_diverge = jnp.argmax(diverged)
    has_diverged = jnp.any(diverged)

    T = traj_pred.shape[0]
    div_step = int(jnp.where(has_diverged, first_diverge, T))
    div_time = div_step * dt

    return {
        "divergence_time_s": div_time,
        "divergence_step": div_step,
        "trajectory_length_s": T * dt,
        "has_diverged": bool(has_diverged),
        "final_relative_error": float(rel_errors[-1]),
        "max_relative_error": float(jnp.max(rel_errors)),
    }


def approximate_lyapunov_exponent(
    traj1: jnp.ndarray,
    traj2: jnp.ndarray,
    dt: float = 0.005,
    d0: float = None,
) -> dict:
    """
    Approximate the maximum Lyapunov exponent from two nearby trajectories.

    λ ≈ (1/T) * ln(‖δx(T)‖ / ‖δx(0)‖)

    Args:
        traj1, traj2: (T, D) two trajectories from nearby initial conditions
        dt: timestep
        d0: initial separation (computed from data if None)

    Returns:
        dict with lyapunov_exponent, growth_rate, is_stable
    """
    delta = traj1 - traj2
    separations = jnp.linalg.norm(delta, axis=-1)

    if d0 is None:
        d0 = float(separations[0])
    d0 = max(d0, 1e-12)

    T = delta.shape[0] * dt
    dT = float(separations[-1])

    lyap = jnp.log(max(dT, 1e-12) / d0) / max(T, 1e-8)

    return {
        "lyapunov_exponent": float(lyap),
        "initial_separation": d0,
        "final_separation": dT,
        "duration_s": T,
        "is_stable": float(lyap) < 0.0,
    }


def detect_numerical_failures(trajectory: jnp.ndarray) -> dict:
    """
    Scan a trajectory for numerical pathologies.

    Returns:
        dict with has_nan, has_inf, nan_count, inf_count, first_nan_step
    """
    has_nan = bool(jnp.any(jnp.isnan(trajectory)))
    has_inf = bool(jnp.any(jnp.isinf(trajectory)))

    nan_per_step = jnp.any(jnp.isnan(trajectory), axis=-1)
    inf_per_step = jnp.any(jnp.isinf(trajectory), axis=-1)

    first_nan = int(jnp.argmax(nan_per_step)) if has_nan else -1
    first_inf = int(jnp.argmax(inf_per_step)) if has_inf else -1

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_steps": int(jnp.sum(nan_per_step)),
        "inf_steps": int(jnp.sum(inf_per_step)),
        "first_nan_step": first_nan,
        "first_inf_step": first_inf,
        "total_steps": int(trajectory.shape[0]),
        "is_valid": not has_nan and not has_inf,
    }
