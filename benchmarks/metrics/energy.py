# benchmarks/metrics/energy.py
# Project-GP — Energy Conservation and Passivity Metrics
# ═══════════════════════════════════════════════════════════════════════════════
"""
Metrics for evaluating energy-level properties:
- Energy drift over trajectories
- Energy balance error (ΔH = W_ext - W_diss)
- Passivity violation detection
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def energy_drift(H_trajectory: jnp.ndarray, dt: float = 0.005) -> dict:
    """
    Measure energy drift over a trajectory.

    Args:
        H_trajectory: (T,) Hamiltonian values at each timestep.
        dt: timestep size.

    Returns:
        dict with drift_rate, total_drift, relative_drift
    """
    H = H_trajectory
    T = H.shape[0]
    duration = T * dt

    total_drift = float(H[-1] - H[0])
    drift_rate = total_drift / max(duration, 1e-8)
    H_mean = float(jnp.mean(jnp.abs(H)))
    relative_drift = abs(total_drift) / max(H_mean, 1e-8)

    return {
        "total_drift": total_drift,
        "drift_rate": drift_rate,
        "relative_drift": relative_drift,
        "H_min": float(jnp.min(H)),
        "H_max": float(jnp.max(H)),
        "H_mean": H_mean,
        "duration_s": duration,
    }


def energy_balance_error(
    H_trajectory: jnp.ndarray,
    W_ext_cumulative: jnp.ndarray,
    W_diss_cumulative: jnp.ndarray,
) -> dict:
    """
    Verify the energy balance: ΔH = W_ext - W_diss.

    Args:
        H_trajectory: (T,) Hamiltonian values
        W_ext_cumulative: (T,) cumulative external work ∫ yᵀu dt
        W_diss_cumulative: (T,) cumulative dissipated energy ∫ ∇Hᵀ R ∇H dt

    Returns:
        dict with balance_error, relative_error at each timestep
    """
    delta_H = H_trajectory - H_trajectory[0]
    expected_delta_H = W_ext_cumulative - W_diss_cumulative

    balance_error = delta_H - expected_delta_H
    abs_max_error = float(jnp.max(jnp.abs(balance_error)))
    energy_scale = float(jnp.max(jnp.abs(delta_H)) + 1e-8)

    return {
        "max_absolute_error": abs_max_error,
        "relative_error": abs_max_error / energy_scale,
        "balance_error_timeseries": balance_error,
    }


def passivity_violation_rate(
    H_dot: jnp.ndarray,
    port_power: jnp.ndarray,
) -> dict:
    """
    Check passivity: dH/dt ≤ yᵀu (port power).

    Args:
        H_dot: (T,) time derivative of Hamiltonian
        port_power: (T,) yᵀu port input power

    Returns:
        dict with violation_rate, max_violation, mean_violation
    """
    violation = H_dot - port_power  # should be ≤ 0 for passive system
    is_violation = violation > 1e-6

    return {
        "violation_rate": float(jnp.mean(is_violation)),
        "max_violation": float(jnp.max(violation)),
        "mean_violation": float(jnp.mean(jnp.where(is_violation, violation, 0.0))),
        "n_violations": int(jnp.sum(is_violation)),
        "n_total": int(violation.shape[0]),
    }
