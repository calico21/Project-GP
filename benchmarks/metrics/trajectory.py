# benchmarks/metrics/trajectory.py
# Project-GP — Trajectory Accuracy Metrics
# ═══════════════════════════════════════════════════════════════════════════════
"""
Comprehensive trajectory accuracy metrics for model comparison.
All functions accept (y_true, y_pred) arrays of shape (N, D) or (N,).
"""

from __future__ import annotations

import jax.numpy as jnp


def mae(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Mean Absolute Error."""
    return float(jnp.mean(jnp.abs(y_true - y_pred)))


def rmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(jnp.sqrt(jnp.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Normalised RMSE (by range of true values)."""
    r = jnp.max(y_true) - jnp.min(y_true)
    r = jnp.maximum(r, 1e-8)
    return float(jnp.sqrt(jnp.mean((y_true - y_pred) ** 2)) / r)


def r_squared(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return float(1.0 - ss_res / jnp.maximum(ss_tot, 1e-8))


def pearson_rho(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    """Pearson correlation coefficient."""
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    mean_t = jnp.mean(y_true_flat)
    mean_p = jnp.mean(y_pred_flat)
    cov = jnp.mean((y_true_flat - mean_t) * (y_pred_flat - mean_p))
    std_t = jnp.std(y_true_flat)
    std_p = jnp.std(y_pred_flat)
    return float(cov / jnp.maximum(std_t * std_p, 1e-8))


def per_state_errors(y_true: jnp.ndarray, y_pred: jnp.ndarray,
                     state_names: list[str] | None = None) -> dict:
    """
    Compute MAE, RMSE, R² per state dimension.

    Args:
        y_true: (N, D) ground truth states
        y_pred: (N, D) predicted states
        state_names: optional list of D state names

    Returns:
        dict mapping state_name → {mae, rmse, r2}
    """
    n, d = y_true.shape
    if state_names is None:
        state_names = [f"x{i}" for i in range(d)]

    results = {}
    for i, name in enumerate(state_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        results[name] = {
            "mae": mae(yt, yp),
            "rmse": rmse(yt, yp),
            "r2": r_squared(yt, yp),
            "pearson": pearson_rho(yt, yp),
        }
    return results


def multi_horizon_evaluation(
    model,
    params: dict,
    x0_batch: jnp.ndarray,
    controls_batch: jnp.ndarray,
    x_true_batch: jnp.ndarray,
    setup_batch: jnp.ndarray,
    dt: float = 0.005,
    horizons: dict = None,
) -> dict:
    """
    Evaluate model accuracy at multiple prediction horizons.

    Args:
        horizons: dict mapping name → duration in seconds.
            Default: {"one_step": dt, "short": 0.5, "long": 5.0, "very_long": 20.0}

    Returns:
        dict mapping horizon_name → {rmse, nrmse, r2, pearson}
    """
    import jax

    if horizons is None:
        horizons = {
            "one_step": dt,
            "short_0.5s": 0.5,
            "long_5s": 5.0,
            "very_long_20s": 20.0,
        }

    results = {}
    for hname, duration in horizons.items():
        n_steps = max(1, int(duration / dt))
        # Truncate controls to horizon length
        n_avail = controls_batch.shape[1]
        n_eval = min(n_steps, n_avail)

        def rollout_single(x0, controls, setup):
            return model.predict_trajectory(params, x0, controls[:n_eval], setup)

        # Evaluate on batch
        traj_pred = jax.vmap(rollout_single)(
            x0_batch, controls_batch, setup_batch)
        traj_true = x_true_batch[:, :n_eval, :]

        results[hname] = {
            "rmse": rmse(traj_true, traj_pred),
            "nrmse": nrmse(traj_true, traj_pred),
            "r2": r_squared(traj_true, traj_pred),
            "pearson": pearson_rho(traj_true, traj_pred),
            "n_steps": n_eval,
            "duration_s": n_eval * dt,
        }
    return results
