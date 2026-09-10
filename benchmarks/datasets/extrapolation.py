# benchmarks/datasets/extrapolation.py
# Project-GP — Extrapolation Dataset Splits
# ═══════════════════════════════════════════════════════════════════════════════
"""
Generates train/test splits for out-of-distribution evaluation:
  1. Velocity extrapolation:  train vx ∈ [10,25], test vx ∈ [25,35]
  2. Lateral extrapolation:   train |ay| ≤ 1.2g, test 1.2g < |ay| ≤ 2.0g
  3. Combined extrapolation
"""

from __future__ import annotations

from benchmarks.datasets.interpolation import (
    generate_interpolation_dataset,
    _sample_operating_point,
    _sample_controls,
    _sample_setup_28,
)

import jax
import jax.numpy as jnp


def generate_extrapolation_splits(
    vehicle,
    n_train: int = 10000,
    n_test: int = 2000,
    dt: float = 0.005,
    seed: int = 123,
) -> dict:
    """
    Generate three extrapolation test sets.

    Returns dict with keys:
        velocity_train, velocity_test,
        lateral_train, lateral_test,
        combined_train, combined_test

    Each is a dict with {x, u, x_next, setup}.
    """
    # ── 1. Velocity extrapolation ──
    vel_data = generate_interpolation_dataset(
        vehicle, n_train=n_train, n_test=0, dt=dt, seed=seed,
        vx_range=(10.0, 25.0),
    )
    vel_test_data = generate_interpolation_dataset(
        vehicle, n_train=0, n_test=n_test, dt=dt, seed=seed + 100,
        vx_range=(25.0, 35.0),
    )

    # ── 2. Lateral extrapolation ──
    # Use wider vy/wz ranges for high lateral-g
    lat_train = generate_interpolation_dataset(
        vehicle, n_train=n_train, n_test=0, dt=dt, seed=seed + 200,
        vx_range=(15.0, 25.0),
    )
    # For test: higher lateral-g (achieved via higher wz and vy)
    lat_test = _generate_high_lateral(vehicle, n_test, dt, seed + 300)

    # ── 3. Combined ──
    combined_test = _generate_high_lateral(vehicle, n_test, dt, seed + 400,
                                           vx_range=(25.0, 35.0))

    return {
        "velocity_train": {
            "x": vel_data["train_x"], "u": vel_data["train_u"],
            "x_next": vel_data["train_x_next"], "setup": vel_data["train_setup"],
        },
        "velocity_test": {
            "x": vel_test_data["test_x"], "u": vel_test_data["test_u"],
            "x_next": vel_test_data["test_x_next"], "setup": vel_test_data["test_setup"],
        },
        "lateral_train": {
            "x": lat_train["train_x"], "u": lat_train["train_u"],
            "x_next": lat_train["train_x_next"], "setup": lat_train["train_setup"],
        },
        "lateral_test": lat_test,
        "combined_test": combined_test,
    }


def _generate_high_lateral(vehicle, n: int, dt: float, seed: int,
                           vx_range: tuple = (15.0, 25.0)) -> dict:
    """Generate samples with |ay| ∈ [1.2g, 2.0g]."""
    rng = jax.random.PRNGKey(seed)
    x_list, u_list, xn_list, s_list = [], [], [], []
    g = 9.81

    count = 0
    max_attempts = n * 10
    attempt = 0

    while count < n and attempt < max_attempts:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        attempt += 1

        vx, _, _ = _sample_operating_point(k1, vx_range=vx_range)
        # Higher wz for more lateral-g
        wz = jax.random.uniform(k1, (), minval=-2.5, maxval=2.5)
        vy = jax.random.uniform(k2, (), minval=-5.0, maxval=5.0)
        u = _sample_controls(k2)
        setup = _sample_setup_28(k3)

        # Approximate ay = vx * wz
        ay_approx = abs(float(vx * wz))
        if ay_approx < 1.2 * g or ay_approx > 2.0 * g:
            continue

        x0 = vehicle.make_initial_state(vx0=float(vx))
        x0 = x0.at[15].set(vy)
        x0 = x0.at[19].set(wz)

        x_next = vehicle.simulate_step(x0, u, setup, dt=dt, n_substeps=1)

        x_list.append(x0[:28])
        u_list.append(u)
        xn_list.append(x_next[:28])
        s_list.append(setup)
        count += 1

    return {
        "x": jnp.stack(x_list) if x_list else jnp.zeros((0, 28)),
        "u": jnp.stack(u_list) if u_list else jnp.zeros((0, 6)),
        "x_next": jnp.stack(xn_list) if xn_list else jnp.zeros((0, 28)),
        "setup": jnp.stack(s_list) if s_list else jnp.zeros((0, 28)),
    }
