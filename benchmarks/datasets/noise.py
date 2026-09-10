# benchmarks/datasets/noise.py
# Project-GP — Noise Robustness Dataset Augmentation
# ═══════════════════════════════════════════════════════════════════════════════
"""
Adds calibrated Gaussian noise to state observations at varying SNR levels,
simulating real sensor noise (IMU, encoders, GPS).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# Realistic per-state noise standard deviations (108-DOF model)
# Derived from sensor specs: MPU-6050 IMU, optical encoders, GPS
_STATE_NOISE_STD_28 = jnp.array([
    # q: positions (14)
    0.01, 0.01, 0.001,       # X, Y [m] (GPS), Z [m] (suspension)
    0.002, 0.002, 0.005,     # roll, pitch, yaw [rad]
    0.0005, 0.0005, 0.0005, 0.0005,  # z_fl..z_rr [m] (pot sensor)
    0.01, 0.01, 0.01, 0.01,  # theta_fl..theta_rr [rad] (encoder)
    # v: velocities (14)
    0.1, 0.15, 0.05,         # vx, vy, vz [m/s]
    0.005, 0.005, 0.005,     # wx, wy, wz [rad/s] (gyro)
    0.001, 0.001, 0.001, 0.001,  # dz_fl..dz_rr [m/s]
    0.3, 0.3, 0.3, 0.3,      # omega_fl..omega_rr [rad/s] (encoder)
])


def add_measurement_noise(
    dataset: dict,
    snr_db: float = 30.0,
    seed: int = 0,
) -> dict:
    """
    Add Gaussian measurement noise to dataset states.

    Args:
        dataset: dict with keys x, u, x_next, setup
        snr_db: signal-to-noise ratio in dB. Lower = noisier.
        seed: random seed

    Returns:
        New dataset dict with noisy x and x_next.
    """
    rng = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(rng)

    # Scale noise by SNR
    snr_linear = 10.0 ** (snr_db / 20.0)
    noise_scale = 1.0 / snr_linear

    x = dataset["x"]
    x_next = dataset["x_next"]
    n, d = x.shape

    std = _STATE_NOISE_STD_28[:d] * noise_scale

    x_noisy = x + std[None, :] * jax.random.normal(k1, x.shape)
    xn_noisy = x_next + std[None, :] * jax.random.normal(k2, x_next.shape)

    return {
        "x": x_noisy,
        "u": dataset["u"],
        "x_next": xn_noisy,
        "setup": dataset["setup"],
    }


def generate_noise_robustness_suite(
    dataset: dict,
    snr_levels: tuple = (40.0, 30.0, 20.0, 10.0),
    seed: int = 0,
) -> dict[float, dict]:
    """
    Generate noisy versions of a dataset at multiple SNR levels.

    Returns:
        dict mapping SNR (dB) → noisy dataset
    """
    results = {}
    for i, snr in enumerate(snr_levels):
        results[snr] = add_measurement_noise(dataset, snr_db=snr, seed=seed + i)
    return results
