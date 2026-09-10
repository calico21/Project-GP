# benchmarks/datasets/interpolation.py
# Project-GP — In-Distribution Dataset Generator
# ═══════════════════════════════════════════════════════════════════════════════
"""
Generates training and test datasets from the same operational envelope.
Uses the 108-DOF DifferentiableMultiBodyVehicle to produce ground-truth
one-step transitions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _sample_operating_point(rng: jax.Array,
                            vx_range: tuple = (10.0, 25.0),
                            vy_range: tuple = (-3.0, 3.0),
                            wz_range: tuple = (-1.5, 1.5),
                            ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample a single random operating point."""
    k1, k2, k3 = jax.random.split(rng, 3)
    vx = jax.random.uniform(k1, (), minval=vx_range[0], maxval=vx_range[1])
    vy = jax.random.uniform(k2, (), minval=vy_range[0], maxval=vy_range[1])
    wz = jax.random.uniform(k3, (), minval=wz_range[0], maxval=wz_range[1])
    return vx, vy, wz


def _sample_controls(rng: jax.Array,
                     delta_range: tuple = (-0.15, 0.15),
                     torque_range: tuple = (-21.0, 21.0),
                     ) -> jax.Array:
    """Sample control inputs: [delta, T_fl, T_fr, T_rl, T_rr, F_brake]."""
    k1, k2, k3 = jax.random.split(rng, 3)
    delta = jax.random.uniform(k1, (), minval=delta_range[0], maxval=delta_range[1])
    torque = jax.random.uniform(k2, (), minval=torque_range[0], maxval=torque_range[1])
    brake = jax.random.uniform(k3, (), minval=0.0, maxval=2000.0)
    # Equal torque split
    return jnp.array([delta, torque, torque, torque, torque, brake])


def _sample_setup_28(rng: jax.Array) -> jax.Array:
    """Sample a random 28-param suspension setup within nominal bounds."""
    k = jax.random.split(rng, 28)
    # Nominal ranges for each of the 28 suspension parameters
    lo = jnp.array([
        25000., 25000., 2000., 2000., 5000., 4000.,   # k_f, k_r, arb_f, arb_r, c_lo_f, c_lo_r
        8000., 7000., 0.05, 0.05, 1.5, 1.5,           # c_hi_f, c_hi_r, v_knee_f, v_knee_r, reb_f, reb_r
        0.020, 0.020, -0.065, -0.055, -0.005, -0.005, # h_ride_f, h_ride_r, camber_f, camber_r, toe_f, toe_r
        0.04, 0.10, 0.15, 0.10, 0.08, 0.0,            # castor, anti_squat, anti_dive_f, anti_dive_r, anti_lift, diff
        0.55, 0.25, -0.10, -0.08,                     # brake_bias, h_cg, bump_steer_f, bump_steer_r
    ])
    hi = jnp.array([
        55000., 55000., 12000., 10000., 6000., 5500.,
        14000., 12000., 0.15, 0.15, 2.5, 2.5,
        0.035, 0.035, -0.020, -0.015, 0.005, 0.005,
        0.08, 0.25, 0.35, 0.25, 0.18, 1.0,
        0.70, 0.35, 0.10, 0.08,
    ])
    return jax.random.uniform(rng, (28,), minval=lo, maxval=hi)


def generate_interpolation_dataset(
    vehicle,
    n_train: int = 10000,
    n_test: int = 2000,
    dt: float = 0.005,
    seed: int = 42,
    vx_range: tuple = (10.0, 25.0),
) -> dict:
    """
    Generate in-distribution train/test datasets.

    Returns:
        dict with keys: train_{x, u, x_next, setup}, test_{x, u, x_next, setup}

    Each x is (N, state_dim), u is (N, 6), setup is (N, 28).
    """
    rng = jax.random.PRNGKey(seed)

    def _generate_batch(rng, n):
        x_list, u_list, xn_list, s_list = [], [], [], []

        for i in range(n):
            rng, k_op, k_ctrl, k_setup = jax.random.split(rng, 4)
            vx, vy, wz = _sample_operating_point(k_op, vx_range=vx_range)
            u = _sample_controls(k_ctrl)
            setup_vec = _sample_setup_28(k_setup)

            # Build initial state
            x0 = vehicle.make_initial_state(vx0=float(vx))
            x0 = x0.at[15].set(vy)   # vy
            x0 = x0.at[19].set(wz)   # wz

            # One-step transition
            x_next = vehicle.simulate_step(x0, u, setup_vec, dt=dt, n_substeps=1)

            # Store mechanical states only (first 28)
            x_list.append(x0[:28])
            u_list.append(u)
            xn_list.append(x_next[:28])
            s_list.append(setup_vec)

        return {
            "x": jnp.stack(x_list),
            "u": jnp.stack(u_list),
            "x_next": jnp.stack(xn_list),
            "setup": jnp.stack(s_list),
        }

    rng, k_train, k_test = jax.random.split(rng, 3)
    train = _generate_batch(k_train, n_train)
    test = _generate_batch(k_test, n_test)

    return {
        "train_x": train["x"], "train_u": train["u"],
        "train_x_next": train["x_next"], "train_setup": train["setup"],
        "test_x": test["x"], "test_u": test["u"],
        "test_x_next": test["x_next"], "test_setup": test["setup"],
    }
