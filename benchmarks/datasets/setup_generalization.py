# benchmarks/datasets/setup_generalization.py
# Project-GP — Setup Extrapolation Dataset
# ═══════════════════════════════════════════════════════════════════════════════
"""
Generates train/test splits where the test suspension setups lie OUTSIDE
the convex hull of training setups. Tests the model's ability to generalise
across the 28-parameter SuspensionSetup space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from benchmarks.datasets.interpolation import (
    _sample_operating_point,
    _sample_controls,
)


# Canonical parameter bounds — inner (train) and outer (test)
_PARAM_BOUNDS_TRAIN = {
    "k_f": (30000., 45000.),
    "k_r": (28000., 42000.),
    "arb_f": (4000., 9000.),
    "arb_r": (3500., 8000.),
    "c_lo_f": (5000., 5800.),
    "c_lo_r": (4200., 5200.),
}

_PARAM_BOUNDS_TEST = {
    "k_f": (22000., 30000., 45000., 58000.),  # below or above train
    "k_r": (20000., 28000., 42000., 55000.),
    "arb_f": (2000., 4000., 9000., 13000.),
    "arb_r": (1500., 3500., 8000., 11000.),
    "c_lo_f": (3500., 5000., 5800., 7000.),
    "c_lo_r": (3000., 4200., 5200., 6500.),
}


def _sample_setup_in_range(rng: jax.Array, bounds: dict,
                           base_setup: jax.Array) -> jax.Array:
    """Sample setup with specific parameters in given bounds, rest nominal."""
    setup = base_setup.copy()
    keys = jax.random.split(rng, len(bounds))
    param_indices = {"k_f": 0, "k_r": 1, "arb_f": 2, "arb_r": 3,
                     "c_lo_f": 4, "c_lo_r": 5}
    for i, (name, bound) in enumerate(bounds.items()):
        idx = param_indices[name]
        setup = setup.at[idx].set(
            jax.random.uniform(keys[i], (), minval=bound[0], maxval=bound[1]))
    return setup


def _sample_setup_ood(rng: jax.Array, bounds: dict,
                      base_setup: jax.Array) -> jax.Array:
    """Sample setup with key params OUTSIDE training bounds."""
    setup = base_setup.copy()
    keys = jax.random.split(rng, len(bounds))
    param_indices = {"k_f": 0, "k_r": 1, "arb_f": 2, "arb_r": 3,
                     "c_lo_f": 4, "c_lo_r": 5}
    for i, (name, (lo1, lo2, hi1, hi2)) in enumerate(bounds.items()):
        idx = param_indices[name]
        k1, k2 = jax.random.split(keys[i])
        # 50% chance below, 50% above
        below = jax.random.uniform(k1, (), minval=lo1, maxval=lo2)
        above = jax.random.uniform(k2, (), minval=hi1, maxval=hi2)
        use_above = jax.random.bernoulli(keys[i])
        setup = setup.at[idx].set(jnp.where(use_above, above, below))
    return setup


# Default base setup (nominal TeR27-class values)
_BASE_SETUP_28 = jnp.array([
    42000., 40000., 8000., 7000., 5400., 4800.,
    12000., 10000., 0.08, 0.08, 2.0, 2.0,
    0.028, 0.028, -0.035, -0.030, 0.001, -0.001,
    0.06, 0.18, 0.25, 0.18, 0.12, 0.0,
    0.60, 0.30, 0.0, 0.0,
])


def generate_setup_generalization_splits(
    vehicle,
    n_train: int = 10000,
    n_test: int = 2000,
    dt: float = 0.005,
    seed: int = 456,
) -> dict:
    """
    Generate train/test splits with setup extrapolation.

    Train setups: key parameters in nominal range.
    Test setups: key parameters OUTSIDE nominal range.
    Operating conditions: identical between train and test.
    """
    rng = jax.random.PRNGKey(seed)

    def _generate(rng, n, setup_sampler):
        x_list, u_list, xn_list, s_list = [], [], [], []
        for i in range(n):
            rng, k_op, k_ctrl, k_setup = jax.random.split(rng, 4)
            vx, vy, wz = _sample_operating_point(k_op, vx_range=(10.0, 25.0))
            u = _sample_controls(k_ctrl)
            setup = setup_sampler(k_setup)

            x0 = vehicle.make_initial_state(vx0=float(vx))
            x0 = x0.at[15].set(vy)
            x0 = x0.at[19].set(wz)
            x_next = vehicle.simulate_step(x0, u, setup, dt=dt, n_substeps=1)

            x_list.append(x0[:28])
            u_list.append(u)
            xn_list.append(x_next[:28])
            s_list.append(setup)

        return {
            "x": jnp.stack(x_list), "u": jnp.stack(u_list),
            "x_next": jnp.stack(xn_list), "setup": jnp.stack(s_list),
        }

    rng, k_train, k_test = jax.random.split(rng, 3)

    train = _generate(
        k_train, n_train,
        lambda k: _sample_setup_in_range(k, _PARAM_BOUNDS_TRAIN, _BASE_SETUP_28),
    )
    test = _generate(
        k_test, n_test,
        lambda k: _sample_setup_ood(k, _PARAM_BOUNDS_TEST, _BASE_SETUP_28),
    )

    return {"train": train, "test": test}
