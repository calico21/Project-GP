"""Regression checks for the production GLRK stage/output contract."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from config.tire_coeffs import tire_coeffs as TC
from config.vehicles.ter27 import vehicle_params_ter27 as VP
from experiments.gradient_accuracy import _stage_solution_diagnostics
from models.vehicle_dynamics import (
    DEFAULT_SETUP,
    SETUP_LB,
    DifferentiableMultiBodyVehicle,
)


def test_production_step_matches_converged_stage_reconstruction():
    """The IFT F/G pair must be the same map as simulate_step()."""
    vehicle = DifferentiableMultiBodyVehicle(VP, TC)
    x0 = vehicle.make_initial_state(vx0=20.0).astype(jnp.float64)
    u = jnp.array([0.03, 10.0, 10.0, 10.0, 10.0, 0.0], dtype=jnp.float64)
    # Valid SI setup in the observed Picard-convergence regime.
    setup = (0.5 * (DEFAULT_SETUP + SETUP_LB)).astype(jnp.float64)

    diagnostic = _stage_solution_diagnostics(vehicle, x0, u, setup, 0.005)
    production = vehicle.simulate_step(x0, u, setup, dt=0.005, n_substeps=1)

    assert diagnostic["residual_inf"] < 1e-8
    assert float(jnp.max(jnp.abs(production - diagnostic["state"]))) < 1e-12
