from models.vehicle_dynamics import SETUP_LB, SETUP_UB
import jax.numpy as jnp
from experiments.paper_suite.setup_optimization_pilot import projected_fraction_step, gradient_comparison

def test_suspension_bounds_are_ordered():
    assert bool((SETUP_LB <= SETUP_UB).all())

def test_fraction_step_enforces_local_physical_box():
    result = projected_fraction_step(jnp.array([1.0, 1.0]), jnp.array([100.0, -100.0]), step_size=1.0)
    assert bool(jnp.all(result >= .85)) and bool(jnp.all(result <= 1.15))

def test_gradient_comparison_schema_and_identity():
    result = gradient_comparison(jnp.array([1.0, -2.0]), jnp.array([1.0, -2.0]))
    assert set(result) == {"absolute_l2", "relative_error", "cosine"}
    assert result["absolute_l2"] == 0.0 and abs(result["cosine"] - 1.0) < 1e-12
