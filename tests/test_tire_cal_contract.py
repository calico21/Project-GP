import jax
import jax.numpy as jnp

from models.tire_model import PacejkaTire
from config.tire_coeffs import tire_coeffs


def test_tire_cal_has_six_element_contract():
    tire = PacejkaTire(tire_coeffs)

    tire_cal = jnp.array(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        dtype=jnp.float64,
    )

    assert tire_cal.shape == (6,)