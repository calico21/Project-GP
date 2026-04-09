import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from config.vehicles.ter27 import vehicle_params_ter27

class TVGeometry(NamedTuple):
    """Static vehicle geometry dynamically loaded from config."""
    lf: float
    lr: float
    track_f: float
    track_r: float
    r_w: float
    mass: float

    @staticmethod
    def from_vehicle_params(vp: dict) -> 'TVGeometry':
        return TVGeometry(
            lf=vp.get('lf', 0.806),
            lr=vp.get('lr', 0.744),
            track_f=vp.get('track_front', 1.220),
            track_r=vp.get('track_rear', 1.200),
            r_w=vp.get('wheel_radius', 0.2032),
            mass=vp.get('total_mass', 320.0),
        )

DEFAULT_TER27_GEO = TVGeometry.from_vehicle_params(vehicle_params_ter27)

@partial(jax.jit, static_argnums=(6, 7, 8))
def simple_dyc_torque_vectoring(
    vx: jax.Array,
    wz: jax.Array,
    delta: jax.Array,
    Fx_driver: jax.Array,
    T_min: jax.Array,
    T_max: jax.Array,
    Kp_yaw: float = 800.0,
    geo: TVGeometry = DEFAULT_TER27_GEO,
    is_rwd: bool = False,
) -> jax.Array:
    """
    Velocity-adaptive P-DYC allocator.

    is_rwd=True  (Ter26): driven=[RL, RR] — T_base = Fx·r_w / 2, arm = track_r.
                          Front torques are hard-zero (undriven axle).
    is_rwd=False (Ter27): driven=[FL,FR,RL,RR] — T_base = Fx·r_w / 4,
                          arm denominator = track_f + track_r (NOT avg — see audit).

    Both branches: Kp decays as v_ref/v above 5 m/s to preserve closed-loop bandwidth.
    Resolved at XLA compile time via Python if — separate graph per config, zero runtime cost.
    """
    T_total_req = Fx_driver * geo.r_w
    vx_safe = jnp.maximum(jnp.abs(vx), 1.0)
    L = geo.lf + geo.lr

    K_us = geo.mass * (geo.lr - geo.lf) / (L * 30000.0)
    wz_target = (vx_safe * delta) / (L + K_us * vx_safe ** 2)

    # Velocity-adaptive Kp: full gain at ≤5 m/s, decays as 1/v above
    Kp_eff = Kp_yaw * jnp.minimum(1.0, 5.0 / vx_safe)
    Mz_demand = Kp_eff * (wz_target - wz)

    if is_rwd:
        # Rear-only drive: Fx split between 2 wheels; front is undriven (hard zero)
        # Moment arm = track_r (rear axle only contributes to M_z)
        T_base = T_total_req / 2.0
        delta_T = (Mz_demand * geo.r_w) / geo.track_r
        T_commanded = jnp.array([
            jnp.zeros(()),     # FL — undriven
            jnp.zeros(()),     # FR — undriven
            T_base - delta_T,  # RL — inner wheel for positive-yaw (left) turn
            T_base + delta_T,  # RR — outer wheel for positive-yaw (left) turn
        ])
    else:
        # All-wheel drive: Fx split between 4 wheels
        # Denominator = track_f + track_r: M_z = delta_T*(track_f+track_r)/r_w → Mz_demand ✓
        T_base = T_total_req / 4.0
        delta_T = (Mz_demand * geo.r_w) / (geo.track_f + geo.track_r)
        T_commanded = jnp.array([
            T_base - delta_T,  # FL
            T_base + delta_T,  # FR
            T_base - delta_T,  # RL
            T_base + delta_T,  # RR
        ])

    return jnp.clip(T_commanded, T_min, T_max)