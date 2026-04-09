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

@partial(jax.jit, static_argnums=(6, 7))
def simple_dyc_torque_vectoring(
    vx: jax.Array,           
    wz: jax.Array,           
    delta: jax.Array,        
    Fx_driver: jax.Array,    
    T_min: jax.Array,        
    T_max: jax.Array,        
    Kp_yaw: float = 800.0,   
    geo: TVGeometry = DEFAULT_TER27_GEO,
) -> jax.Array:
    """
    Algebraic Direct Yaw Control (DYC) allocator.
    Dynamically pulls geometry from the Ter27 master config to maintain a single source of truth.
    """
    
    T_total_req = Fx_driver * geo.r_w
    T_base = T_total_req / 4.0
    
    vx_safe = jnp.maximum(jnp.abs(vx), 1.0)
    L = geo.lf + geo.lr
    
    K_us = geo.mass * (geo.lr - geo.lf) / (L * 30000.0) 
    wz_target = (vx_safe * delta) / (L + K_us * vx_safe ** 2)
    
    yaw_error = wz_target - wz
    Mz_demand = Kp_yaw * yaw_error
    

    avg_track = (geo.track_f + geo.track_r) / 2.0
    delta_T = (Mz_demand * geo.r_w) / avg_track
    
    T_fl = T_base - delta_T
    T_fr = T_base + delta_T
    T_rl = T_base - delta_T
    T_rr = T_base + delta_T
    
    T_commanded = jnp.array([T_fl, T_fr, T_rl, T_rr])
    
    T_final = jnp.clip(T_commanded, T_min, T_max)
    
    return T_final