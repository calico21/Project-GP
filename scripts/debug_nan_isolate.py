# scripts/debug_nan_isolate2.py — localize which of the 108 blocks explodes
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
jax.config.update("jax_debug_nans", False)   # we want to inspect, not trap
jax.config.update("jax_disable_jit", True)

import jax.numpy as jnp
from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT

vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
setup = vehicle._default_setup_vec
u_zero = jnp.array([-0.007, 0., 0., 0., 0., 0.])
tire_cal = jnp.array([1.0, 1.0, -1.0, 1.0])

_BLOCKS = [
    ("q(kinematic pos)",   0,  14),
    ("v(kinematic vel)",  14,  28),
    ("thermal3D",         28,  56),
    ("transient_slip",    56,  72),
    ("damper_hyst",       72,  84),
    ("elastokin",         84, 108),
]

x = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=1.0)
x = x.at[15].set(0.0).at[19].set(0.0)

dt_sub = 0.005 / 2  # n_substeps=2 case
substep_idx = 0
for outer in range(4):
    for sub in range(2):
        x_next = vehicle._glrk4_step(x, u_zero, setup, dt_sub, tire_cal)
        bad = ~jnp.isfinite(x_next)
        if bool(jnp.any(bad)):
            print(f"NaN/Inf first appears at outer={outer} sub={sub} (global substep {substep_idx})")
            for name, lo, hi in _BLOCKS:
                n_bad = int(jnp.sum(bad[lo:hi]))
                if n_bad:
                    idxs = jnp.nonzero(bad[lo:hi])[0] + lo
                    print(f"  {name:<16} {n_bad} bad  idx={idxs.tolist()}  "
                          f"vals={x_next[idxs].tolist()}")
            print(f"  prev x at those idx: {[float(x[i]) for i in jnp.nonzero(bad)[0].tolist()]}")
            sys.exit(0)
        x = x_next
        substep_idx += 1
print("no NaN in 8 substeps")