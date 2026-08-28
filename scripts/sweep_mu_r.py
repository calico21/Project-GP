#!/usr/bin/env python3
"""Isolate whether mu_r=1.600 is a genuine minimum or a soft-clip artifact."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.calibrate_mu_from_telemetry import _build_calib_batch, _soft_corr_loss
from scripts.run_can_backtest import decode_can_csv_to_dataframe

MU_F_FIXED = 1.196

vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
setup = vehicle._default_setup_vec
rng = np.random.default_rng(0)

files = sorted(Path("data/raw_can_logs").glob("*.csv"))
batches = [_build_calib_batch(decode_can_csv_to_dataframe(f, dbc_path=Path("TER.dbc"), dt=0.005),
                               0.005, 1.0, rng) for f in files]

def eval_mu_r(mu_r_val, u_seq, x0, wz_real, ay_real):
    tire_cal = jnp.array([MU_F_FIXED, mu_r_val, -1.0, 1.0], dtype=jnp.float32)
    def step_fn(x, u):
        x_next = vehicle.simulate_step(x, u, setup, dt=0.005, n_substeps=4, tire_cal=tire_cal)
        return x_next, jnp.array([x_next[19], x_next[14]*x_next[19]])
    from jax import lax, vmap
    def one(x0_, u_seq_):
        _, out = lax.scan(step_fn, x0_, u_seq_)
        return out[:, 0], out[:, 1]
    wz_sim, ay_sim = vmap(one)(x0, u_seq)
    return float(0.5*_soft_corr_loss(wz_sim, wz_real) + 0.5*_soft_corr_loss(ay_sim, ay_real))

for mu_r in np.arange(1.0, 2.01, 0.1):
    losses = [eval_mu_r(mu_r, *b[:2], b[2], b[3]) for b in batches]
    print(f"mu_r={mu_r:.2f}  mean_loss={np.mean(losses):.4f}")