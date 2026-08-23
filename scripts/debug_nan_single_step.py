# scripts/debug_nan_single_step.py
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", True)   # eager from the start — no slow de-opt rerun

import jax.numpy as jnp
from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import decode_can_csv_to_dataframe, _extract_1d

vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
setup = vehicle._default_setup_vec

df = decode_can_csv_to_dataframe(Path("data/raw_can_logs/1.csv"),
                                  dbc_path=Path("TER.dbc"), dt=0.005)

steer = jnp.array(jnp.deg2rad(_extract_1d(df, 'steer_deg')[0]))
u0 = jnp.array([
    steer,
    _extract_1d(df, 't_fl')[0],
    _extract_1d(df, 't_fr')[0],
    _extract_1d(df, 't_rl')[0],
    _extract_1d(df, 't_rr')[0],
    _extract_1d(df, 'brake_press')[0],
])

vx0 = float(max(_extract_1d(df, 'vx_mps')[0], 1.0))
print(f"[diag] vx0={vx0}  u0={u0}")

x0 = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx0)

x1 = vehicle.simulate_step(
    x0, u0, setup, dt=0.005, n_substeps=1,
    tire_cal=jnp.array([1.0, 1.0, -1.0, 1.0]),
)
print("[diag] step OK:", x1)