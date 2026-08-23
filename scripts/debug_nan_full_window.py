# scripts/debug_nan_full_window.py
import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", True)

import numpy as np
import jax.numpy as jnp
from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import decode_can_csv_to_dataframe, _extract_1d, _estimate_vy_kinematic, WINDOW_LEN

vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
setup = vehicle._default_setup_vec

df = decode_can_csv_to_dataframe(Path("data/raw_can_logs/1.csv"),
                                  dbc_path=Path("TER.dbc"), dt=0.005)

steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg'))
t_fl, t_fr = _extract_1d(df, 't_fl'), _extract_1d(df, 't_fr')
t_rl, t_rr = _extract_1d(df, 't_rl'), _extract_1d(df, 't_rr')
p_hyd = _extract_1d(df, 'brake_press')
u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)
u_all = np.nan_to_num(u_all, nan=0.0, posinf=0.0, neginf=0.0)
u_all[:, 1:5] = np.clip(u_all[:, 1:5], -50.0, 400.0)
u_all[:, 5]   = np.clip(u_all[:, 5],   0.0,   2000.0)

real_vx = _extract_1d(df, 'vx_mps')
real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
real_ay = _extract_1d(df, 'ay_mps2')
vy_est  = _estimate_vy_kinematic(real_ay, real_vx, real_wz, 0.005)

WINDOW = 0
s = WINDOW * WINDOW_LEN
e = s + WINDOW_LEN

vx0 = float(max(real_vx[s], 1.0))
vy0 = float(np.clip(vy_est[s], -15.0, 15.0))
wz0 = float(real_wz[s])
print(f"[diag] window={WINDOW} vx0={vx0} vy0={vy0} wz0={wz0}")

x = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx0)
x = x.at[15].set(vy0).at[19].set(wz0)

for i in range(WINDOW_LEN):
    u = jnp.array(u_all[s + i])
    try:
        x = vehicle.simulate_step(
            x, u, setup, dt=0.005, n_substeps=2,
            tire_cal=jnp.array([1.0, 1.0, -1.0, 1.0]),
        )
    except Exception as ex:
        print(f"[diag] BROKE at step {i}  u={u}")
        raise
    if i % 25 == 0:
        print(f"  step {i:3d}  vx={x[14]:+.3f}  vy={x[15]:+.3f}  wz={x[19]:+.3f}  "
              f"z=[{x[6]:+.4f},{x[7]:+.4f},{x[8]:+.4f},{x[9]:+.4f}]")

print("[diag] full window OK")