# scripts/debug_yaw_instability.py
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
jax.config.update("jax_debug_nans", True) 
import jax.numpy as jnp
import numpy as np

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import (
    decode_can_csv_to_dataframe, _extract_1d, WINDOW_LEN, _estimate_vy_kinematic,
    _probe_best_steer_sign,
)

vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
setup   = vehicle._default_setup_vec

df = decode_can_csv_to_dataframe(
    __import__('pathlib').Path("data/raw_can_logs/1.csv"),
    dbc_path=__import__('pathlib').Path("TER.dbc"), dt=0.005)
steer_sign = _probe_best_steer_sign(vehicle, df, dt=0.005)

real_vx = _extract_1d(df, 'vx_mps')
MIN_VX0 = 3.0
n_windows = len(df) // WINDOW_LEN
candidate = [w for w in range(n_windows) if real_vx[w*WINDOW_LEN] >= MIN_VX0]
w0 = candidate[0]   # first qualifying window — deterministic, not random-sampled
s, e = w0*WINDOW_LEN, w0*WINDOW_LEN + WINDOW_LEN

steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg')) * steer_sign
t_fl, t_fr = _extract_1d(df, 't_fl'), _extract_1d(df, 't_fr')
t_rl, t_rr = _extract_1d(df, 't_rl'), _extract_1d(df, 't_rr')
p_hyd = _extract_1d(df, 'brake_press')
u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)
u_all = np.nan_to_num(u_all, nan=0.0, posinf=0.0, neginf=0.0)
u_all[:, 1:5] = np.clip(u_all[:, 1:5], -50.0, 400.0)
u_all[:, 5]   = np.clip(u_all[:, 5],   0.0,   2000.0)

print(f"\n--- window w0={w0}  s={s}  raw inputs, first 14 steps ---")
print(f"{'i':>3} {'steer_rad':>10} {'t_rl':>8} {'t_rr':>8} {'brake':>8} {'vx_real':>8}")
for i in range(14):
    print(f"{i:3d} {u_all[s+i,0]:10.4f} {u_all[s+i,3]:8.2f} {u_all[s+i,4]:8.2f} "
          f"{u_all[s+i,5]:8.2f} {real_vx[s+i]:8.3f}")

torque_asym = u_all[s:s+14, 3] - u_all[s:s+14, 4]   # t_rl - t_rr
print(f"\nmax |t_rl - t_rr| over window: {np.max(np.abs(torque_asym)):.2f} Nm")
print(f"mean t_rl: {np.mean(u_all[s:s+14,3]):.2f}   mean t_rr: {np.mean(u_all[s:s+14,4]):.2f}")

real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
real_ay = _extract_1d(df, 'ay_mps2')
vy_est  = _estimate_vy_kinematic(real_ay, real_vx, real_wz, 0.005)

x0 = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=float(max(real_vx[s], MIN_VX0)))
x0 = x0.at[15].set(float(np.clip(vy_est[s], -15.0, 15.0))).at[19].set(float(real_wz[s]))

tire_cal_nominal = jnp.array([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32)
with jax.disable_jit():
    x = x0
    print(f"{'step':>4} {'wz_sim':>10} {'wz_real':>10} {'ay_sim':>10} {'ay_real':>10} {'vx':>8} {'vy':>8}")
    for i in range(40):
        u = jnp.array(u_all[s+i])
        x = vehicle.simulate_step(x, u, setup, dt=0.005, n_substeps=2, tire_cal=tire_cal_nominal)
        print(f"{i:4d} {float(x[19]):10.4f} {real_wz[s+i]:10.4f} "
            f"{float(x[14]*x[19]):10.4f} {real_ay[s+i]:10.4f} "
            f"{float(x[14]):8.3f} {float(x[15]):8.3f}")
        if not bool(jnp.all(jnp.isfinite(x))):
            print("NON-FINITE — stopping")
            break