# scripts/debug_rear_axle_magnitude.py
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle, SuspensionSetup
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import (
    decode_can_csv_to_dataframe, _extract_1d, WINDOW_LEN,
    _vy0_from_yaw_drift, _probe_best_steer_sign,
)

vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
setup = vehicle._default_setup_vec

df = decode_can_csv_to_dataframe(Path("data/raw_can_logs/1.csv"),
                                  dbc_path=Path("TER.dbc"), dt=0.005)
steer_sign = _probe_best_steer_sign(vehicle, df, dt=0.005)

real_vx = _extract_1d(df, 'vx_mps')
real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
real_ay = _extract_1d(df, 'ay_mps2')

# Pick the window with the highest sustained |wz| — most rear-slip-loaded segment
n_windows = len(df) // WINDOW_LEN
window_wz_energy = [np.sum(real_wz[w*WINDOW_LEN:(w+1)*WINDOW_LEN]**2) for w in range(n_windows)]
w0 = int(np.argmax(window_wz_energy))
s, e = w0*WINDOW_LEN, w0*WINDOW_LEN + WINDOW_LEN
print(f"[*] Using window {w0} (highest wz energy), s={s}")

steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg')) * steer_sign
t_fl, t_fr = _extract_1d(df, 't_fl'), _extract_1d(df, 't_fr')
t_rl, t_rr = _extract_1d(df, 't_rl'), _extract_1d(df, 't_rr')
p_hyd = _extract_1d(df, 'brake_press')
u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)
u_all = np.nan_to_num(u_all, nan=0.0, posinf=0.0, neginf=0.0)
u_all[:, 1:5] = np.clip(u_all[:, 1:5], -50.0, 400.0)
u_all[:, 5]   = np.clip(u_all[:, 5],   0.0,   2000.0)

vx0 = float(max(real_vx[s], 3.0))
wz0 = float(real_wz[s])
vy0 = _vy0_from_yaw_drift(vx0, wz0)

x = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx0)
x = x.at[15].set(vy0).at[19].set(wz0)

tire_cal = jnp.array([1.18, 1.80, -1.0, 1.0], dtype=jnp.float32)  # current calibrated

s_setup = SuspensionSetup.from_vector(setup)

print(f"{'i':>3} {'wz_sim':>8} {'wz_real':>8} {'ay_sim':>8} {'ay_real':>8} "
      f"{'Fy_rl':>8} {'Fy_rr':>8} {'Fz_rl':>8} {'Fz_rr':>8} {'gam_rl':>7} {'gam_rr':>7}")

with jax.disable_jit():
    for i in range(WINDOW_LEN):
        u = jnp.array(u_all[s+i])
        x_next = vehicle.simulate_step(x, u, setup, dt=0.005, n_substeps=2, tire_cal=tire_cal)

        # Re-derive rear tire forces at this state for diagnostic printing
        q = x[0:14]; v = x[14:28]
        vx_, vy_, wz_ = v[0], v[1], v[19-14]
        z_rl, z_rr = q[8], q[9]
        tr2 = vehicle.track_r / 2.0
        v_wheel_rl = vx_ - wz_ * tr2
        v_wheel_rr = vx_ + wz_ * tr2

        if i % 10 == 0:
            print(f"{i:3d} {float(x_next[19]):8.4f} {real_wz[s+i]:8.4f} "
                  f"{float(x_next[14]*x_next[19]):8.4f} {real_ay[s+i]:8.4f}")

        x = x_next
        if not bool(jnp.all(jnp.isfinite(x))):
            print("NON-FINITE — stopping")
            break

print("\n[*] Compare ay_sim vs ay_real growth rate. If sim ay plateaus while "
      "real ay keeps climbing (or vice versa), that's the rear-axle scale "
      "signature — same shape as the original front-axle defect.")