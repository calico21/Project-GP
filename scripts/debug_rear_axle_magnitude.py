# scripts/debug_rear_axle_magnitude.py
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# MUST be set before any vehicle_dynamics import/trace picks it up
os.environ["GP_DEBUG_YAW"] = "1"

import argparse
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

# ─────────────────────────────────────────────────────────────────────────────
# CLI — point this at the specific offending window from the debug breakdown,
# e.g. session 5 window 1273 (ay_peak=10.7, rmse=7.7) or session 2 window 151
# (ay_peak=13.35, rmse=6.6, part of that session's saturation cluster).
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--session", type=Path, default=Path("data/raw_can_logs/1.csv"))
ap.add_argument("--window", type=int, default=None,
                 help="Window index (from reports/debug_backtest/<session>_windows.csv). "
                      "If omitted, falls back to the old highest-wz-energy heuristic.")
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--mu-f", type=float, default=1.18)
ap.add_argument("--mu-r", type=float, default=1.80)
args = ap.parse_args()

vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
setup = vehicle._default_setup_vec

df = decode_can_csv_to_dataframe(args.session, dbc_path=Path("TER.dbc"), dt=args.dt)
steer_sign = _probe_best_steer_sign(vehicle, df, dt=args.dt)

real_vx = _extract_1d(df, 'vx_mps')
real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
real_ay = _extract_1d(df, 'ay_mps2')

n_windows = len(df) // WINDOW_LEN

if args.window is not None:
    w0 = args.window
    if w0 >= n_windows:
        raise ValueError(f"--window {w0} out of range (session has {n_windows} windows)")
    print(f"[*] Using explicit window {w0} from {args.session.name}")
else:
    window_wz_energy = [np.sum(real_wz[w*WINDOW_LEN:(w+1)*WINDOW_LEN]**2) for w in range(n_windows)]
    w0 = int(np.argmax(window_wz_energy))
    print(f"[*] No --window given — using window {w0} (highest wz energy)")

s, e = w0 * WINDOW_LEN, w0 * WINDOW_LEN + WINDOW_LEN
print(f"[*] s={s}  ay_peak={np.max(np.abs(real_ay[s:e])):.2f}  "
      f"wz_peak={np.max(np.abs(real_wz[s:e])):.3f}")

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

tire_cal = jnp.array([args.mu_f, args.mu_r, -1.0, 1.0], dtype=jnp.float32)

print(f"\n[*] tire_cal: mu_f={args.mu_f}  mu_r={args.mu_r}")
print(f"[*] Rear-axle quantities stream from vehicle_dynamics.py's GP_DEBUG_YAW "
      f"jax.debug.print block below. jax.debug.print works under jit, so this "
      f"runs compiled (fast) — no disable_jit() needed.\n")

print(f"{'i':>3} {'wz_sim':>8} {'wz_real':>8} {'ay_sim':>8} {'ay_real':>8}")

for i in range(WINDOW_LEN):
    u = jnp.array(u_all[s + i])
    x_next = vehicle.simulate_step(x, u, setup, dt=args.dt, n_substeps=2, tire_cal=tire_cal)

    if i % 10 == 0:
        print(f"{i:3d} {float(x_next[19]):8.4f} {real_wz[s+i]:8.4f} "
              f"{float(x_next[14]*x_next[19]):8.4f} {real_ay[s+i]:8.4f}")

    x = x_next
    if not bool(jnp.all(jnp.isfinite(x))):
        print("NON-FINITE — stopping")
        break

print("\n[*] What to look for in the GP_DEBUG_YAW stream above:")
print("    - Does Fy_rl/Fy_rr PLATEAU while Fz_rl/Fz_rr keeps climbing through")
print("      the window? That's the rear tire force ceiling — confirms mu_r=1.80")
print("      is a hard cap, not converged physics.")
print("    - Compare ay_sim vs ay_real growth rate above: if sim ay plateaus")
print("      while real ay keeps climbing (or vice versa), that's the same")
print("      rear-axle scale signature at the aggregate level.")