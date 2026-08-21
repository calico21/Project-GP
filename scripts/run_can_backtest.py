"""
scripts/run_can_backtest.py — Safe Streaming Multiplexed CAN DBC Decoded 108-DOF Track Telemetry Replay Engine
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import cantools

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT

WINDOW_LEN = 250  # 1.25s rolling validation window (200 Hz)


def decode_can_csv_to_dataframe(file_path: Path, dbc_path: Path, dt: float = 0.005, lag_samples: int = 14) -> pd.DataFrame:
    # Safe line-by-line stream reading to prevent C-parser buffer overflows on malformed inputs
    records = []
    has_can_headers = False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_line = f.readline().strip()
        if 'ID' in header_line or 'Bus' in header_line or 'DataLen' in header_line:
            has_can_headers = True

        for line in f:
            parts = [p.strip() for p in line.strip().replace(';', ',').replace(' ', ',').split(',') if p.strip()]
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
                raw_id = parts[1]
                msg_id = int(raw_id, 16) if raw_id.lower().startswith('0x') else int(raw_id)
                data_hex = "".join(parts[2:])
                data_bytes = bytes.fromhex(data_hex) if len(data_hex) % 2 == 0 else bytes([int(p, 16) for p in parts[2:] if len(p) <= 2])
                
                records.append({'timestamp': t, 'id': msg_id, 'data': data_bytes})
            except Exception:
                continue

    # If rows successfully parsed as raw CAN frames
    if records and has_can_headers:
        db = cantools.database.load_file(dbc_path)
        decoded_rows = []
        for r in records:
            try:
                decoded = db.decode_message(r['id'], r['data'])
                decoded['timestamp'] = r['timestamp']
                decoded_rows.append(decoded)
            except Exception:
                continue
        raw_df = pd.DataFrame(decoded_rows)
    else:
        # Fallback to standard robust python engine parse if not raw CAN
        raw_df = pd.read_csv(file_path, engine='python', on_bad_lines='skip', low_memory=False)
        raw_df.columns = [c.strip() for c in raw_df.columns]

    return _standardize_and_resample(raw_df, dt, lag_samples)


def _extract_1d(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32).ravel()


def _standardize_and_resample(df: pd.DataFrame, dt: float, lag_samples: int) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        'ANGLE': 'steer_deg',
        'Yaw_Rate_z': 'yaw_rate_deg_s',
        'a_y': 'ay_mps2',
        'a_x': 'ax_mps2',
        'v_x': 'vx_mps',
        'vx_av': 'vx_mps_alt',
        'speed': 'speed_raw',
        'BPPS': 'bpps_raw',
        'APPS_AV': 'apps_raw',
        'leftDem': 't_dem_l',
        'rightDem': 't_dem_r',
        'rlTRQ': 't_rl',
        'rrTRQ': 't_rr',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in ['steer_deg', 'yaw_rate_deg_s', 'ay_mps2', 'ax_mps2', 'vx_mps', 'bpps_raw', 'apps_raw', 't_rl', 't_rr', 't_dem_l', 't_dem_r']:
        if col not in df.columns:
            df[col] = 0.0

    t_col = 'timestamp' if 'timestamp' in df.columns else ('time' if 'time' in df.columns else None)
    if t_col:
        df = df.dropna(subset=[t_col]).sort_values(by=t_col)
        t_arr = _extract_1d(df, t_col)
        valid_idx = np.where(np.isfinite(t_arr))[0]
        if len(valid_idx) > 0:
            df = df.iloc[valid_idx].copy()
            t_rel = t_arr[valid_idx] - t_arr[valid_idx][0]
            if len(t_rel) > 1 and t_rel[-1] > 0:
                t_uniform = np.arange(0, t_rel[-1], dt)
                df_resampled = pd.DataFrame({'t_rel': t_uniform})
                for c in df.columns:
                    if c not in [t_col, 't_rel']:
                        col_vals = _extract_1d(df, c)
                        df_resampled[c] = np.interp(t_uniform, t_rel, col_vals)
                df = df_resampled

    vx = _extract_1d(df, 'vx_mps')
    if np.max(np.abs(vx)) < 0.5 and 'speed_raw' in df.columns:
        vx = _extract_1d(df, 'speed_raw')
        if np.max(np.abs(vx)) > 50.0:
            vx = vx * (2.0 * np.pi / 60.0) * 0.2045
    df['vx_mps'] = vx

    ay = _extract_1d(df, 'ay_mps2')
    if np.std(ay) > 0.01 and np.std(ay) < 2.5:
        ay = ay * 9.81
    df['ay_mps2'] = ay

    steer = _extract_1d(df, 'steer_deg')
    steer_ratio = VP_DICT.get('steering_ratio', 4.5)
    if np.max(np.abs(steer)) > 35.0:
        steer = steer / steer_ratio

    wz = _extract_1d(df, 'yaw_rate_deg_s')
    if np.std(steer) > 1e-3 and np.std(wz) > 1e-3 and np.corrcoef(steer, wz)[0, 1] < -0.2:
        steer = -steer
    if np.std(wz) > 1e-3 and np.std(ay) > 1e-3 and np.corrcoef(wz, ay)[0, 1] < -0.2:
        ay = -ay

    df['ay_mps2'] = ay
    df['steer_deg'] = steer

    if lag_samples > 0 and len(steer) > lag_samples:
        shifted = np.roll(steer, -lag_samples)
        shifted[-lag_samples:] = shifted[-lag_samples - 1]
        steer = shifted
    df['steer_deg'] = steer

    t_dem_l = _extract_1d(df, 't_dem_l')
    t_dem_r = _extract_1d(df, 't_dem_r')
    if np.max(np.abs(t_dem_l)) > 0.1:
        df['t_fl'] = 0.0
        df['t_fr'] = 0.0
        df['t_rl'] = t_dem_l
        df['t_rr'] = t_dem_r
    else:
        apps = _extract_1d(df, 'apps_raw')
        dem_tq = (apps / 100.0) * 150.0
        df['t_fl'] = 0.0
        df['t_fr'] = 0.0
        df['t_rl'] = dem_tq * 0.5
        df['t_rr'] = dem_tq * 0.5

    bpps = _extract_1d(df, 'bpps_raw')
    df['brake_press'] = bpps * 10.0

    return df


@partial(jax.jit, static_argnums=(0, 3))
def _simulate_all_windows_jit(vehicle: DifferentiableMultiBodyVehicle, x0_batch: jax.Array, u_batch: jax.Array, dt: float):
    setup = vehicle._default_setup_vec

    def sim_one_window(x0, u_seq):
        def step_fn(x, u):
            x_next = vehicle.simulate_step(x, u, setup, dt=dt, n_substeps=2)
            ay = x_next[14] * x_next[19]
            return x_next, jnp.stack([x_next[14], x_next[19], ay])

        _, out = jax.lax.scan(step_fn, x0, u_seq)
        return out

    return jax.vmap(sim_one_window)(x0_batch, u_batch)


def run_session_backtest(vehicle: DifferentiableMultiBodyVehicle, df: pd.DataFrame, dt: float = 0.005):
    N = len(df)
    n_windows = N // WINDOW_LEN
    if n_windows == 0:
        return {'duration_s': N * dt, 'windows': 0, 'rmse_vx': 0, 'rmse_wz': 0, 'rmse_ay': 0, 'r_wz': 0, 'r_ay': 0, 'score': 0.0}

    steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg'))
    t_fl = _extract_1d(df, 't_fl')
    t_fr = _extract_1d(df, 't_fr')
    t_rl = _extract_1d(df, 't_rl')
    t_rr = _extract_1d(df, 't_rr')
    p_hyd = _extract_1d(df, 'brake_press')

    u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)

    real_wz_all = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
    real_ay_all = _extract_1d(df, 'ay_mps2')
    real_vx_all = _extract_1d(df, 'vx_mps')
    real_vy_all = _extract_1d(df, 'v_y') if 'v_y' in df.columns else np.zeros(N)

    u_windows, x0_windows, real_vx_wins, real_wz_wins, real_ay_wins = [], [], [], [], []

    for w in range(n_windows):
        start = w * WINDOW_LEN
        end = start + WINDOW_LEN
        u_windows.append(u_all[start:end])
        
        vx0 = float(max(real_vx_all[start], 1.0))
        vy0 = float(real_vy_all[start])
        wz0 = float(real_wz_all[start])
        
        x0 = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx0)
        x0 = x0.at[15].set(vy0)
        x0 = x0.at[19].set(wz0)
        x0_windows.append(x0)
        
        real_vx_wins.append(real_vx_all[start:end])
        real_wz_wins.append(real_wz_all[start:end])
        real_ay_wins.append(real_ay_all[start:end])

    u_batch = jnp.asarray(np.stack(u_windows))
    x0_batch = jnp.asarray(np.stack(x0_windows))

    sim_out = _simulate_all_windows_jit(vehicle, x0_batch, u_batch, dt)
    sim_out_np = np.array(sim_out)

    sim_vx = sim_out_np[:, :, 0].flatten()
    sim_wz = sim_out_np[:, :, 1].flatten()
    sim_ay = sim_out_np[:, :, 2].flatten()

    real_vx = np.concatenate(real_vx_wins)
    real_wz = np.concatenate(real_wz_wins)
    real_ay = np.concatenate(real_ay_wins)

    rmse_vx = float(np.sqrt(np.mean((sim_vx - real_vx) ** 2)))
    rmse_wz = float(np.sqrt(np.mean((sim_wz - real_wz) ** 2)))
    rmse_ay = float(np.sqrt(np.mean((sim_ay - real_ay) ** 2)))

    std_real_wz = np.std(real_wz)
    std_real_ay = np.std(real_ay)

    r_wz = float(np.corrcoef(sim_wz, real_wz)[0, 1]) if (std_real_wz > 1e-4 and np.std(sim_wz) > 1e-4) else 0.0
    r_ay = float(np.corrcoef(sim_ay, real_ay)[0, 1]) if (std_real_ay > 1e-4 and np.std(sim_ay) > 1e-4) else 0.0

    r_wz = float(np.nan_to_num(r_wz, nan=0.0))
    r_ay = float(np.nan_to_num(r_ay, nan=0.0))

    corr_score = (max(0.0, r_wz) * 0.45 + max(0.0, r_ay) * 0.45 + max(0.0, 1.0 - rmse_vx / 3.0) * 0.10) * 100.0

    return {
        'duration_s': N * dt,
        'windows': n_windows,
        'rmse_vx': rmse_vx,
        'rmse_wz': rmse_wz,
        'rmse_ay': rmse_ay,
        'r_wz': r_wz,
        'r_ay': r_ay,
        'score': corr_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_can_logs"))
    parser.add_argument("--dbc", type=Path, default=Path("TER.dbc"))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--lag-samples", type=int, default=14)
    args = parser.parse_args()

    files = sorted(list(args.data_dir.glob("*.csv")))
    if not files:
        print(f"[!] No CSV logs found in {args.data_dir}")
        return

    print(f"[*] Initializing Project-GP 108-DOF Engine for {len(files)} session log(s)...")
    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)

    print("\n=========================================================================================")
    print(f"{'Session File':<16} | {'Dur(s)':<7} | {'RMSE Vx':<9} | {'RMSE Yaw':<10} | {'RMSE Ay':<9} | {'Corr %':<7} | {'Phase r(Ay)'}")
    print("-----------------------------------------------------------------------------------------")

    scores = []
    for i, f in enumerate(files, 1):
        df = decode_can_csv_to_dataframe(f, dbc_path=args.dbc, dt=args.dt, lag_samples=args.lag_samples)
        res = run_session_backtest(vehicle, df, dt=args.dt)
        scores.append(res['score'])
        print(f"{f.name:<16} | {res['duration_s']:<7.1f} | {res['rmse_vx']:<9.3f} | {res['rmse_wz']:<10.3f} | {res['rmse_ay']:<9.3f} | {res['score']:<6.1f}% | {res['r_ay']:+.3f}")

    print("=========================================================================================")
    print(f"\n[*] Overall Mean Fleet Correlation Score: {np.mean(scores):.2f} %\n")


if __name__ == "__main__":
    main()