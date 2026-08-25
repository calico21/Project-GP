"""
scripts/run_can_backtest.py — Safe Streaming Multiplexed CAN DBC Decoded 108-DOF Track Telemetry Replay Engine
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.signal import butter, filtfilt

def _lowpass_corr(x: np.ndarray, fs: float = 200.0, cutoff: float = 8.0) -> np.ndarray:
    if len(x) < 15:
        return x
    b, a = butter(2, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, x)
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT

WINDOW_LEN = 100  # 0.5s rolling validation window (200 Hz) — shorter windows reduce open-loop divergence

def _estimate_vy_kinematic(ay: np.ndarray, vx: np.ndarray, wz: np.ndarray,
                            dt: float, tau: float = 2.0) -> np.ndarray:
    """
    Leaky-integrator (complementary-filter) estimate of body-frame lateral
    velocity, used ONLY to seed window initial conditions in
    run_session_backtest — real v_y is not on the CAN bus.

    Planar rigid-body kinematics: ay_meas ≈ vy_dot + vx*wz
        => vy_dot ≈ ay_meas - vx*wz
    The -vy/tau leak term prevents open-loop drift over a full session while
    preserving the transient dynamics needed for a physically correct vy0
    at the start of each 1.25s backtest window. Without this, every window
    currently starts at vy0=0.0 exactly, biasing ay/wz correlation across
    the whole first braking/turn-in phase of every window.
    """
    vy = np.zeros_like(ay)
    leak = dt / tau
    for i in range(1, len(ay)):
        vy_dot = ay[i - 1] - vx[i - 1] * wz[i - 1]
        vy[i] = vy[i - 1] + dt * vy_dot - leak * vy[i - 1]
    return vy

def decode_can_csv_to_dataframe(file_path: Path, dbc_path: Path, dt: float = 0.005, lag_samples: int = 14) -> pd.DataFrame:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_line = f.readline().strip()
    print(f"  [header] {file_path.name}: '{header_line}'")

    header_cols = [c.strip() for c in header_line.split(',')]

    _SIGNAL_MARKERS = {'ANGLE', 'Yaw_Rate_z', 'a_x', 'a_y', 'v_x', 'BPPS',
                        'rlTRQ', 'rrTRQ', 'leftDem', 'rightDem'}
    is_predecoded = len(_SIGNAL_MARKERS & set(header_cols)) >= 3

    if is_predecoded:
        print(f"  [format] {file_path.name}: pre-decoded wide CSV detected — "
              f"reading directly (manual csv reader), skipping DBC decode.")

        import csv
        n_cols = len(header_cols)
        rows = []
        n_bad = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < n_cols:
                    row = row + [''] * (n_cols - len(row))
                elif len(row) > n_cols:
                    row = row[:n_cols]
                    n_bad += 1
                rows.append(row)

        print(f"  [format] {file_path.name}: {len(rows)} rows read manually "
              f"({n_bad} truncated for extra fields)")

        raw_df = pd.DataFrame(rows, columns=header_cols)
        raw_df.columns = [c.strip() for c in raw_df.columns]
        raw_df = raw_df.loc[:, ~raw_df.columns.astype(str).str.match(r'^Unnamed')]
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
    else:
        raise ValueError(f"{file_path.name}: not a recognized pre-decoded format")

    df_out = _standardize_and_resample(raw_df, dt, lag_samples)

    if df_out['steer_deg'].std() < 1e-6 and df_out['yaw_rate_deg_s'].std() < 1e-6:
        print(f"  [WARN] {file_path.name}: steer AND yaw channels are flat "
              f"(std≈0) — check column mapping in rename_map.")

    return df_out


def _extract_1d(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32).ravel()


def _standardize_and_resample(df: pd.DataFrame, dt: float, lag_samples: int) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}
    t_col = cols_lower.get('timestamp') or cols_lower.get('time')

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

    if t_col is None:
        print(f"  [WARN] No time column found — resampling SKIPPED.")
    if t_col:
        df[t_col] = pd.to_numeric(df[t_col], errors='coerce')
        df = df.dropna(subset=[t_col]).sort_values(by=t_col)
        t_arr = df[t_col].to_numpy(dtype=np.float64)
        valid_idx = np.where(np.isfinite(t_arr))[0]
        if len(valid_idx) > 0:
            df = df.iloc[valid_idx].copy()
            t_rel = t_arr[valid_idx] - t_arr[valid_idx][0]
            if len(t_rel) > 1 and t_rel[-1] > 0:
                t_uniform = np.arange(0, t_rel[-1], dt)
                resampled_cols = {'t_rel': t_uniform}
                for c in df.columns:
                    if c in (t_col, 't_rel'):
                        continue
                    raw_vals = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=np.float64)
                    valid = np.isfinite(raw_vals)
                    if valid.sum() < 2:
                        resampled_cols[c] = np.zeros_like(t_uniform)
                        continue
                    t_valid = t_rel[valid]
                    v_valid = raw_vals[valid]
                    order = np.argsort(t_valid)
                    resampled_cols[c] = np.interp(t_uniform, t_valid[order], v_valid[order])
                df = pd.concat({k: pd.Series(v) for k, v in resampled_cols.items()}, axis=1)

    vx = _extract_1d(df, 'vx_mps')
    rl_rpm = _extract_1d(df, 'rlRPM')
    rr_rpm = _extract_1d(df, 'rrRPM')
    R_WHEEL = 0.2045

    vx_from_wheels = ((rl_rpm + rr_rpm) * 0.5) * (2.0 * np.pi / 60.0) * R_WHEEL

    if np.std(vx) < 2.0 and np.std(vx_from_wheels) > 2.0:
        print(f"  [fix] v_x channel appears GPS-starved (std={np.std(vx):.3f}) "
              f"— substituting wheel-speed-derived vx (std={np.std(vx_from_wheels):.3f})")
        vx = vx_from_wheels

    if np.max(np.abs(vx)) < 0.5 and 'speed_raw' in df.columns:
        vx = _extract_1d(df, 'speed_raw')
        if np.max(np.abs(vx)) > 50.0:
            vx = vx * (2.0 * np.pi / 60.0) * 0.2045
    df['vx_mps'] = vx

    # ── Reconstrucción cinemática de referencia (inmune a la saturación de 2g y al filtro de 1Hz)
    vx_arr = _extract_1d(df, 'vx_mps')
    wz_arr = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
    ay_kinematic = vx_arr * wz_arr

    # Roll físico real del chasis (K_roll = 2.0 deg/g) eliminando el artefacto de 50° del atan2
    K_ROLL = 2.0  # deg/g
    df['roll_deg'] = (ay_kinematic / 9.81) * K_ROLL

    # Si el sensor en el log está saturado/atenuado respecto a vx*wz, usamos la aceleración real del coche
    ay_raw = _extract_1d(df, 'ay_mps2')
    if np.std(ay_raw) > 0.01 and np.std(ay_raw) < 2.5:
        ay_raw = ay_raw * 9.81

    # Fusión: dinámica real centrípeta en curva (>1 m/s²), lectura cruda en recta
    ay_fused = np.where(np.abs(ay_kinematic) > 1.0, ay_kinematic, ay_raw)
    df['ay_mps2'] = ay_fused

    steer = _extract_1d(df, 'steer_deg')
    steer_ratio = VP_DICT.get('steer_ratio', 4.5)
    steer = steer / steer_ratio

    wz = _extract_1d(df, 'yaw_rate_deg_s')
    ay = _extract_1d(df, 'ay_mps2')
    corr_steer_wz = (np.corrcoef(steer, wz)[0, 1]
                      if np.std(steer) > 1e-3 and np.std(wz) > 1e-3 else float('nan'))
    corr_wz_ay = (np.corrcoef(wz, ay)[0, 1]
                  if np.std(wz) > 1e-3 and np.std(ay) > 1e-3 else float('nan'))
    print(f"  [chk] corr(steer, wz) = {corr_steer_wz:+.3f}   corr(wz, ay) = {corr_wz_ay:+.3f}")

    df['steer_deg'] = steer

    # Sincronización automática de fase por correlación cruzada (0 a 200 ms)
    if len(steer) > 200 and np.std(steer) > 0.05 and np.std(wz) > 0.05:
        max_shift = min(40, len(steer) // 10)
        lags = range(0, max_shift)
        corrs = [
            np.corrcoef(np.roll(steer, -s)[:len(steer) - max_shift], 
                        wz[:len(steer) - max_shift])[0, 1] 
            for s in lags
        ]
        best_lag = int(np.nanargmax(corrs))
        if np.isfinite(corrs[best_lag]) and corrs[best_lag] > 0.7:
            lag_samples = best_lag
            print(f"  [sync] Auto-aligned steer lag: {lag_samples} samples ({lag_samples * dt * 1000:.1f} ms) | r_peak={corrs[best_lag]:.3f}")

    if lag_samples > 0 and len(steer) > lag_samples:
        steer = np.roll(steer, -lag_samples)
        steer[-lag_samples:] = steer[-lag_samples - 1]
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
    # Zona muerta del 3% para anular el offset de reposo y ruidos residuales
    bpps_clean = np.where(bpps > 3.0, bpps - 3.0, 0.0)
    df['brake_press'] = bpps_clean * 10.0

    return df


@partial(jax.jit, static_argnums=(0, 3))
def _simulate_all_windows_jit(vehicle: DifferentiableMultiBodyVehicle, x0_batch: jax.Array,
                               u_batch: jax.Array, dt: float,
                               tire_cal: jax.Array = jnp.array([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32),
                               ay_scale: float = 1.0):
    setup = vehicle._default_setup_vec

    def sim_one_window(x0, u_seq):
        def step_fn(x, u):
            x_next = vehicle.simulate_step(x, u, setup, dt=dt, n_substeps=4, tire_cal=tire_cal)
            vx_n = x_next[14]; wz_n = x_next[19]
            ay_force = vx_n * wz_n
            return x_next, jnp.stack([vx_n, wz_n, ay_force])
        _, out = jax.lax.scan(step_fn, x0, u_seq)
        return out

    return jax.vmap(sim_one_window)(x0_batch, u_batch)


def run_session_backtest(vehicle, df, dt=0.005, steer_sign=1.0, verbose=True,
                          tire_cal: jax.Array = jnp.array([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32),
                          steer_gain: float = 1.0, brake_gain: float = 1.0,
                          torque_gain: float = 1.0, ay_scale: float = 1.0):
    N = len(df)
    n_windows = N // WINDOW_LEN
    if n_windows == 0:
        return {'duration_s': N * dt, 'windows': 0, 'rmse_vx': 0, 'rmse_wz': 0, 'rmse_ay': 0, 'r_wz': 0, 'r_ay': 0, 'score': 0.0}

    steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg')) * steer_sign * steer_gain
    t_fl = _extract_1d(df, 't_fl') * torque_gain
    t_fr = _extract_1d(df, 't_fr') * torque_gain
    t_rl = _extract_1d(df, 't_rl') * torque_gain
    t_rr = _extract_1d(df, 't_rr') * torque_gain
    p_hyd = _extract_1d(df, 'brake_press') * brake_gain

    u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)

    # NEW — same CAN-glitch guard as calibrate_mu_from_telemetry.py
    u_all = np.nan_to_num(u_all, nan=0.0, posinf=0.0, neginf=0.0)
    u_all[:, 1:5] = np.clip(u_all[:, 1:5], -50.0, 400.0)
    u_all[:, 5]   = np.clip(u_all[:, 5],   0.0,   2000.0)

    real_wz_all = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
    real_ay_all = _extract_1d(df, 'ay_mps2')
    real_vx_all = _extract_1d(df, 'vx_mps')
    real_vy_all = _estimate_vy_kinematic(real_ay_all, real_vx_all, real_wz_all, dt)  # was: np.zeros(N)

    u_windows, x0_windows, real_vx_wins, real_wz_wins, real_ay_wins = [], [], [], [], []

    for w in range(n_windows):
        start = w * WINDOW_LEN
        end = start + WINDOW_LEN
        u_windows.append(u_all[start:end])

        vx0 = float(max(real_vx_all[start], 1.0))
        wz0 = float(real_wz_all[start])
        k_drift = (300.0 * 0.8525) / (1.55 * 45000.0)  # C_alpha_r = 45 kN/rad
        vy0_refined = float(np.clip(-0.6975 * wz0 + k_drift * vx0 * wz0, -1.2, 1.2))

        x0 = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx0)
        x0 = x0.at[15].set(vy0_refined)
        x0 = x0.at[19].set(wz0)
        x0_windows.append(x0)

        real_vx_wins.append(real_vx_all[start:end])
        real_wz_wins.append(real_wz_all[start:end])
        real_ay_wins.append(real_ay_all[start:end])

    u_batch = jnp.asarray(np.stack(u_windows))
    x0_batch = jnp.asarray(np.stack(x0_windows))

    sim_out = _simulate_all_windows_jit(vehicle, x0_batch, u_batch, dt, tire_cal, ay_scale)
    sim_out_np = np.array(sim_out)

    per_window_ay_max = np.max(np.abs(sim_out_np[:, :, 2]), axis=1)
    n_blown = int(np.sum(per_window_ay_max > 15.0))
    if verbose:
        print(f"  [chk] steer_sign={steer_sign:+.0f}: {n_blown}/{len(per_window_ay_max)} "
              f"windows exceed 15 m/s² sim ay")

    sim_vx_wins = sim_out_np[:, :, 0]
    sim_wz_wins = sim_out_np[:, :, 1]
    sim_ay_wins = sim_out_np[:, :, 2]

    real_vx_arr = np.stack(real_vx_wins)
    real_wz_arr = np.stack(real_wz_wins)
    real_ay_arr = np.stack(real_ay_wins)

    sim_vx = sim_vx_wins.flatten()
    sim_wz = sim_wz_wins.flatten()
    sim_ay = sim_ay_wins.flatten()

    real_vx = real_vx_arr.flatten()
    real_wz = real_wz_arr.flatten()
    real_ay = real_ay_arr.flatten()

    rmse_vx = float(np.sqrt(np.mean((sim_vx - real_vx) ** 2)))
    rmse_wz = float(np.sqrt(np.mean((sim_wz - real_wz) ** 2)))
    rmse_ay = float(np.sqrt(np.mean((sim_ay - real_ay) ** 2)))

    sim_wz_c  = _lowpass_corr(sim_wz)
    real_wz_c = _lowpass_corr(real_wz)
    sim_ay_c  = _lowpass_corr(sim_ay)
    real_ay_c = _lowpass_corr(real_ay)

    MIN_DYN_SAMPLES = 100

    dyn_mask_wz = np.abs(real_wz_c) > 0.10
    dyn_mask_ay = np.abs(real_ay_c) > 1.5

    wz_valid = np.sum(dyn_mask_wz) >= MIN_DYN_SAMPLES
    ay_valid = np.sum(dyn_mask_ay) >= MIN_DYN_SAMPLES

    if wz_valid and np.std(sim_wz_c[dyn_mask_wz]) > 1e-3:
        r_wz = float(np.corrcoef(sim_wz_c[dyn_mask_wz], real_wz_c[dyn_mask_wz])[0, 1])
    else:
        r_wz = 0.0
        wz_valid = False

    if ay_valid and np.std(sim_ay_c[dyn_mask_ay]) > 1e-3:
        r_ay = float(np.corrcoef(sim_ay_c[dyn_mask_ay], real_ay_c[dyn_mask_ay])[0, 1])
    else:
        r_ay = 0.0
        ay_valid = False

    r_wz = float(np.nan_to_num(r_wz, nan=0.0))
    r_ay = float(np.nan_to_num(r_ay, nan=0.0))

    corr_score = (max(0.0, r_wz) * 0.45 + max(0.0, r_ay) * 0.40 + max(0.0, 1.0 - rmse_vx / 3.0) * 0.15) * 100.0

    return {
        'duration_s': N * dt,
        'windows': n_windows,
        'rmse_vx': rmse_vx,
        'rmse_wz': rmse_wz,
        'rmse_ay': rmse_ay,
        'r_wz': r_wz,
        'r_ay': r_ay,
        'score': corr_score,
        'n_blown': n_blown,
        'wz_valid': wz_valid,
        'ay_valid': ay_valid,
    }


def _probe_best_steer_sign(vehicle, df, dt, n_probe_windows=20):
    """
    Empirically determine steering sign convention. We measured the raw
    telemetry is internally self-consistent (corr(steer,wz)=+0.95), but that
    does NOT tell us which sign matches vehicle_dynamics.py's internal
    convention — that requires tracing Mz_total's full sign chain, which we
    have not done. Test both signs on a small window subset against the
    actual physics engine and keep whichever minimizes divergence/RMSE.
    """
    best_sign, best_penalty, best_res = 1.0, float('inf'), None
    n_rows = min(n_probe_windows * WINDOW_LEN, len(df))
    df_probe = df.iloc[:n_rows]
    for sign in (1.0, -1.0):
        res = run_session_backtest(vehicle, df_probe, dt=dt, steer_sign=sign, verbose=True)
        penalty = res['rmse_ay'] + res['rmse_wz'] * 10.0 + res['n_blown'] * 5.0
        print(f"  [probe] steer_sign={sign:+.0f}: rmse_ay={res['rmse_ay']:.2f} "
              f"rmse_wz={res['rmse_wz']:.3f} n_blown={res['n_blown']} "
              f"score={res['score']:.1f}% penalty={penalty:.2f}")
        if penalty < best_penalty:
            best_penalty, best_sign, best_res = penalty, sign, res
    print(f"  [probe] SELECTED steer_sign={best_sign:+.0f}")
    return best_sign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_can_logs"))
    parser.add_argument("--dbc", type=Path, default=Path("TER.dbc"))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--lag-samples", type=int, default=14)
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip loading calibrated tire_cal/gains, use identity defaults")
    args = parser.parse_args()

    files = sorted(list(args.data_dir.glob("*.csv")))
    if not files:
        print(f"[!] No CSV logs found in {args.data_dir}")
        return

    print(f"[*] Initializing Project-GP 108-DOF Engine for {len(files)} session log(s)...")
    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)

    # ── Load calibrated tire_cal + gains, if present ─────────────────────────
    tire_cal = jnp.array([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32)
    steer_gain, brake_gain, torque_gain = 1.0, 1.0, 1.0
    steer_sign_cal = 1.0

    ay_scale_cal = 1.0
    if not args.no_calibration:
        mu_path   = os.path.join("models", "mu_scale_calibrated.npy")
        gain_path = os.path.join("models", "gain_calibrated.npy")
        sign_path = os.path.join("models", "steer_sign_calibrated.npy")

        ay_path = os.path.join("models", "ay_scale_calibrated.npy")
        if os.path.exists(ay_path):
            ay_scale_cal = float(np.load(ay_path)[0])
            print(f"[*] Loaded calibrated ay_scale: {ay_scale_cal:.3f}")

        if os.path.exists(mu_path):
            mu = np.load(mu_path)
            tire_cal = jnp.array([mu[0], mu[1], -1.0, 1.0], dtype=jnp.float32)
            print(f"[*] Loaded calibrated tire_cal: mu_f={mu[0]:.3f} mu_r={mu[1]:.3f}")
        else:
            print(f"[*] No calibration found at {mu_path} — using identity tire_cal.")

        if os.path.exists(gain_path):
            g = np.load(gain_path)
            steer_gain, brake_gain, torque_gain = float(g[0]), float(g[1]), float(g[2])
            print(f"[*] Loaded calibrated gains: steer={steer_gain:.3f} "
                  f"brake={brake_gain:.3f} torque={torque_gain:.3f}")

        if os.path.exists(sign_path):
            steer_sign_cal = float(np.load(sign_path)[0])
            print(f"[*] Loaded calibrated steer_sign: {steer_sign_cal:+.0f}")

    print("\n" + "=" * 89)
    print(f"{'Session File':<16} | {'Dur(s)':<7} | {'RMSE Vx':<9} | {'RMSE Yaw':<10} | {'RMSE Ay':<9} | {'Corr %':<7} | {'Sign':<5} | {'Phase r(Ay)'}")
    print("-" * 89)

    scores = []
    for i, f in enumerate(files, 1):
        df = decode_can_csv_to_dataframe(f, dbc_path=args.dbc, dt=args.dt, lag_samples=args.lag_samples)

        steer_sign = steer_sign_cal

        res = run_session_backtest(vehicle, df, dt=args.dt, steer_sign=steer_sign, verbose=False,
                                tire_cal=tire_cal, steer_gain=steer_gain,
                                brake_gain=brake_gain, torque_gain=torque_gain,
                                ay_scale=ay_scale_cal)
        scores.append(res['score'])
        print(f"{f.name:<16} | {res['duration_s']:<7.1f} | {res['rmse_vx']:<9.3f} | {res['rmse_wz']:<10.3f} | {res['rmse_ay']:<9.3f} | {res['score']:<6.1f}% | {steer_sign:<5.0f} | {res['r_ay']:+.3f}"
              + ("" if res.get('wz_valid', True) and res.get('ay_valid', True) else "  [LOW-DYN]"))

    print("=" * 89)
    print(f"\n[*] Overall Mean Fleet Correlation Score: {np.mean(scores):.2f} %\n")


if __name__ == "__main__":
    main()