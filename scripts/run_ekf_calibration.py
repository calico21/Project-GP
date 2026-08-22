#!/usr/bin/env python3
# scripts/run_ekf_calibration.py
# Project-GP — Online EKF Grip/Thermal Adaptation Against Real Telemetry
# ═══════════════════════════════════════════════════════════════════════════════
# Drives DifferentiableEKF.update() step-by-step over real CAN telemetry,
# seeded from the offline-calibrated mu_scale (scripts/calibrate_mu_from_telemetry.py)
# rather than the flat 1.0 prior. Logs theta_hat convergence and, at the end,
# re-runs the standard windowed backtest with BOTH the static-calibrated
# tire_cal and the EKF-converged tire_cal so you can directly compare
# correlation before/after online adaptation.
#
# USAGE:
#   python -m scripts.run_ekf_calibration --data-dir data/raw_can_logs
#   python -m scripts.run_ekf_calibration --session data/raw_can_logs/2.csv --plot
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from models.differentiable_ekf import DifferentiableEKF, IDX_LAMBDA_MU_F, IDX_LAMBDA_MU_R
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import (
    decode_can_csv_to_dataframe, _extract_1d, _probe_best_steer_sign,
    run_session_backtest, WINDOW_LEN,
)

EKF_DECIMATE = 5   # run the EKF update every 5th physics step (25ms @ 200Hz) —
                    # jax.jacobian(theta -> 2) is cheap but running it at the
                    # full 5ms rate is unnecessary; sensor noise dominates
                    # below 40Hz anyway.


# ─────────────────────────────────────────────────────────────────────────────
# §1  Per-session EKF pass
# ─────────────────────────────────────────────────────────────────────────────

def run_ekf_over_session(
    ekf: DifferentiableEKF,
    vehicle: DifferentiableMultiBodyVehicle,
    df,
    dt: float,
    steer_sign: float,
    max_steps: int | None = None,
    log_every: int = 200,
) -> dict:
    """
    Steps the EKF forward through one decoded session.

    Design: the "true" state trajectory used for linearisation is advanced
    with vehicle.simulate_step() directly (full 108-DOF forward pass, using
    the CURRENT best tire_cal from theta_hat), while ekf.update() separately
    computes the Jacobian/innovation against real ay/wz at each decimated
    step. This mirrors how the EKF would run onboard: the physics model
    advances the state estimate every tick; the filter corrects theta_hat
    (and, in a full implementation, the state covariance) whenever a new
    measurement arrives.
    """
    N = len(df) if max_steps is None else min(max_steps, len(df))

    steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg')) * steer_sign
    t_fl, t_fr = _extract_1d(df, 't_fl'), _extract_1d(df, 't_fr')
    t_rl, t_rr = _extract_1d(df, 't_rl'), _extract_1d(df, 't_rr')
    p_hyd = _extract_1d(df, 'brake_press')

    R_wheel = vehicle.R_wheel
    net_force = ((t_fl + t_fr + t_rl + t_rr) / R_wheel) - (p_hyd / 4.0)

    real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
    real_ay = _extract_1d(df, 'ay_mps2')
    real_vx = _extract_1d(df, 'vx_mps')

    x_state = DifferentiableMultiBodyVehicle.make_initial_state(
        T_env=25.0, vx0=float(max(real_vx[0], 1.0)))
    x_state = x_state.at[19].set(float(real_wz[0]))
    setup = vehicle._default_setup_vec

    theta_hist = np.zeros((N // EKF_DECIMATE + 2, 5), dtype=np.float32)
    step_idx   = np.zeros(N // EKF_DECIMATE + 2, dtype=np.int32)
    h_ptr = 0

    print(f"    [EKF] Running {N} steps ({N*dt:.1f}s), "
          f"update every {EKF_DECIMATE} steps...")

    for i in range(N - 1):
        u_i = jnp.array([float(steer_rad[i]), float(net_force[i])])

        if i % EKF_DECIMATE == 0:
            # Correct theta_hat against the real measurement at this instant,
            # linearised around the current state x_state.
            theta_new, P_new = ekf.update(
                x_state, u_i, setup,
                ay_measured=float(real_ay[i]),
                wz_measured=float(real_wz[i]),
                dt=dt,
            )
            theta_hist[h_ptr] = np.array(theta_new)
            step_idx[h_ptr]   = i
            h_ptr += 1

            if i % log_every == 0:
                p = ekf.get_calibrated_params()
                print(f"      step {i:6d}/{N}  "
                      f"mu_f={p['lambda_mu_front']:.3f}  mu_r={p['lambda_mu_rear']:.3f}  "
                      f"T_opt={p['T_opt_estimated']:.1f}°C  "
                      f"h_cg={p['h_cg_estimated']*1000:.0f}mm  "
                      f"alpha_peak={p['alpha_peak_estimated_deg']:.2f}°")

        # Advance the true state one physics step using the LATEST theta_hat
        # (this is what "online" means: the plant model tightens itself as
        # the estimator converges, rather than running open-loop on the prior).
        lambda_mu_f = ekf.theta_hat[IDX_LAMBDA_MU_F]
        lambda_mu_r = ekf.theta_hat[IDX_LAMBDA_MU_R]
        t_opt       = ekf.theta_hat[2]
        alpha_peak  = ekf.theta_hat[4]
        b_scale     = 0.13 / jnp.clip(alpha_peak, 0.05, 0.30)
        tire_cal    = jnp.array([lambda_mu_f, lambda_mu_r, t_opt, b_scale], dtype=jnp.float32)

        steer, force = u_i[0], u_i[1]
        T_each  = jax.nn.relu(force) * R_wheel / 4.0
        F_brake = jax.nn.relu(-force)
        u_full  = jnp.array([steer, T_each, T_each, T_each, T_each, F_brake])

        x_state = vehicle.simulate_step(
            x_state, u_full, setup, dt=dt, n_substeps=1, tire_cal=tire_cal)

    return {
        'theta_hist':  theta_hist[:h_ptr],
        'step_idx':    step_idx[:h_ptr],
        'theta_final': np.array(ekf.theta_hat),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §2  Before/after correlation comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_static_vs_ekf(
    vehicle, df, dt, steer_sign, static_tire_cal, ekf_tire_cal,
) -> None:
    print(f"\n  ── Correlation comparison: STATIC calib vs EKF-adapted ──")
    res_static = run_session_backtest(vehicle, df, dt=dt, steer_sign=steer_sign,
                                       verbose=False, tire_cal=static_tire_cal)
    res_ekf    = run_session_backtest(vehicle, df, dt=dt, steer_sign=steer_sign,
                                       verbose=False, tire_cal=ekf_tire_cal)

    print(f"  {'Metric':<14} {'Static':>10} {'EKF-adapted':>14} {'Δ':>8}")
    print(f"  {'-'*50}")
    for key, label in [('score', 'Corr score %'), ('rmse_wz', 'RMSE wz'),
                        ('rmse_ay', 'RMSE ay'), ('r_wz', 'r(wz)'), ('r_ay', 'r(ay)')]:
        v0, v1 = res_static[key], res_ekf[key]
        print(f"  {label:<14} {v0:>10.3f} {v1:>14.3f} {v1-v0:>+8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# §3  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw_can_logs"))
    ap.add_argument("--session",  type=Path, default=None,
                     help="Run on a single session instead of the whole dir")
    ap.add_argument("--dbc",      type=Path, default=Path("TER.dbc"))
    ap.add_argument("--dt",       type=float, default=0.005)
    ap.add_argument("--max-steps", type=int, default=None,
                     help="Cap steps per session (for quick smoke tests)")
    args = ap.parse_args()

    files = [args.session] if args.session else sorted(args.data_dir.glob("*.csv"))
    if not files:
        print(f"[!] No CSV logs found.")
        return

    print("[*] Initialising Project-GP 108-DOF Engine...")
    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)

    # ── Seed theta_hat from the offline-calibrated mu_scale, if present ─────
    mu_path = os.path.join("models", "mu_scale_calibrated.npy")
    ekf = DifferentiableEKF(vehicle)
    static_tire_cal = jnp.array([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32)
    if os.path.exists(mu_path):
        mu_static = np.load(mu_path)
        ekf.theta_hat = ekf.theta_hat.at[IDX_LAMBDA_MU_F].set(float(mu_static[0]))
        ekf.theta_hat = ekf.theta_hat.at[IDX_LAMBDA_MU_R].set(float(mu_static[1]))
        static_tire_cal = jnp.array([mu_static[0], mu_static[1], -1.0, 1.0], dtype=jnp.float32)
        print(f"[*] Seeded EKF prior from offline calibration: "
              f"mu_f={mu_static[0]:.3f}  mu_r={mu_static[1]:.3f}")
    else:
        print("[*] No offline calibration found — EKF starts from flat 1.0/1.0 prior. "
              "Run scripts/calibrate_mu_from_telemetry.py first for a better seed.")

    print(f"\n{'='*80}")
    for f in files:
        print(f"\n[Session] {f.name}")
        df = decode_can_csv_to_dataframe(f, dbc_path=args.dbc, dt=args.dt)
        steer_sign = _probe_best_steer_sign(vehicle, df, dt=args.dt)

        result = run_ekf_over_session(
            ekf, vehicle, df, dt=args.dt, steer_sign=steer_sign,
            max_steps=args.max_steps)

        theta_f = result['theta_final']
        print(f"  [EKF] Converged theta_hat: mu_f={theta_f[0]:.3f}  mu_r={theta_f[1]:.3f}  "
              f"T_opt={theta_f[2]:.1f}°C  h_cg={theta_f[3]*1000:.0f}mm  "
              f"alpha_peak={np.degrees(theta_f[4]):.2f}°")

        # Build tire_cal from the converged theta for a direct correlation compare
        b_scale = 0.13 / np.clip(theta_f[4], 0.05, 0.30)
        ekf_tire_cal = jnp.array([theta_f[0], theta_f[1], theta_f[2], b_scale],
                                  dtype=jnp.float32)

        compare_static_vs_ekf(vehicle, df, args.dt, steer_sign,
                               static_tire_cal, ekf_tire_cal)

        # Save per-session convergence history for offline plotting
        out_dir = Path("reports") / "ekf_calibration"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / f"{f.stem}_ekf_history.npz",
                  theta_hist=result['theta_hist'], step_idx=result['step_idx'])
        print(f"  [Saved] {out_dir / (f.stem + '_ekf_history.npz')}")

    print(f"\n{'='*80}")
    print("[*] EKF calibration pass complete.")


if __name__ == "__main__":
    main()