#!/usr/bin/env python3
# scripts/calibrate_mu_from_telemetry.py
# Project-GP — Gradient-Based Digital-Twin Calibration Against Real CAN Telemetry
# ═══════════════════════════════════════════════════════════════════════════════
# Fits [mu_f, mu_r, steer_gain, brake_gain, torque_gain] directly against real
# wz/ay telemetry via jax.grad through the full GLRK-4 physics stack.
#
# WHY CORRELATION, NOT RMSE:
#   RMSE is dominated by the long straight-line segments where wz≈ay≈0 —
#   the optimizer gets almost no gradient signal about grip from those.
#   1 - Pearson_r isolates whether the SHAPE of the response matches, which
#   is exactly what a correlation-based fidelity score measures, and gives
#   a strong, well-conditioned gradient in the corners where it matters.
#
# USAGE:
#   python -m scripts.calibrate_mu_from_telemetry --data-dir data/raw_can_logs
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import (
    decode_can_csv_to_dataframe, _extract_1d, WINDOW_LEN, _estimate_vy_kinematic,
)

N_CAL_WINDOWS = 40   # subsample per session — a 5-param fit doesn't need the full log


# ─────────────────────────────────────────────────────────────────────────────
# §1  Batch construction (mirrors run_session_backtest's windowing exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _build_calib_batch(df, dt: float, steer_sign: float, rng: np.random.Generator):
    N = len(df)
    n_windows = N // WINDOW_LEN
    idx = rng.choice(n_windows, size=min(N_CAL_WINDOWS, n_windows), replace=False)

    steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg')) * steer_sign
    t_fl, t_fr = _extract_1d(df, 't_fl'), _extract_1d(df, 't_fr')
    t_rl, t_rr = _extract_1d(df, 't_rl'), _extract_1d(df, 't_rr')
    p_hyd = _extract_1d(df, 'brake_press')
    u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)

    real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
    real_ay = _extract_1d(df, 'ay_mps2')
    real_vx = _extract_1d(df, 'vx_mps')
    vy_est  = _estimate_vy_kinematic(real_ay, real_vx, real_wz, dt)

    u_win, x0_win, wz_win, ay_win = [], [], [], []
    for w in idx:
        s, e = w * WINDOW_LEN, w * WINDOW_LEN + WINDOW_LEN
        u_win.append(u_all[s:e])
        x0 = DifferentiableMultiBodyVehicle.make_initial_state(
            T_env=25.0, vx0=float(max(real_vx[s], 1.0)))
        x0 = x0.at[15].set(float(vy_est[s])).at[19].set(float(real_wz[s]))
        x0_win.append(x0)
        wz_win.append(real_wz[s:e])
        ay_win.append(real_ay[s:e])

    return (jnp.asarray(np.stack(u_win),  dtype=jnp.float32),
            jnp.asarray(np.stack(x0_win), dtype=jnp.float32),
            jnp.asarray(np.stack(wz_win), dtype=jnp.float32),
            jnp.asarray(np.stack(ay_win), dtype=jnp.float32))


# ─────────────────────────────────────────────────────────────────────────────
# §2  Correlation loss
# ─────────────────────────────────────────────────────────────────────────────

def _soft_corr_loss(sim: jax.Array, real: jax.Array) -> jax.Array:
    """1 - Pearson r, averaged over the window batch. Scale-invariant,
    differentiable, and directly targets the fidelity metric being reported."""
    sim_c  = sim  - jnp.mean(sim,  axis=1, keepdims=True)
    real_c = real - jnp.mean(real, axis=1, keepdims=True)
    num = jnp.sum(sim_c * real_c, axis=1)
    den = jnp.sqrt(jnp.sum(sim_c ** 2, axis=1) * jnp.sum(real_c ** 2, axis=1)) + 1e-6
    r = num / den
    return jnp.mean(1.0 - r)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Calibration loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw_can_logs"))
    ap.add_argument("--dbc",      type=Path, default=Path("TER.dbc"))
    ap.add_argument("--dt",       type=float, default=0.005)
    ap.add_argument("--steps",    type=int,   default=200)
    ap.add_argument("--lr",       type=float, default=0.05)
    ap.add_argument("--steer-sign", type=float, default=1.0,
                     help="Use the sign already selected by _probe_best_steer_sign "
                          "in run_can_backtest.py for this fleet.")
    args = ap.parse_args()

    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
    setup   = vehicle._default_setup_vec
    rng     = np.random.default_rng(0)

    files = sorted(args.data_dir.glob("*.csv"))
    if not files:
        print(f"[!] No CSV logs found in {args.data_dir}")
        return

    print(f"[Calibrate] Building calibration batches from {len(files)} session(s)...")
    batches = []
    for f in files:
        df = decode_can_csv_to_dataframe(f, dbc_path=args.dbc, dt=args.dt)
        batches.append(_build_calib_batch(df, args.dt, args.steer_sign, rng))
    n_win_total = sum(b[0].shape[0] for b in batches)
    print(f"[Calibrate] {len(batches)} session(s), {n_win_total} windows total "
          f"({n_win_total * WINDOW_LEN * args.dt:.0f}s of telemetry)")

    # ── Rollout: params -> (wz_hist, ay_hist) per window ────────────────────
    def rollout(mu_scale, steer_gain, brake_gain, torque_gain, u_seq, x0):
        u_scaled = u_seq.at[:, 0].multiply(steer_gain)        # steer channel
        u_scaled = u_scaled.at[:, 1:5].multiply(torque_gain)  # 4 hub torques
        u_scaled = u_scaled.at[:, 5].multiply(brake_gain)     # hydraulic brake

        def step_fn(x, u):
            x_next = vehicle.simulate_step(x, u, setup, dt=args.dt,
                                            n_substeps=2, mu_scale=mu_scale)
            return x_next, jnp.array([x_next[19], x_next[14] * x_next[19]])

        _, out = jax.lax.scan(step_fn, x0, u_scaled)
        return out[:, 0], out[:, 1]   # wz_hist, ay_hist

    v_rollout = jax.vmap(rollout, in_axes=(None, None, None, None, 0, 0))

    def loss_fn(theta, u_seq, x0, wz_real, ay_real):
        mu_scale    = jnp.exp(theta[0:2])   # [mu_f, mu_r]   — positivity via log-param
        steer_gain  = jnp.exp(theta[2])
        brake_gain  = jnp.exp(theta[3])
        torque_gain = jnp.exp(theta[4])
        wz_sim, ay_sim = v_rollout(mu_scale, steer_gain, brake_gain, torque_gain, u_seq, x0)
        return 0.6 * _soft_corr_loss(wz_sim, wz_real) + 0.4 * _soft_corr_loss(ay_sim, ay_real)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # theta = log([mu_f, mu_r, steer_gain, brake_gain, torque_gain]), all start at 1.0
    theta = jnp.zeros(5)
    theta_lb = jnp.log(jnp.array([0.5, 0.5, 0.6, 0.3, 0.3]))
    theta_ub = jnp.log(jnp.array([1.8, 1.8, 1.6, 3.0, 3.0]))

    opt = optax.adam(args.lr)
    opt_state = opt.init(theta)

    print("[Calibrate] Fitting [mu_f, mu_r, steer_gain, brake_gain, torque_gain] "
          "via jax.grad + Adam ...")
    for step in range(args.steps):
        u_seq, x0, wz_real, ay_real = batches[step % len(batches)]
        loss, g = grad_fn(theta, u_seq, x0, wz_real, ay_real)
        if not bool(jnp.all(jnp.isfinite(g))):
            print(f"  step {step:4d}  NaN gradient — skipping update")
            continue
        updates, opt_state = opt.update(g, opt_state, theta)
        theta = jnp.clip(optax.apply_updates(theta, updates), theta_lb, theta_ub)

        if step % 10 == 0:
            p = jnp.exp(theta)
            print(f"  step {step:4d}  loss={float(loss):.4f}  "
                  f"mu_f={p[0]:.3f} mu_r={p[1]:.3f} "
                  f"steer={p[2]:.3f} brake={p[3]:.3f} torque={p[4]:.3f}")

    p_final = np.array(jnp.exp(theta))
    print(f"\n[Calibrate] DONE.")
    print(f"  mu_scale_f    = {p_final[0]:.4f}")
    print(f"  mu_scale_r    = {p_final[1]:.4f}")
    print(f"  steer_gain    = {p_final[2]:.4f}  (correction on ANGLE/steer_ratio)")
    print(f"  brake_gain    = {p_final[3]:.4f}  (correction on BPPS->N scale, was fixed *10.0)")
    print(f"  torque_gain   = {p_final[4]:.4f}  (correction on APPS->Nm scale, was fixed *150)")

    os.makedirs("models", exist_ok=True)
    np.save("models/mu_scale_calibrated.npy", p_final[0:2])
    np.save("models/gain_calibrated.npy", p_final[2:5])
    print("\n[Calibrate] Saved -> models/mu_scale_calibrated.npy, models/gain_calibrated.npy")
    print("[Calibrate] run_can_backtest.py will auto-load mu_scale_calibrated.npy if present.")


if __name__ == "__main__":
    main()