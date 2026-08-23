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
    _probe_best_steer_sign,                                   # NEW import

)

N_CAL_WINDOWS = 15   # subsample per session — a 5-param fit doesn't need the full log

jax.config.update("jax_debug_nans", True)

# ─────────────────────────────────────────────────────────────────────────────
# §1  Batch construction (mirrors run_session_backtest's windowing exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _build_calib_batch(df, dt: float, steer_sign: float, rng: np.random.Generator):
    N = len(df)
    n_windows = N // WINDOW_LEN

    real_vx = _extract_1d(df, 'vx_mps')   # single source of truth

    MIN_VX0 = 3.0
    candidate_windows = np.array([
        w for w in range(n_windows)
        if real_vx[w * WINDOW_LEN] >= MIN_VX0
    ])
    if len(candidate_windows) == 0:
        print(f"  [WARN] no windows with vx0 >= {MIN_VX0} m/s — falling back to all windows")
        candidate_windows = np.arange(n_windows)

    idx = rng.choice(candidate_windows,
                      size=min(N_CAL_WINDOWS, len(candidate_windows)),
                      replace=False)

    steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg')) * steer_sign
    t_fl, t_fr = _extract_1d(df, 't_fl'), _extract_1d(df, 't_fr')
    t_rl, t_rr = _extract_1d(df, 't_rl'), _extract_1d(df, 't_rr')
    p_hyd = _extract_1d(df, 'brake_press')
    u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)

    u_all = np.nan_to_num(u_all, nan=0.0, posinf=0.0, neginf=0.0)
    u_all[:, 1:5] = np.clip(u_all[:, 1:5], -50.0, 400.0)
    u_all[:, 5]   = np.clip(u_all[:, 5],   0.0,   2000.0)

    real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
    real_ay = _extract_1d(df, 'ay_mps2')
    vy_est  = _estimate_vy_kinematic(real_ay, real_vx, real_wz, dt)

    u_win, x0_win, wz_win, ay_win = [], [], [], []
    for w in idx:
        s, e = w * WINDOW_LEN, w * WINDOW_LEN + WINDOW_LEN
        u_win.append(u_all[s:e])
        x0 = DifferentiableMultiBodyVehicle.make_initial_state(
            T_env=25.0, vx0=float(max(real_vx[s], MIN_VX0)))
        vy0_clipped = float(np.clip(vy_est[s], -15.0, 15.0))
        x0 = x0.at[15].set(vy0_clipped).at[19].set(float(real_wz[s]))
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
    """1 - Pearson r, averaged over the window batch. Windows where the real
    signal has near-zero variance (straight-line segments) are excluded from
    both the loss value AND the gradient — Pearson r is mathematically
    undefined there, and even though num/den ~ 0/eps is forward-finite, the
    backward pass through that division has a true 0/0 singularity as the
    variance -> 0, which poisons the whole batch via jnp.mean (a single NaN
    anywhere in the batch NaNs the mean)."""
    sim_c  = sim  - jnp.mean(sim,  axis=1, keepdims=True)
    real_c = real - jnp.mean(real, axis=1, keepdims=True)

    var_sim  = jnp.sum(sim_c ** 2,  axis=1)
    var_real = jnp.sum(real_c ** 2, axis=1)

    # Variance floor: below this, the window carries no correlation signal.
    # Units: (rad/s)^2 for wz, (m/s^2)^2 for ay — 1e-3 is well below sensor
    # noise floor variance over a 250-sample (1.25s) window for either channel.
    VAR_FLOOR = 1e-3

    # Regularized denominator: floor keeps it bounded away from 0, so the
    # gradient d(num/den)/d(...) stays finite everywhere (no true 0/0).
    den = jnp.sqrt(jnp.maximum(var_sim * var_real, VAR_FLOOR ** 2))
    num = jnp.sum(sim_c * real_c, axis=1)
    r   = num / den

    # Per-window validity mask: only windows with enough real-signal variance
    # contribute to the loss. stop_gradient on the mask itself (it's a boolean
    # selection, not something we differentiate through).
    valid = jax.lax.stop_gradient((var_real > VAR_FLOOR) & (var_sim > VAR_FLOOR))
    valid_f = valid.astype(r.dtype)

    per_window_loss = (1.0 - r) * valid_f
    n_valid = jnp.maximum(jnp.sum(valid_f), 1.0)
    return jnp.sum(per_window_loss) / n_valid


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
                     help="Initial guess only — auto-overridden by an empirical "
                          "probe against files[0] before calibration begins.")
    args = ap.parse_args()

    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
    setup   = vehicle._default_setup_vec
    rng     = np.random.default_rng(0)

    files = sorted(args.data_dir.glob("*.csv"))
    if not files:
        print(f"[!] No CSV logs found in {args.data_dir}")
        return

    # NEW — empirically probe steer sign against the real physics engine,
    # exactly as run_can_backtest.py does, instead of trusting the --steer-sign
    # CLI default (1.0) blind. A wrong sign here silently drives mu_f/mu_r to
    # their optimizer bounds trying to compensate for a Mz sign mismatch.
    print("[Calibrate] Probing steer sign against physics engine...")
    df_probe     = decode_can_csv_to_dataframe(files[0], dbc_path=args.dbc, dt=args.dt)
    probed_sign  = _probe_best_steer_sign(vehicle, df_probe, dt=args.dt)
    if probed_sign != args.steer_sign:
        print(f"[Calibrate] Overriding --steer-sign={args.steer_sign:+.0f} "
              f"-> probed={probed_sign:+.0f}")
        args.steer_sign = probed_sign

    print(f"[Calibrate] Building calibration batches from {len(files)} session(s)...")
    batches = []
    for f in files:
        df = decode_can_csv_to_dataframe(f, dbc_path=args.dbc, dt=args.dt)
        batches.append(_build_calib_batch(df, args.dt, args.steer_sign, rng))
    n_win_total = sum(b[0].shape[0] for b in batches)
    print(f"[Calibrate] {len(batches)} session(s), {n_win_total} windows total "
          f"({n_win_total * WINDOW_LEN * args.dt:.0f}s of telemetry)")

    # NEW — diagnostic: how many windows actually carry correlation signal?
    for i, (u_seq, x0, wz_real, ay_real) in enumerate(batches):
        var_wz = jnp.var(wz_real, axis=1)
        var_ay = jnp.var(ay_real, axis=1)
        n_valid = int(jnp.sum((var_wz > 1e-3) & (var_ay > 1e-3)))
        print(f"  batch {i}: {n_valid}/{u_seq.shape[0]} windows have signal "
              f"(var_wz range [{float(var_wz.min()):.2e},{float(var_wz.max()):.2e}])")

    # ── Rollout: params -> (wz_hist, ay_hist) per window ────────────────────
    def rollout(mu_scale, steer_gain, brake_gain, torque_gain, u_seq, x0):
        u_scaled = u_seq.at[:, 0].multiply(steer_gain)
        u_scaled = u_scaled.at[:, 1:5].multiply(torque_gain)
        u_scaled = u_scaled.at[:, 5].multiply(brake_gain)

        tire_cal = jnp.array([mu_scale[0], mu_scale[1], -1.0, 1.0], dtype=jnp.float32)

        def step_fn(x, u):
            x_next = vehicle.simulate_step(x, u, setup, dt=args.dt,
                                            n_substeps=2, tire_cal=tire_cal)
            # DEBUG: also emit vx, vy, wz, and the raw derivative at this state
            return x_next, jnp.array([x_next[19], x_next[14] * x_next[19],
                                        x_next[14], x_next[15]])  # wz, ay, vx, vy

        _, out = jax.lax.scan(step_fn, x0, u_scaled)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    v_rollout = jax.vmap(rollout, in_axes=(None, None, None, None, 0, 0))

    def loss_fn(theta, u_seq, x0, wz_real, ay_real):
        mu_scale    = jnp.exp(theta[0:2])   # [mu_f, mu_r]   — positivity via log-param
        steer_gain  = jnp.exp(theta[2])
        brake_gain  = jnp.exp(theta[3])
        torque_gain = jnp.exp(theta[4])
        wz_sim, ay_sim, _vx_sim, _vy_sim = v_rollout(
            mu_scale, steer_gain, brake_gain, torque_gain, u_seq, x0)
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

        # NEW — fail fast with a diagnosable batch index instead of a bare
        # FloatingPointError from inside lax.scan.
        for name, arr in [("u_seq", u_seq), ("x0", x0),
                           ("wz_real", wz_real), ("ay_real", ay_real)]:
            if not bool(jnp.all(jnp.isfinite(arr))):
                raise ValueError(
                    f"[Calibrate] Non-finite values in {name} "
                    f"(batch {step % len(batches)})"
                )

        loss, g = grad_fn(theta, u_seq, x0, wz_real, ay_real)
        g_clean = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if not bool(jnp.all(jnp.isfinite(g))):
            n_bad = int(jnp.sum(~jnp.isfinite(g)))
            print(f"  step {step:4d}  {n_bad}/5 NaN grad components — zeroed and continuing")
        updates, opt_state = opt.update(g_clean, opt_state, theta)
        theta = jnp.clip(optax.apply_updates(theta, updates), theta_lb, theta_ub)

        if step % 10 == 0:
            p = jnp.exp(theta)
            print(f"  step {step:4d}  loss={float(loss):.4f}  "
                  f"mu_f={p[0]:.3f} mu_r={p[1]:.3f} "
                  f"steer={p[2]:.3f} brake={p[3]:.3f} torque={p[4]:.3f}")

    # NEW — dump one window's sim-vs-real trace for visual/manual inspection
    os.makedirs("reports", exist_ok=True)
    u_seq, x0, wz_real, ay_real = batches[0]
    mu_scale    = jnp.exp(theta[0:2])
    steer_gain  = jnp.exp(theta[2])
    brake_gain  = jnp.exp(theta[3])
    torque_gain = jnp.exp(theta[4])
    wz_sim, ay_sim, vx_sim, vy_sim = v_rollout(
        mu_scale, steer_gain, brake_gain, torque_gain, u_seq, x0)
    np.savez("reports/calib_window0_debug.npz",
             wz_sim=np.array(wz_sim), wz_real=np.array(wz_real),
             ay_sim=np.array(ay_sim), ay_real=np.array(ay_real),
             vx_sim=np.array(vx_sim), vy_sim=np.array(vy_sim))
    print(f"  window 0 u_seq[0,:5,0] (steer_rad, pre-gain): {u_seq[0,:5,0]}")
    print(f"  window 0 sample: wz_real[0,:5]={wz_real[0,:5]}  wz_sim[0,:5]={wz_sim[0,:5]}")
    print(f"  window 0 sample: ay_real[0,:5]={ay_real[0,:5]}  ay_sim[0,:5]={ay_sim[0,:5]}")
    print(f"  window 0 sample: vx_sim[0,:5]={vx_sim[0,:5]}  vy_sim[0,:5]={vy_sim[0,:5]}")

    p_final = np.array(jnp.exp(theta))
    print(f"\n[Calibrate] DONE.")


if __name__ == "__main__":
    main()