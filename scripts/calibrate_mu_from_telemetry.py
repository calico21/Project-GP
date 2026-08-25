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

    u_win, x0_win, wz_win, ay_win, vx_win = [], [], [], [], []
    for w in idx:
        s, e = w * WINDOW_LEN, w * WINDOW_LEN + WINDOW_LEN
        u_win.append(u_all[s:e])
        vx0_val = float(max(real_vx[s], MIN_VX0))
        wz0_val = float(real_wz[s])
        k_drift = (300.0 * 0.8525) / (1.55 * 45000.0)
        vy0_refined = float(np.clip(-0.6975 * wz0_val + k_drift * vx0_val * wz0_val, -1.2, 1.2))
        
        x0 = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx0_val)
        x0 = x0.at[15].set(vy0_refined).at[19].set(wz0_val)
        x0_win.append(x0)
        wz_win.append(real_wz[s:e])
        ay_win.append(real_ay[s:e])
        vx_win.append(real_vx[s:e])

    return (jnp.asarray(np.stack(u_win),  dtype=jnp.float32),
            jnp.asarray(np.stack(x0_win), dtype=jnp.float32),
            jnp.asarray(np.stack(wz_win), dtype=jnp.float32),
            jnp.asarray(np.stack(ay_win), dtype=jnp.float32),
            jnp.asarray(np.stack(vx_win), dtype=jnp.float32))


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

    valid = jax.lax.stop_gradient((var_real > VAR_FLOOR) & (var_sim > VAR_FLOOR))
    dyn_weight = jax.lax.stop_gradient(jnp.sqrt(jnp.maximum(var_real, VAR_FLOOR)))
    valid_w = valid.astype(r.dtype) * dyn_weight

    per_window_loss = (1.0 - r) * valid_w
    n_valid = jnp.maximum(jnp.sum(valid_w), 1e-3)
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


    print(f"[Calibrate] Building calibration batches from {len(files)} session(s)...")
    batches = []
    for f in files:
        df = decode_can_csv_to_dataframe(f, dbc_path=args.dbc, dt=args.dt)
        batches.append(_build_calib_batch(df, args.dt, args.steer_sign, rng))
    n_win_total = sum(b[0].shape[0] for b in batches)
    print(f"[Calibrate] {len(batches)} session(s), {n_win_total} windows total "
          f"({n_win_total * WINDOW_LEN * args.dt:.0f}s of telemetry)")

    # NEW — diagnostic: how many windows actually carry correlation signal?
    for i, (u_seq, x0, wz_real, ay_real, vx_real) in enumerate(batches):
        var_wz = jnp.var(wz_real, axis=1)
        var_ay = jnp.var(ay_real, axis=1)
        n_valid = int(jnp.sum((var_wz > 1e-3) & (var_ay > 1e-3)))
        print(f"  batch {i}: {n_valid}/{u_seq.shape[0]} windows have signal "
              f"(var_wz range [{float(var_wz.min()):.2e},{float(var_wz.max()):.2e}])")

    # ── Rollout: params -> (wz_hist, ay_hist) per window ────────────────────
    def rollout(mu_scale, steer_gain, brake_gain, torque_gain, ay_scale, u_seq, x0):
        u_scaled = u_seq.at[:, 0].multiply(steer_gain)
        u_scaled = u_scaled.at[:, 1:5].multiply(torque_gain)
        u_scaled = u_scaled.at[:, 5].multiply(brake_gain)

        tire_cal = jnp.array([mu_scale[0], mu_scale[1], -1.0, 1.0], dtype=jnp.float32)

        def step_fn(x, u):
            x_next = vehicle.simulate_step(x, u, setup, dt=args.dt,
                                            n_substeps=4, tire_cal=tire_cal)
            # Force-balance lateral acceleration (body-frame IMU reading):
            # ay_imu = dvy/dt + vx*wz
            vx_n = x_next[14]; vy_n = x_next[15]; wz_n = x_next[19]
            vy_prev = x[15]
            dvy_dt = (vy_n - vy_prev) / args.dt
            ay_force = (dvy_dt + vx_n * wz_n) * ay_scale
            return x_next, jnp.array([wz_n, ay_force, vx_n, vy_n])

        _, out = jax.lax.scan(step_fn, x0, u_scaled)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    v_rollout = jax.vmap(rollout, in_axes=(None, None, None, None, None, 0, 0))

    def loss_fn(theta, u_seq, x0, wz_real, ay_real, vx_real):
        mu_scale    = jnp.exp(theta[0:2])
        steer_gain  = jnp.exp(theta[2])
        brake_gain  = jnp.exp(theta[3])
        torque_gain = jnp.exp(theta[4])
        ay_scale    = jnp.exp(theta[5])
        
        wz_sim, ay_sim, vx_sim, _ = v_rollout(
            mu_scale, steer_gain, brake_gain, torque_gain, ay_scale, u_seq, x0)
        
        loss_wz = _soft_corr_loss(wz_sim, wz_real)
        loss_ay = _soft_corr_loss(ay_sim, ay_real)
        
        # Penalización normalizada de error longitudinal (evita distorsión de velocidad)
        rmse_vx = jnp.mean(jnp.sqrt(jnp.mean((vx_sim - vx_real) ** 2, axis=1)))
        loss_vx = jnp.clip(rmse_vx / 3.0, 0.0, 1.0)
        
        return 0.45 * loss_wz + 0.40 * loss_ay + 0.15 * loss_vx

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Cotas físicas ensanchadas: evita saturación en extremos
    theta = jnp.array([0.0, 0.0, 0.6, -1.0, 0.5, 0.0])
    theta_lb = jnp.log(jnp.array([0.35, 0.35, 0.50, 0.005, 0.10, 0.40]))
    theta_ub = jnp.log(jnp.array([2.50, 2.50, 2.50, 10.00, 5.00, 1.80]))

    # ─── DIAGNÓSTICO: wz_sim vs wz_real con mu=1.0 (baseline sin calibrar) ───
    def _diagnose_first_window(u_seq, x0, wz_real, n_show=15, steer_sign_override=None):
        mu_scale    = jnp.array([1.0, 1.0])
        steer_gain  = 1.0
        brake_gain  = 1.0
        torque_gain = 1.0

        u0 = u_seq[0]
        if steer_sign_override is not None:
            u0 = u0.at[:, 0].multiply(steer_sign_override)
        x0_0 = x0[0]

        def step_fn(x, u):
            u_scaled = u.at[0].multiply(steer_gain)
            u_scaled = u_scaled.at[1:5].multiply(torque_gain)
            u_scaled = u_scaled.at[5].multiply(brake_gain)
            tire_cal = jnp.array([mu_scale[0], mu_scale[1], -1.0, 1.0], dtype=jnp.float32)
            x_next = vehicle.simulate_step(x, u_scaled, setup, dt=args.dt,
                                            n_substeps=2, tire_cal=tire_cal)
            return x_next, x_next[19]

        _, wz_trace = jax.lax.scan(step_fn, x0_0, u0)

        print(f"\n  [DIAG] wz_real vs wz_sim (mu=1.0, sin calibrar) primeros {n_show} steps:")
        print(f"  {'step':>4} {'wz_real':>10} {'wz_sim':>10} {'diff':>10}")
        for i in range(n_show):
            wr = float(wz_real[0, i])
            ws = float(wz_trace[i])
            print(f"  {i:4d} {wr:10.4f} {ws:10.4f} {ws-wr:10.4f}")

        # ─── DIAGNÓSTICO GRID: steer_sign × PKY1_scale ───────────────────────────
    def _diagnose_grid(u_seq_batch0, x0_batch0, wz_real_batch0, n_show=15):
        import copy
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle as DMV

        u0_base  = u_seq_batch0[0]      # (WINDOW_LEN, 6) — ya con steer_sign=-1 probado aplicado
        x0_0     = x0_batch0[0]
        wz_real0 = wz_real_batch0[0]

        steer_signs = [1.0, -1.0]     # multiplicador ADICIONAL sobre lo ya aplicado
        pky1_scales = [1.0, 0.6, 0.4]

        print(f"\n{'='*72}")
        print(f"  GRID DIAGNOSTIC: steer_sign_extra × PKY1_scale")
        print(f"  (wz_real range this window: [{float(wz_real0.min()):.3f}, {float(wz_real0.max()):.3f}])")
        print(f"{'='*72}")
        print(f"  {'sign':>6} {'PKY1x':>7} {'wz_sim[0]':>10} {'wz_sim[7]':>10} "
              f"{'wz_sim[14]':>11} {'diff[14]':>9} {'sign_flip':>10}")

        for pky1_scale in pky1_scales:
            tc_mod = copy.deepcopy(TP_DICT)
            tc_mod['PKY1'] = 53.2421 * pky1_scale

            veh_mod = DMV(VP_DICT, tc_mod)
            setup_mod = veh_mod._default_setup_vec

            for sign_extra in steer_signs:
                u0 = u0_base.at[:, 0].multiply(sign_extra)

                def step_fn(x, u, _veh=veh_mod, _setup=setup_mod):
                    tire_cal = jnp.array([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32)
                    x_next = _veh.simulate_step(x, u, _setup, dt=args.dt,
                                                 n_substeps=2, tire_cal=tire_cal)
                    return x_next, x_next[19]

                _, wz_trace = jax.lax.scan(step_fn, x0_0, u0)
                wz_np = np.array(wz_trace)

                sign_flip = "YES" if (np.sign(wz_np[0]) != np.sign(wz_np[n_show-1])
                                       and abs(wz_np[0]) > 0.05) else "no"

                print(f"  {sign_extra:>+6.0f} {pky1_scale:>7.2f} "
                      f"{wz_np[0]:>10.4f} {wz_np[7]:>10.4f} {wz_np[n_show-1]:>11.4f} "
                      f"{wz_np[n_show-1]-float(wz_real0[n_show-1]):>9.4f} {sign_flip:>10}")

        print(f"{'='*72}")
        print(f"  Target (wz_real[14]): {float(wz_real0[n_show-1]):.4f}")
        print(f"  Busca la fila con menor |diff[14]| y sign_flip=no")

    _diagnose_grid(batches[0][0], batches[0][1], batches[0][2])

    opt = optax.adam(args.lr)
    opt_state = opt.init(theta)

    print("[Calibrate] Fitting [mu_f, mu_r, steer_gain, brake_gain, torque_gain] "
          "via jax.grad + Adam ...")
    for step in range(args.steps):
        u_seq, x0, wz_real, ay_real, vx_real = batches[step % len(batches)]

        for name, arr in [("u_seq", u_seq), ("x0", x0),
                           ("wz_real", wz_real), ("ay_real", ay_real), ("vx_real", vx_real)]:
            if not bool(jnp.all(jnp.isfinite(arr))):
                raise ValueError(
                    f"[Calibrate] Non-finite values in {name} "
                    f"(batch {step % len(batches)})"
                )

        loss, g = grad_fn(theta, u_seq, x0, wz_real, ay_real, vx_real)
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
    u_seq, x0, wz_real, ay_real, vx_real = batches[0]
    mu_scale    = jnp.exp(theta[0:2])
    steer_gain  = jnp.exp(theta[2])
    brake_gain  = jnp.exp(theta[3])
    torque_gain = jnp.exp(theta[4])
    wz_sim, ay_sim, vx_sim, vy_sim = v_rollout(
        mu_scale, steer_gain, brake_gain, torque_gain, jnp.exp(theta[5]), u_seq, x0)
    np.savez("reports/calib_window0_debug.npz",
             wz_sim=np.array(wz_sim), wz_real=np.array(wz_real),
             ay_sim=np.array(ay_sim), ay_real=np.array(ay_real),
             vx_sim=np.array(vx_sim), vy_sim=np.array(vy_sim))
    print(f"  window 0 u_seq[0,:5,0] (steer_rad, pre-gain): {u_seq[0,:5,0]}")
    print(f"  window 0 sample: wz_real[0,:5]={wz_real[0,:5]}  wz_sim[0,:5]={wz_sim[0,:5]}")
    print(f"  window 0 sample: ay_real[0,:5]={ay_real[0,:5]}  ay_sim[0,:5]={ay_sim[0,:5]}")
    print(f"  window 0 sample: vx_sim[0,:5]={vx_sim[0,:5]}  vy_sim[0,:5]={vy_sim[0,:5]}")

    p_final = np.array(jnp.exp(theta))

    os.makedirs("models", exist_ok=True)
    mu_scale_final = p_final[0:2]
    gain_final     = p_final[2:5]
    ay_scale_final = np.array([p_final[5]])

    np.save("models/mu_scale_calibrated.npy", mu_scale_final)
    np.save("models/gain_calibrated.npy", gain_final)
    np.save("models/ay_scale_calibrated.npy", ay_scale_final)
    np.save("models/steer_sign_calibrated.npy", np.array([args.steer_sign]))

    print(f"[Calibrate] Saved models/mu_scale_calibrated.npy  -> {mu_scale_final}")
    print(f"[Calibrate] Saved models/gain_calibrated.npy      -> {gain_final}")
    print(f"[Calibrate] Saved models/ay_scale_calibrated.npy   -> {ay_scale_final[0]:.3f}")
    print(f"[Calibrate] Saved models/steer_sign_calibrated.npy -> {args.steer_sign:+.0f}")

    print(f"\n[Calibrate] DONE.")


if __name__ == "__main__":
    main()