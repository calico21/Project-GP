#!/usr/bin/env python3
# scripts/calibrate_mu_from_telemetry.py
# Project-GP — Automated Multi-Objective Vehicle Parameter Identification
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

N_CAL_WINDOWS = 25   # Submuestreo balanceado por sesión

jax.config.update("jax_debug_nans", True)

# ─────────────────────────────────────────────────────────────────────────────
# §1  Construcción de Batches con canal vx, wz y ay
# ─────────────────────────────────────────────────────────────────────────────

def _build_calib_batch(df, dt: float, steer_sign: float, rng: np.random.Generator):
    N = len(df)
    n_windows = N // WINDOW_LEN

    real_vx = _extract_1d(df, 'vx_mps')

    MIN_VX0 = 3.0
    candidate_windows = np.array([
        w for w in range(n_windows)
        if real_vx[w * WINDOW_LEN] >= MIN_VX0
    ])
    if len(candidate_windows) == 0:
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
    u_all[:, 5]   = np.clip(u_all[:, 5],   0.0,   5000.0)

    real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
    real_ay = _extract_1d(df, 'ay_mps2')

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
# §2  Funciones de Pérdida
# ─────────────────────────────────────────────────────────────────────────────

def _soft_corr_loss(sim: jax.Array, real: jax.Array) -> jax.Array:
    sim_c  = sim  - jnp.mean(sim,  axis=1, keepdims=True)
    real_c = real - jnp.mean(real, axis=1, keepdims=True)

    var_sim  = jnp.sum(sim_c ** 2,  axis=1)
    var_real = jnp.sum(real_c ** 2, axis=1)

    VAR_FLOOR = 1e-3
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
# §3  Bucle de Estimación de Parámetros en JAX
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw_can_logs"))
    ap.add_argument("--dbc",      type=Path, default=Path("TER.dbc"))
    ap.add_argument("--dt",       type=float, default=0.005)
    ap.add_argument("--steps",    type=int,   default=250)
    ap.add_argument("--lr",       type=float, default=0.03)
    ap.add_argument("--steer-sign", type=float, default=1.0)
    args = ap.parse_args()

    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
    setup   = vehicle._default_setup_vec
    rng     = np.random.default_rng(0)

    files = sorted(args.data_dir.glob("*.csv"))
    if not files:
        print(f"[!] No CSV logs found in {args.data_dir}")
        return

    print(f"[AutoCalib] Sincronizando batches desde {len(files)} sesiones...")
    batches = []
    for f in files:
        df = decode_can_csv_to_dataframe(f, dbc_path=args.dbc, dt=args.dt)
        batches.append(_build_calib_batch(df, args.dt, args.steer_sign, rng))
    n_win_total = sum(b[0].shape[0] for b in batches)
    print(f"[AutoCalib] {len(batches)} sesiones, {n_win_total} ventanas ({n_win_total * WINDOW_LEN * args.dt:.0f}s de pista)")

    def rollout(mu_scale, steer_gain, brake_gain, torque_gain, u_seq, x0):
        u_scaled = u_seq.at[:, 0].multiply(steer_gain)
        u_scaled = u_scaled.at[:, 1:5].multiply(torque_gain)
        u_scaled = u_scaled.at[:, 5].multiply(brake_gain)

        tire_cal = jnp.array([mu_scale[0], mu_scale[1], -1.0, 1.0], dtype=jnp.float32)

        def step_fn(x, u):
            x_next = vehicle.simulate_step(x, u, setup, dt=args.dt,
                                            n_substeps=4, tire_cal=tire_cal)
            vx_n = x_next[14]; wz_n = x_next[19]
            ay_force = vx_n * wz_n
            return x_next, jnp.array([wz_n, ay_force, vx_n])

        _, out = jax.lax.scan(step_fn, x0, u_scaled)
        return out[:, 0], out[:, 1], out[:, 2]

    v_rollout = jax.vmap(rollout, in_axes=(None, None, None, None, 0, 0))

    def loss_fn(theta, u_seq, x0, wz_real, ay_real, vx_real):
        mu_scale    = jnp.exp(theta[0:2])
        steer_gain  = jnp.exp(theta[2])
        brake_gain  = jnp.exp(theta[3])
        torque_gain = jnp.exp(theta[4])
        
        wz_sim, ay_sim, vx_sim = v_rollout(
            mu_scale, steer_gain, brake_gain, torque_gain, u_seq, x0)
        
        loss_wz = _soft_corr_loss(wz_sim, wz_real)
        loss_ay = _soft_corr_loss(ay_sim, ay_real)
        
        # Pérdida combinada en velocidad: forma (correlación) + magnitud (RMSE)
        rmse_vx = jnp.mean(jnp.sqrt(jnp.mean((vx_sim - vx_real) ** 2, axis=1)))
        loss_vx_mag = jnp.clip(rmse_vx / 3.0, 0.0, 1.0)
        loss_vx_shape = _soft_corr_loss(vx_sim, vx_real)
        loss_vx = 0.5 * loss_vx_mag + 0.5 * loss_vx_shape
        
        return 0.40 * loss_wz + 0.35 * loss_ay + 0.25 * loss_vx

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Vector theta = log([mu_f, mu_r, steer_gain, brake_gain, torque_gain])
    theta = jnp.zeros(5)
    # Subir cota de mu_r a 2.00 para absorber la carga aerodinámica trasera
    theta_lb = jnp.log(jnp.array([0.50, 0.50, 0.70, 0.05, 0.20]))
    theta_ub = jnp.log(jnp.array([1.80, 2.00, 1.30, 2.50, 2.00]))

    opt = optax.adam(args.lr)
    opt_state = opt.init(theta)

    print("[AutoCalib] Optimizando parámetros físicos vía jax.grad + Adam...")
    for step in range(args.steps):
        u_seq, x0, wz_real, ay_real, vx_real = batches[step % len(batches)]

        loss, g = grad_fn(theta, u_seq, x0, wz_real, ay_real, vx_real)
        g_clean = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        updates, opt_state = opt.update(g_clean, opt_state, theta)
        theta = jnp.clip(optax.apply_updates(theta, updates), theta_lb, theta_ub)

        if step % 25 == 0 or step == args.steps - 1:
            p = jnp.exp(theta)
            print(f"  step {step:4d} | loss={float(loss):.4f} | "
                  f"mu_f={p[0]:.3f} mu_r={p[1]:.3f} | "
                  f"steer={p[2]:.3f} brake={p[3]:.3f} trq={p[4]:.3f}")

    p_final = np.array(jnp.exp(theta))

    os.makedirs("models", exist_ok=True)
    np.save("models/mu_scale_calibrated.npy", p_final[0:2])
    np.save("models/gain_calibrated.npy", p_final[2:5])
    np.save("models/ay_scale_calibrated.npy", np.array([1.0]))
    np.save("models/steer_sign_calibrated.npy", np.array([args.steer_sign]))

    print(f"\n[AutoCalib] Calibración finalizada. Parámetros guardados:")
    print(f"  -> mu_f:       {p_final[0]:.3f}")
    print(f"  -> mu_r:       {p_final[1]:.3f}")
    print(f"  -> steer_gain: {p_final[2]:.3f}")
    print(f"  -> brake_gain: {p_final[3]:.3f}")
    print(f"  -> torque_gain:{p_final[4]:.3f}")


if __name__ == "__main__":
    main()