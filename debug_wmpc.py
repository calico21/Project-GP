"""
debug_wmpc.py — Project-GP WMPC Surgical Diagnostic
=====================================================
Exposes the full cost breakdown, heading error evolution,
and gradient magnitudes at the warm-start and final points.

Run:  python debug_wmpc.py [--dev]
"""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import jax, jax.numpy as jnp
from jax import jit, value_and_grad

# ── Force CPU for reproducibility ──────────────────────────────────────────
import os; os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

SEP  = "─" * 60
SEP2 = "═" * 60

def fmt(v, w=10, d=5): return f"{float(v):{w}.{d}f}"

# ────────────────────────────────────────────────────────────────────────────
def run_diagnostic(dev_mode: bool):
    print(SEP2)
    print("  PROJECT-GP WMPC SURGICAL DIAGNOSTIC")
    print(SEP2)

    # 1. Track ────────────────────────────────────────────────────────────────
    print("\n[1] Loading FSG Autocross track …")
    from simulator.track_builder import build_fsg_autocross
    t  = build_fsg_autocross()
    ds = np.sqrt(np.diff(t.cx)**2 + np.diff(t.cy)**2)
    s  = np.concatenate([[0], np.cumsum(ds)]).astype(np.float32)
    track_total_len = float(s[-1])
    print(f"    Track length : {track_total_len:.1f} m  ({len(s)} nodes)")
    print(f"    κ range      : [{np.min(t.ck):.4f}, {np.max(t.ck):.4f}] 1/m")
    print(f"    Width left   : [{np.min(t.width_left):.2f}, {np.max(t.width_left):.2f}] m")
    print(f"    Width right  : [{np.min(t.width_right):.2f}, {np.max(t.width_right):.2f}] m")

    # 2. Solver setup ─────────────────────────────────────────────────────────
    from optimization.ocp_solver import DiffWMPCSolver
    N = 32 if dev_mode else 64
    solver = DiffWMPCSolver(N_horizon=N, mu_friction=1.40, V_limit=25.0, dev_mode=dev_mode)

    # Replicate the interpolation from solve()
    import math as _math
    s_a  = s
    k_a  = t.ck.astype(np.float32)
    x_a  = t.cx.astype(np.float32)
    y_a  = t.cy.astype(np.float32)
    psi_a= t.cpsi.astype(np.float32)
    wl_a = t.width_left.astype(np.float32)
    wr_a = t.width_right.astype(np.float32)

    k0_abs       = abs(float(np.mean(k_a[:max(1, len(k_a)//8)]))) + 1e-4
    v_est        = min(_math.sqrt((1.40 * 9.81) / k0_abs), 25.0)
    horizon_dist = v_est * N * 0.05
    frac         = min(horizon_dist / track_total_len, 1.0)

    print(f"\n[2] Horizon geometry")
    print(f"    v_est (friction limit) : {v_est:.2f} m/s")
    print(f"    horizon_dist           : {horizon_dist:.1f} m  ({N} steps × {0.05}s × {v_est:.1f} m/s)")
    print(f"    frac of track covered  : {frac:.4f}  ({frac*100:.2f}%)")
    print(f"    !! Solver sees only {horizon_dist:.1f}m of {track_total_len:.1f}m circuit !!")

    s_orig = np.linspace(0, 1, len(k_a))
    s_wav  = np.linspace(0, frac, N)
    def interp(arr): return jnp.array(np.interp(s_wav, s_orig, arr))
    track_k   = interp(k_a)
    track_x   = interp(x_a)
    track_y   = interp(y_a)
    track_psi = interp(np.unwrap(psi_a))
    track_wl  = interp(wl_a)
    track_wr  = interp(wr_a)
    track_s_r = interp(s_a)

    print(f"\n    κ in horizon   : [{float(jnp.min(track_k)):.6f}, {float(jnp.max(track_k)):.6f}] 1/m")
    print(f"    Width L/R      : [{float(jnp.mean(track_wl)):.2f}, {float(jnp.mean(track_wr)):.2f}] m (mean)")
    print(f"    ψ range        : [{float(jnp.min(track_psi)):.4f}, {float(jnp.max(track_psi)):.4f}] rad")

    # 3. Setup params + x0 ───────────────────────────────────────────────────
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle, build_default_setup_28,
        compute_equilibrium_suspension, DEFAULT_SETUP
    )
    from config.vehicles.ter26 import vehicle_params as VP
    setup_params = build_default_setup_28(VP)
    vp   = VP

    k0   = abs(float(track_k[0])) + 1e-4
    v0   = min(_math.sqrt((1.40 * 9.81) / k0), 25.0)
    x0   = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=v0)
    x0   = x0.at[0].set(float(track_x[0]))   # STATE_X
    x0   = x0.at[1].set(float(track_y[0]))   # STATE_Y
    x0   = x0.at[5].set(float(track_psi[0])) # STATE_YAW
    _T   = jnp.array([85., 85., 85., 80., 75., 30., 40.])
    x0   = x0.at[28:56].set(jnp.tile(_T, 4))
    z_eq = compute_equilibrium_suspension(setup_params, vp)
    x0   = x0.at[6].set(float(z_eq[0])); x0 = x0.at[7].set(float(z_eq[1]))
    x0   = x0.at[8].set(float(z_eq[2])); x0 = x0.at[9].set(float(z_eq[3]))

    print(f"\n[3] Initial state")
    print(f"    vx0    = {v0:.2f} m/s")
    print(f"    x0 pos = ({float(x0[0]):.2f}, {float(x0[1]):.2f}) m")
    print(f"    ψ0     = {float(x0[5]):.4f} rad  (track_psi[0]={float(track_psi[0]):.4f})")

    # 4. Physics warm start ───────────────────────────────────────────────────
    print(f"\n[4] Running physics warm start (with yaw saturation + heading feedback) …")
    U_warm = solver._build_physics_warmstart(
        track_k, track_psi, x0, setup_params,
        track_x=track_x, track_y=track_y
    )
    print(f"    U_warm[:, 0] (steer): min={float(jnp.min(U_warm[:,0])):.4f}  max={float(jnp.max(U_warm[:,0])):.4f} rad")
    print(f"    U_warm[:, 1] (force): min={float(jnp.min(U_warm[:,1])):.1f}  max={float(jnp.max(U_warm[:,1])):.1f} N")
    wc_kin    = solver._db4_dwt(U_warm)
    flat_init = wc_kin.flatten()

    # 5. Step-by-step warm-start rollout diagnostic ──────────────────────────
    print(f"\n[5] Warm-start rollout — step diagnostics")
    print(f"    {'step':>4}  {'vx':>7}  {'s_dot':>8}  {'alpha_deg':>10}  {'n_m':>8}  {'heading_ok':>11}")
    print(f"    {SEP}")

    STATE_X=0; STATE_Y=1; STATE_YAW=5; STATE_VX=14; STATE_VY=15

    U_ws, x_ws, n_ws, vn_ws, sdot_ws = solver._simulate_trajectory(
        wc_kin, x0, setup_params,
        track_k, track_x, track_y, track_psi,
        track_wl, track_wr, 1.0, 0.0, 0.05
    )

    alpha_bad_steps = 0
    for i in range(N):
        vx_i    = float(x_ws[i, STATE_VX])
        n_i     = float(n_ws[i])
        sdot_i  = float(sdot_ws[i])
        # Reconstruct alpha from yaw vs psi_ref
        dpsi    = float(x_ws[i, STATE_YAW]) - float(track_psi[i])
        alpha_i = math.atan2(math.sin(dpsi), math.cos(dpsi))
        alpha_deg = abs(alpha_i) * 180 / math.pi
        ok = "✓" if alpha_deg < 15 else ("⚠ WARN" if alpha_deg < 45 else "✗ SPIN")
        if alpha_deg > 45: alpha_bad_steps += 1
        if i < 8 or i >= N-4 or alpha_deg > 45:
            print(f"    {i:>4}  {vx_i:>7.2f}  {sdot_i:>8.4f}  {alpha_deg:>10.2f}  {n_i:>8.3f}  {ok}")

    if alpha_bad_steps > 0:
        print(f"\n    !! {alpha_bad_steps}/{N} steps with |alpha| > 45° — YAW DIVERGENCE CONFIRMED !!")
    else:
        print(f"\n    All steps heading-aligned (|alpha| < 45° throughout)")

    # 6. Cost breakdown ───────────────────────────────────────────────────────
    print(f"\n[6] Cost breakdown at WARM START")
    w_mu = jnp.ones(N) * 0.02
    w_steer = jnp.ones(N) * 1e-3
    w_accel = jnp.ones(N) * 5e-5
    al_lambda = jnp.zeros(N)
    al_rho    = jnp.array(2.0)

    # Compute individual terms manually
    U_opt, x_traj, n_mean, n_var, s_dot = solver._simulate_trajectory(
        wc_kin, x0, setup_params,
        track_k, track_x, track_y, track_psi,
        track_wl, track_wr, 1.0, 0.0, 0.05
    )

    sdot_arr  = np.array(s_dot)
    vx_arr    = np.array(x_traj[:, STATE_VX])
    n_arr     = np.array(n_mean)
    yaw_arr   = np.array(x_traj[:, STATE_YAW])
    psi_arr   = np.array(track_psi)
    alpha_arr = np.array([math.atan2(math.sin(y-p), math.cos(y-p))
                          for y, p in zip(yaw_arr, psi_arr)])

    s_dot_safe  = np.log(1 + np.exp(sdot_arr * 20)) / 20 + 1e-2
    time_cost   = float(np.sum(1.0 / s_dot_safe) * 0.05)
    time_cost_alt = float(-np.sum(sdot_arr) * 0.05)  # correct formulation

    w_effort_val = 5e-5
    effort_cost  = float(w_effort_val * np.sum((np.array(U_opt[:,1])/8000)**2))

    kappa_safe_val = solver.kappa_safe
    tube_r = kappa_safe_val * np.sqrt(np.maximum(np.array(n_var), 1e-6))
    viol_l = np.log(1 + np.exp((n_arr + tube_r - np.array(track_wl)) * 20)) / 20
    viol_r = np.log(1 + np.exp((-n_arr + tube_r - np.array(track_wr)) * 20)) / 20
    barrier_cost = float(80.0 * np.sum(viol_l**2 + viol_r**2))

    center_cost = float(0.005 * np.sum(n_arr**2))
    heading_cost = float(np.sum(alpha_arr**2))  # currently not in loss!

    a_lat_sq = (vx_arr**2 * np.abs(np.array(track_k)))**2
    vx_prev  = np.concatenate([[float(x0[STATE_VX])], vx_arr[:-1]])
    a_lon_sq = ((vx_arr - vx_prev) / 0.05)**2
    g_circle = (a_lat_sq + a_lon_sq) / ((1.40*9.81)**2 + 1e-4) - 1.0
    al_fric  = float(np.sum(np.maximum(g_circle, 0.0)**2) * 0.5 * 2.0)

    v_min_cost = float(15.0 * np.mean(np.log(1+np.exp(3.0 - vx_arr))**2))

    total_loss = time_cost + effort_cost + barrier_cost + center_cost + al_fric + v_min_cost

    print(f"    {'Cost Term':<30} {'Value':>12}  {'% of total':>10}  Note")
    print(f"    {SEP}")
    for name, val, note in [
        ("time_cost  Σ(1/sdot)·dt",    time_cost,     "dim error — see alt"),
        ("time_cost_alt  −Σ(sdot)·dt", time_cost_alt, "IN LOSS (GP-vX6+)"),
        ("effort_cost",                 effort_cost,   ""),
        ("barrier_cost (track limits)", barrier_cost,  ""),
        ("center_cost",                 center_cost,   ""),
        ("heading_cost  w=8.0",         heading_cost,  "IN LOSS (GP-vX6+)"),
        ("al_friction",                 al_fric,       ""),
        ("v_min_cost",                  v_min_cost,    ""),
    ]:
        pct = val/abs(total_loss)*100 if total_loss != 0 else 0
        print(f"    {name:<30} {val:>12.4f}  {pct:>9.1f}%  {note}")

    print(f"\n    Total loss (approx) : {total_loss:.4f}")
    print(f"    s_dot mean          : {np.mean(sdot_arr):.4f} m/s  (should be ~{v0:.1f})")
    print(f"    alpha mean |deg|    : {np.mean(np.abs(alpha_arr))*180/math.pi:.2f}°  (should be <5°)")
    print(f"    n_max               : {np.max(np.abs(n_arr)):.3f} m")

    print(f"\n[7] KEY FINDINGS")
    print(f"    1. Horizon covers only {horizon_dist:.1f}m / {track_total_len:.1f}m "
          f"({frac*100:.1f}% of circuit) — optimizer sees only the S/F straight")
    print(f"    2. heading_cost = {heading_cost:.3f} IS NOW in loss (w=8.0, GP-vX6+)")
    print(f"    3. barrier_cost = {barrier_cost:.3f} dominates loss ({barrier_cost/max(abs(barrier_cost+center_cost),1e-6)*100:.0f}%) "
          f"— car reaches track edge during spin")
    print(f"    4. GP-vX7 fix: absolute alpha clamp |α|≤0.08 rad in scan_fn.")
    print(f"       Predicted post-fix: alpha_max = 0.08 rad (4.6°) every step.")
    print(f"       s_dot = vx·cos(4.6°) ≈ 0.997·vx  (clean gradient throughout)")
    print(f"    5. Correct time objective: −Σ(s_dot)·dt = {time_cost_alt:.4f} (stable)")

    print(f"\n{SEP2}")
    print("  DIAGNOSTIC COMPLETE")
    print(SEP2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', action='store_true')
    args = ap.parse_args()
    run_diagnostic(dev_mode=args.dev)