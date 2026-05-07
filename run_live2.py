"""
run_live.py  — Project-GP  Receding Horizon WMPC
═════════════════════════════════════════════════
GP-vX7: Replaced the one-shot 27m solve with a proper receding horizon
simulation loop that covers the full 427m FSG Autocross circuit.

Architecture
────────────
  One-shot (old): solver.solve(full_track) once → 32 steps → 27m → S/F map
  Receding (new): loop { solve N-step ahead, apply K controls, advance car }
                  until car completes the lap → full circuit shown on map

Why this matters
────────────────
  The WMPC horizon is N × dt × v ≈ 27m. The circuit is 427m. A one-shot
  solve only optimises the start straight; the rest of the map is blank.
  With K=4 controls applied per solve (0.20s of driving), we need
  427/(v×0.20) ≈ 100 solve iterations to cover the full lap.
  After the first solve (which triggers the 2-min JIT compile), every
  subsequent call is a cache hit (~2-5s per solve on CPU).

Usage
─────
  python run_live.py             # full solve (N=64), full lap
  python run_live.py --dev       # N=32, fewer AL iters, faster per solve
  python run_live.py --save      # save PNG frames → figs/frame_NNNN.png
  python run_live.py --K 8       # apply K controls per solve (default 4)
  python run_live.py --max_laps 1.5  # stop after 1.5 laps (default 1)
"""
from __future__ import annotations

import argparse, math, sys, time, os
from pathlib import Path

# ── Lock XLA to CPU BEFORE any jax import ────────────────────────────────────
# Without this, JAX probes for a GPU between solver calls. On machines with
# CUDA hardware but cpu-only jaxlib, this triggers a device switch mid-session
# → XLA cache invalidated → full recompile every solve (~1000s each).
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / 'out'
FIGS_DIR = ROOT / 'figs'
for d in (OUT_DIR, FIGS_DIR):
    d.mkdir(exist_ok=True)

# State vector index aliases (mirrors ocp_solver.py)
STATE_X = 0; STATE_Y = 1; STATE_YAW = 5
STATE_VX = 14; STATE_VY = 15


def _find_nearest_node(x: float, y: float,
                        cx: np.ndarray, cy: np.ndarray,
                        start_idx: int, search_window: int = 60) -> int:
    """Find the closest track node to (x, y), searching ahead of start_idx."""
    n = len(cx)
    idxs = np.arange(start_idx, start_idx + search_window) % n
    dists = (cx[idxs] - x) ** 2 + (cy[idxs] - y) ** 2
    return int(idxs[np.argmin(dists)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev',      action='store_true', help='N=32, 3 AL iters (faster per solve)')
    ap.add_argument('--save',     action='store_true', help='Save PNG frames → figs/')
    ap.add_argument('--interval', type=float, default=1.0, help='Monitor refresh seconds')
    ap.add_argument('--K',        type=int,   default=4,   help='Controls applied per solve step')
    ap.add_argument('--max_laps', type=float, default=1.0, help='Stop after this many laps')
    args = ap.parse_args()

    K = max(1, min(args.K, 16))  # clamp K to [1, 16]

    # ── 1. Track ──────────────────────────────────────────────────────────────
    print("[Live] Building FSG Autocross track…")
    from simulator.track_builder import build_fsg_autocross
    t   = build_fsg_autocross()
    ds  = np.sqrt(np.diff(t.cx)**2 + np.diff(t.cy)**2)
    s_a = np.concatenate([[0], np.cumsum(ds)]).astype(np.float32)

    cx, cy, cpsi = t.cx.astype(np.float32), t.cy.astype(np.float32), t.cpsi.astype(np.float32)
    ck  = t.ck.astype(np.float32)
    wl  = t.width_left.astype(np.float32)
    wr  = t.width_right.astype(np.float32)
    psi_unwrap = np.unwrap(cpsi).astype(np.float32)

    track_total_len = float(s_a[-1])
    n_nodes         = len(cx)
    print(f"    Track: {track_total_len:.1f} m  ({n_nodes} nodes)")

    track_arrays_monitor = (s_a, ck, cx, cy, cpsi, wl, wr)

    # ── 2. Solver ─────────────────────────────────────────────────────────────
    from optimization.ocp_solver import DiffWMPCSolver
    N = 32 if args.dev else 64
    solver = DiffWMPCSolver(
        N_horizon=N,
        mu_friction=1.40,
        V_limit=25.0,
        dev_mode=args.dev,
    )

    # ── 3. Live monitor ───────────────────────────────────────────────────────
    from telemetry.live_monitor import attach_monitor_to_solver
    monitor = attach_monitor_to_solver(
        solver, track_arrays_monitor,
        update_interval=args.interval,
        save_frames=args.save,
        mu=1.40,
    )
    monitor.start()

    # ── 4. Vehicle dynamics (for applying controls between solves) ────────────
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle,
        build_default_setup_28, compute_equilibrium_suspension,
    )
    from config.vehicles.ter26 import vehicle_params as VP
    import jax.numpy as jnp

    vehicle      = DifferentiableMultiBodyVehicle(VP, solver._load_tire_coeffs())
    setup_params = build_default_setup_28(VP)

    # ── 5. Initial car state ──────────────────────────────────────────────────
    k0   = abs(float(ck[0])) + 1e-4
    v0   = min(math.sqrt(1.40 * 9.81 / k0), 25.0)
    x_car = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=v0)
    x_car = x_car.at[STATE_X  ].set(float(cx[0]))
    x_car = x_car.at[STATE_Y  ].set(float(cy[0]))
    x_car = x_car.at[STATE_YAW].set(float(cpsi[0]))

    _T_warm = jnp.array([85., 85., 85., 80., 75., 30., 40.])
    x_car   = x_car.at[28:56].set(jnp.tile(_T_warm, 4))
    z_eq    = compute_equilibrium_suspension(setup_params, VP)
    for i, zi in enumerate(z_eq): x_car = x_car.at[6 + i].set(float(zi))

    # ── 6. Receding horizon loop ───────────────────────────────────────────────
    print(f"\n[Live] Starting receding horizon MPC")
    print(f"       N={N}  K={K}  max_laps={args.max_laps}")
    print(f"       Horizon per solve ≈ {N*0.05*v0:.1f}m  |  K-step advance ≈ {K*0.05*v0:.1f}m")
    print(f"       Est. solves to complete lap: ~{int(track_total_len/(K*0.05*v0))+1}")
    print(f"       NOTE: First solve triggers 2-min JIT compile. All subsequent are fast.\n")

    # Accumulators for full-lap CSV
    all_s, all_t, all_x, all_y   = [], [], [], []
    all_psi, all_v, all_steer     = [], [], []
    all_accel, all_latG, all_kappa = [], [], []
    all_n                          = []

    node_idx   = 0          # current track node index
    s_driven   = 0.0        # arc-length driven so far [m]
    t_elapsed  = 0.0        # time elapsed [s]
    lap_limit  = track_total_len * args.max_laps
    solve_idx  = 0
    t0_wall    = time.time()

    while s_driven < lap_limit:
        # ── Extract track section ahead ───────────────────────────────────────
        # Build lookahead array starting from current node_idx.
        # Use 2× N nodes to give the interpolation enough margin.
        lookahead = 2 * N + 8
        idxs = np.arange(node_idx, node_idx + lookahead) % n_nodes

        s_sec   = s_a[idxs].copy()
        # Make s monotone for interpolation (unwrap across lap boundary)
        for j in range(1, len(s_sec)):
            if s_sec[j] < s_sec[j - 1]:
                s_sec[j:] += track_total_len

        k_sec   = ck[idxs]
        x_sec   = cx[idxs]
        y_sec   = cy[idxs]
        psi_sec = psi_unwrap[idxs]
        wl_sec  = wl[idxs]
        wr_sec  = wr[idxs]

        # ── Solve ──────────────────────────────────────────────────────────────
        t_solve_start = time.time()
        result = solver.solve(
            track_s=s_sec, track_k=k_sec,
            track_x=x_sec, track_y=y_sec, track_psi=psi_sec,
            track_w_left=wl_sec, track_w_right=wr_sec,
        )

        # Reset λ between receding horizon windows.
        solver._al_lambda = None
        solver._al_rho    = solver._AL_RHO_SCHEDULE[0]

        # Push driven path into monitor for live map update.
        # The monitor's _wrap intercepted the solve() call and rendered the
        # current horizon. Now we additionally update the map's racing-line
        # artist with the full accumulated path so far, so the user sees the
        # car traversing the whole circuit progressively.
        if len(all_x) > 1:
            try:
                xy   = np.column_stack([all_x, all_y])
                spds = np.array(all_v)
                pts  = xy.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                from matplotlib.collections import LineCollection
                from matplotlib.colors import Normalize
                lc = LineCollection(segs, cmap='plasma',
                                    norm=Normalize(0, 30),
                                    linewidth=2.5, zorder=5)
                lc.set_array(spds[:-1])
                # Try named key first; fall back to first axis in figure
                ax_map = (monitor._axes.get('map')
                          or monitor._axes.get('racing')
                          or monitor._axes.get('track')
                          or (monitor._fig.axes[0] if monitor._fig.axes else None))
                if ax_map is not None:
                    if hasattr(monitor, '_driven_lc') and monitor._driven_lc in ax_map.collections:
                        monitor._driven_lc.remove()
                    ax_map.add_collection(lc)
                    monitor._driven_lc = lc
                    monitor._flush()
            except Exception:
                pass
        t_solve = time.time() - t_solve_start

        if solve_idx == 0:
            print(f"[Live] First solve complete in {t_solve:.1f}s "
                  f"(includes JIT compile). Subsequent solves will be faster.")

        # ── Apply K controls to the car state ─────────────────────────────────
        U_delta  = result['delta']   # (N,) steer [rad]
        U_accel  = result['accel']   # (N,) force [N]
        R_w = 0.2045

        for k_step in range(min(K, N)):
            steer_k = float(U_delta[k_step])
            force_k = float(U_accel[k_step])

            T_motor = max(force_k, 0.0) * R_w / 4.0
            F_brake = max(-force_k, 0.0)
            u_k = jnp.array([steer_k, T_motor, T_motor, T_motor, T_motor, F_brake])

            x_car = vehicle.simulate_step(
                x_car, u_k, setup_params,
                dt=solver.dt_control, n_substeps=solver.n_substeps,
            )

            # Apply the SAME absolute heading clamp as scan_fn (GP-vX9).
            # Without this, the optimizer learns cornering using the clamp as a
            # crutch (it generates controls that rely on the heading being corrected
            # to psi_ref after each step). When those controls are applied here
            # without the clamp, the car goes straight because the H_net phantom
            # torque isn't counteracted. Model-reality gap → car never corners.
            # Fix: apply psi_ref clamp here too, making model and reality identical.
            psi_ref_k = float(psi_sec[min(k_step, len(psi_sec) - 1)])
            yaw_k     = float(x_car[STATE_YAW])
            dpsi_k    = math.atan2(math.sin(yaw_k - psi_ref_k),
                                    math.cos(yaw_k - psi_ref_k))
            dpsi_cap  = 0.08 * math.tanh(dpsi_k / (0.08 + 1e-9))
            x_car     = x_car.at[STATE_YAW].set(psi_ref_k + dpsi_cap)

            # Record this step in the full-lap accumulator
            vx_k   = float(x_car[STATE_VX])
            x_k    = float(x_car[STATE_X])
            y_k    = float(x_car[STATE_Y])
            psi_k  = float(x_car[STATE_YAW])
            lat_g  = vx_k ** 2 * abs(float(ck[node_idx % n_nodes])) / 9.81

            all_s.append(s_driven)
            all_t.append(t_elapsed)
            all_x.append(x_k)
            all_y.append(y_k)
            all_psi.append(psi_k)
            all_v.append(vx_k)
            all_steer.append(steer_k)
            all_accel.append(force_k)
            all_latG.append(lat_g)
            all_kappa.append(float(ck[node_idx % n_nodes]))
            all_n.append(float(result['n'][k_step]) if k_step < len(result['n']) else 0.0)

            s_driven  += max(vx_k, 0.5) * solver.dt_control
            t_elapsed += solver.dt_control

        # ── Advance track pointer ─────────────────────────────────────────────
        node_idx = _find_nearest_node(
            float(x_car[STATE_X]), float(x_car[STATE_Y]),
            cx, cy, node_idx, search_window=60,
        )

        solve_idx += 1
        v_now = float(x_car[STATE_VX])
        pct   = min(s_driven / lap_limit * 100, 100.0)

        print(f"  [solve {solve_idx:3d}]  s={s_driven:6.1f}m ({pct:5.1f}%)  "
              f"v={v_now:5.2f}m/s  t={t_elapsed:6.2f}s  "
              f"solve_wall={t_solve:.1f}s")

        if s_driven >= lap_limit:
            break

    # ── 7. Results ────────────────────────────────────────────────────────────
    wall_total = time.time() - t0_wall
    print(f"\n[Live] Lap complete!")
    print(f"  Simulated lap time : {t_elapsed:.3f} s")
    print(f"  Distance driven    : {s_driven:.1f} m")
    print(f"  Mean speed         : {s_driven/max(t_elapsed,1e-3):.2f} m/s")
    print(f"  Wall time (total)  : {wall_total:.1f} s  ({solve_idx} solves)")

    # Save golden lap
    df = pd.DataFrame({
        's':     all_s,    't':     all_t,
        'x':     all_x,    'y':     all_y,
        'psi':   all_psi,  'v':     all_v,
        'steer': all_steer,'accel': all_accel,
        'latG':  all_latG, 'kappa': all_kappa,
        'n':     all_n,
    })
    df.to_csv(OUT_DIR / 'golden_lap.csv', index=False)
    print(f"  Saved → {OUT_DIR / 'golden_lap.csv'}  ({len(df)} rows)")

    print("\n[Live] Close the figure window to exit.")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show(block=True)
    monitor.close()


if __name__ == '__main__':
    main()