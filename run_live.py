"""
run_live.py
───────────
Identical to run_validation.py but attaches the LiveOptMonitor before
calling solver.solve(), so you can watch the racing line being built
in real time.

Usage:
    python run_live.py             # full solve, N=64, live window
    python run_live.py --dev       # N=32, faster, live window
    python run_live.py --save      # save a PNG frame every 2 s → figs/frame_NNNN.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / 'out'
FIGS_DIR = ROOT / 'figs'
REP_DIR  = ROOT / 'reports'
for d in (OUT_DIR, FIGS_DIR, REP_DIR):
    d.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev',   action='store_true', help='N=32, 3 AL iters')
    ap.add_argument('--save',  action='store_true', help='Save PNG frames to figs/')
    ap.add_argument('--interval', type=float, default=2.0,
                    help='Seconds between figure refreshes (default 2.0)')
    args = ap.parse_args()

    # ── 1. Track ──────────────────────────────────────────────────────────────
    print("[Live] Building FSG Autocross track…")
    from simulator.track_builder import build_fsg_autocross
    t  = build_fsg_autocross()
    ds = np.sqrt(np.diff(t.cx)**2 + np.diff(t.cy)**2)   # was track.cx/track.cy
    s  = np.concatenate([[0], np.cumsum(ds)]).astype(np.float32)
    track_arrays = (
        s,
        t.ck.astype(np.float32),
        t.cx.astype(np.float32),
        t.cy.astype(np.float32),
        t.cpsi.astype(np.float32),
        t.width_left.astype(np.float32),
        t.width_right.astype(np.float32),
    )

    # ── 2. Solver ─────────────────────────────────────────────────────────────
    from optimization.ocp_solver import DiffWMPCSolver
    N = 32 if args.dev else 64
    solver = DiffWMPCSolver(
        N_horizon=N,
        mu_friction=1.40,
        V_limit=25.0,
        dev_mode=args.dev,
    )

    # ── 3. Attach live monitor (two lines) ────────────────────────────────────
    from telemetry.live_monitor import attach_monitor_to_solver
    monitor = attach_monitor_to_solver(
        solver, track_arrays,
        update_interval=args.interval,
        save_frames=args.save,
        mu=1.40,
    )
    monitor.start()   # opens the window

    # ── 4. Solve (monitor updates automatically) ──────────────────────────────
    print(f"[Live] Starting solve (N={N}, dev={args.dev}) — watch the window…")
    s_a, k_a, x_a, y_a, psi_a, wl_a, wr_a = track_arrays
    result = solver.solve(
        track_s=s_a, track_k=k_a,
        track_x=x_a, track_y=y_a, track_psi=psi_a,
        track_w_left=wl_a, track_w_right=wr_a,
    )

    # ── 5. Final state ────────────────────────────────────────────────────────
    print(f"\n[Live] Solve complete.")
    print(f"  Lap time           : {result['time']:.3f} s")
    print(f"  Max G combined     : {result['g_combined_max']:.3f}")
    print(f"  Friction compliance: {result['friction_compliance_pct']:.1f}%")

    # Save golden lap CSV
    v      = result['v']
    psi_r  = result['psi']
    n_r    = result['n']
    k_r    = result['k']
    ds     = np.diff(result['s'], prepend=result['s'][0])
    t_arr  = np.cumsum(ds / np.maximum(v, 0.5))

    pd.DataFrame({
        's':     result['s'],
        't':     t_arr,
        'x':     x_a[:N] + n_r * (-np.sin(psi_r)),
        'y':     y_a[:N] + n_r * ( np.cos(psi_r)),
        'psi':   psi_r,
        'v':     v,
        'steer': result['delta'],
        'accel': result['accel'],
        'latG':  result['lat_g'],
        'kappa': k_r,
        'n':     n_r,
    }).to_csv(OUT_DIR / 'golden_lap.csv', index=False)
    print(f"  Saved → {OUT_DIR / 'golden_lap.csv'}")

    # Keep window open until user closes it
    print("\n[Live] Close the figure window to exit.")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show(block=True)
    monitor.close()


if __name__ == '__main__':
    main()