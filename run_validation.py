"""
run_validation.py
─────────────────
Single-entry-point runner for Project-GP telemetry validation.

Workflow:
  1. Build FSG Autocross track geometry (track_builder.py)
  2. Run DiffWMPCSolver → save out/golden_lap.csv
  3. (Optional) stub Simple-TV and mpQP-TV laps by perturbing the golden solution
     — useful before real closed-loop sims are wired
  4. Call ValidationPipeline.render_all() → figs/ + reports/

Usage:
    cd ~/FS_Driver_Setup_Optimizer
    python run_validation.py                     # full pipeline, golden + stubs
    python run_validation.py --no-stubs          # golden only
    python run_validation.py --skip-solve        # reuse existing out/*.csv
    python run_validation.py --dev               # fast dev-mode solve (N=32, fewer AL iters)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path bootstrap (run from project root) ───────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / 'out'
FIGS_DIR = ROOT / 'figs'
REP_DIR  = ROOT / 'reports'

OUT_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)
REP_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# §1  Track geometry
# ════════════════════════════════════════════════════════════════════════════

def load_track():
    """Returns (track_s, track_k, track_x, track_y, track_psi, w_left, w_right)
    as plain numpy arrays from the FSG Autocross preset."""
    from simulator.track_builder import build_fsg_autocross
    t = build_fsg_autocross()

    n = len(t.cx)
    s = np.arange(n, dtype=np.float32)  # 1 m spacing by construction
    return (
        s,
        t.ck.astype(np.float32),
        t.cx.astype(np.float32),
        t.cy.astype(np.float32),
        t.cpsi.astype(np.float32),
        t.width_left.astype(np.float32),
        t.width_right.astype(np.float32),
    )


# ════════════════════════════════════════════════════════════════════════════
# §2  Solver → CSV
# ════════════════════════════════════════════════════════════════════════════

def run_solver(track_arrays, dev_mode: bool = False) -> pd.DataFrame:
    """Runs DiffWMPCSolver and returns a DataFrame with all channels
    needed by ValidationPipeline."""
    from optimization.ocp_solver import DiffWMPCSolver

    s, k, x, y, psi, wl, wr = track_arrays

    N = 32 if dev_mode else 64
    solver = DiffWMPCSolver(
        N_horizon=N,
        mu_friction=1.40,
        V_limit=25.0,
        dev_mode=dev_mode,
    )

    print(f"\n[Runner] Solving Golden Lap  (N={N}, dev={dev_mode}) …")
    result = solver.solve(
        track_s=s, track_k=k, track_x=x, track_y=y, track_psi=psi,
        track_w_left=wl, track_w_right=wr,
    )
    print(f"[Runner] Golden Lap time: {result['time']:.3f} s  |  "
          f"G-max: {result['g_combined_max']:.3f}  |  "
          f"Friction compliance: {result['friction_compliance_pct']:.1f}%")

    # Build time axis from s_dot approximation (uniform dt, cumulative)
    v   = result['v']
    s_r = result['s']
    ds  = np.diff(s_r, prepend=s_r[0])
    dt  = np.where(v > 0.5, ds / np.maximum(v, 0.5), 0.05)
    t   = np.cumsum(dt)

    # Reconstruct (x, y) by integrating (psi, v) — ocp_solver returns
    # track-frame x/y which are already stored in result, but also compute
    # vx, vy, wz from the heading and track curvature for downstream channels.
    psi_r = result['psi']
    vx    = v * np.cos(psi_r)
    vy    = v * np.sin(psi_r)
    wz    = v * result['k']   # ω_z ≈ v·κ (kinematic approximation for CSV)

    accel = result['accel']   # Newton-scale, signed: + = drive, - = brake

    df = pd.DataFrame({
        's':     s_r,
        't':     t,
        # spatial: reconstruct x/y from track reference + lateral deviation n
        'x':     x[:N] + np.sin(psi_r) * result['n'],   # x_track + n·N̂_x
        'y':     y[:N] - np.cos(psi_r) * result['n'],   # y_track + n·N̂_y  (signed)
        'psi':   psi_r,
        'v':     v,
        'vx':    vx,
        'vy':    vy,
        'wz':    wz,
        'steer': result['delta'],    # rad
        'accel': accel,              # N, signed
        'latG':  result['lat_g'],
        'kappa': result['k'],
        'n':     result['n'],
        'var_n': result['var_n'],
    })
    return df


# ════════════════════════════════════════════════════════════════════════════
# §3  Stub laps (perturbed golden) — stand-ins until real sims are wired
# ════════════════════════════════════════════════════════════════════════════

def make_stub_simple_tv(golden: pd.DataFrame) -> pd.DataFrame:
    """Simulates a 'Simple TV' strategy by:
      - Clipping peak lateral G to 85% of golden (conservative yaw control)
      - Adding high-frequency steering jitter (PD over-correction artefact)
      - Scaling v down by ~4% through corners
    Stubs are clearly labelled and should be replaced with real powertrain sims.
    """
    rng  = np.random.default_rng(42)
    df   = golden.copy()
    k    = df['kappa'].to_numpy()
    v    = df['v'].to_numpy()
    s_raw = df['steer'].to_numpy()

    # Speed penalty in corners
    v_pen   = np.where(np.abs(k) > 0.04, v * 0.96, v)
    v_pen   = v_pen + rng.normal(0, 0.05, len(v_pen))  # sensor noise

    # Steering jitter (high-freq content that WMPC suppresses)
    jitter  = rng.normal(0, 0.008, len(s_raw))
    steer_s = s_raw + jitter

    # Recalculate latG and time axis
    latG_s  = v_pen ** 2 * np.abs(k) / 9.81 * 0.88   # ~12% grip loss
    ds      = np.diff(df['s'].to_numpy(), prepend=df['s'].iloc[0])
    dt      = ds / np.maximum(v_pen, 0.5)
    t_s     = np.cumsum(dt)

    df = df.copy()
    df['v']     = v_pen
    df['steer'] = steer_s
    df['latG']  = latG_s
    df['t']     = t_s
    df['wz']    = v_pen * k   # recalculate

    # Fake per-wheel torques (simple TV: equal rear split + passive front)
    total_drive = np.maximum(df['accel'].to_numpy(), 0.0) / 4.0
    df['T_fl']  =  total_drive * 0.45
    df['T_fr']  =  total_drive * 0.45
    df['T_rl']  =  total_drive * 0.55
    df['T_rr']  =  total_drive * 0.55
    return df


def make_stub_mpqp_tv(golden: pd.DataFrame) -> pd.DataFrame:
    """Simulates mpQP-TV: tighter yaw control than Simple, but still not optimal.
    Aggressive rear-torque split on corner exit (RL+RR split) visible in
    the TV panel.
    """
    rng  = np.random.default_rng(99)
    df   = golden.copy()
    k    = df['kappa'].to_numpy()
    v    = df['v'].to_numpy()

    # Speed: 98% of golden (mpQP is closer to optimal)
    v_pen = np.where(np.abs(k) > 0.04, v * 0.98, v)
    v_pen = v_pen + rng.normal(0, 0.02, len(v_pen))

    latG_s = v_pen ** 2 * np.abs(k) / 9.81 * 0.97
    ds     = np.diff(df['s'].to_numpy(), prepend=df['s'].iloc[0])
    dt     = ds / np.maximum(v_pen, 0.5)
    t_s    = np.cumsum(dt)

    df = df.copy()
    df['v']    = v_pen
    df['latG'] = latG_s
    df['t']    = t_s
    df['wz']   = v_pen * k

    # mpQP TV: aggressive rear split → RL/RR imbalance in corner exit phase
    total_drive  = np.maximum(df['accel'].to_numpy(), 0.0)
    corner_exit  = (np.abs(k) > 0.03) & (np.gradient(v_pen) > 0)
    # On corner exit: outer rear gets boosted, inner rear reduced
    split_factor = np.where(corner_exit, 0.35, 0.25)   # outer rear bias
    df['T_fl']   =  total_drive * 0.20
    df['T_fr']   =  total_drive * 0.20
    # K > 0 → left turn → right (outer) rear gets boost
    right_boost  = np.where(k > 0, split_factor, 0.25)
    left_boost   = np.where(k < 0, split_factor, 0.25)
    df['T_rl']   =  total_drive * left_boost
    df['T_rr']   =  total_drive * right_boost
    return df


# ════════════════════════════════════════════════════════════════════════════
# §4  Entrypoint
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Project-GP — Generate CSVs + run validation pipeline')
    ap.add_argument('--skip-solve', action='store_true',
                    help='Skip solver, reuse existing out/*.csv')
    ap.add_argument('--no-stubs',  action='store_true',
                    help='Do not generate stub Simple/mpQP laps')
    ap.add_argument('--dev',       action='store_true',
                    help='Dev-mode: N=32, fewer AL iterations (fast but lower quality)')
    ap.add_argument('--bench-tv',  action='store_true',
                    help='Run TV latency benchmark (requires real TV methods)')
    args = ap.parse_args()

    golden_csv = OUT_DIR / 'golden_lap.csv'
    simple_csv = OUT_DIR / 'simple_tv.csv'
    mpqp_csv   = OUT_DIR / 'mpqp_tv.csv'

    # ── Step 1: solve or load ─────────────────────────────────────────────
    if args.skip_solve and golden_csv.exists():
        print(f"[Runner] Loading existing {golden_csv}")
        golden_df = pd.read_csv(golden_csv)
    else:
        print("[Runner] Loading FSG Autocross track …")
        track_arrays = load_track()
        golden_df    = run_solver(track_arrays, dev_mode=args.dev)
        golden_df.to_csv(golden_csv, index=False)
        print(f"[Runner] Saved → {golden_csv}  ({len(golden_df)} rows)")

    # ── Step 2: stubs ─────────────────────────────────────────────────────
    laps_for_pipeline = [('Golden', str(golden_csv), True)]

    if not args.no_stubs:
        simple_df = make_stub_simple_tv(golden_df)
        mpqp_df   = make_stub_mpqp_tv(golden_df)
        simple_df.to_csv(simple_csv, index=False)
        mpqp_df.to_csv(mpqp_csv,   index=False)
        print(f"[Runner] Saved stubs → {simple_csv}, {mpqp_csv}")
        laps_for_pipeline += [
            ('Simple', str(simple_csv), False),
            ('mpQP',   str(mpqp_csv),   False),
        ]

    # ── Step 3: pipeline ──────────────────────────────────────────────────
    from telemetry.validation_pipeline import ValidationPipeline, MoTeCTheme, load_csv

    theme   = MoTeCTheme()
    pipe    = ValidationPipeline(theme=theme)
    palette = iter(theme.cmp)

    for name, path, is_ref in laps_for_pipeline:
        df = load_csv(path)
        pipe.add_lap(name, df, color=next(palette), is_reference=is_ref)

    print(f"\n[Runner] Rendering {len(pipe.laps)} lap(s) → {FIGS_DIR}/")
    report = pipe.render_all(
        out_dir=str(FIGS_DIR),
        report_path=str(REP_DIR / 'validation_report.json'),
        bench_tv=args.bench_tv,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  VALIDATION COMPLETE")
    print("═" * 60)
    for lap in report['laps']:
        t_str = f"{lap['lap_time']:.3f}s" if lap['lap_time'] else "N/A"
        print(f"  {lap['name']:12s}  t={t_str:>8}  "
              f"v_max={lap['v_max_kph']:.1f}km/h  "
              f"G_max={lap['g_combined_max']:.3f}  "
              f">0.9μ={lap['pct_above_p9_mu']:.1f}%")
    print(f"\n  Figures → {FIGS_DIR}/")
    print(f"  Report  → {REP_DIR}/validation_report.json")
    print("═" * 60)


if __name__ == '__main__':
    main()