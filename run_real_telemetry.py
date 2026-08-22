#!/usr/bin/env python3
# run_real_telemetry.py
# Project-GP — Real Telemetry Validation & Setup Optimisation Entry-Point
# ═══════════════════════════════════════════════════════════════════════════════
#
# USAGE
# ─────
#   Validate only (print fidelity report + save MoTeC-style dashboard):
#     python run_real_telemetry.py --session data/raw_can_logs/2.csv --validate-only
#
#   Full MORL-SB-TRPO optimisation grounded in real telemetry:
#     python run_real_telemetry.py --session data/raw_can_logs/2.csv --optimize
#
#   Use the best extracted lap only + 300 optimizer iterations:
#     python run_real_telemetry.py --session data/raw_can_logs/2.csv \
#                                  --optimize --best-lap --iterations 300
#
#   Override GPS reference (lat, lon) for lap detection:
#     python run_real_telemetry.py --session data/raw_can_logs/2.csv \
#                                  --validate-only --ref-lat 48.1234 --ref-lon 11.5678
#
# OUTPUTS
# ───────
#   reports/twin_fidelity/twin_fidelity_report.json   — per-channel R², NRMSE, lag
#   figs/real_telemetry_dashboard.png                 — 6-panel MoTeC-style overlay
#   reports/setup_recommendations.json               — Pareto setups (optimize mode)
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# §1  Helpers
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def _ensure_dirs():
    for d in ("reports/twin_fidelity", "figs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def _soft_r2(sim: np.ndarray, real: np.ndarray) -> float:
    """NumPy soft R² for reporting (mirrors the JAX objective)."""
    mse    = np.mean((sim - real) ** 2)
    var    = np.mean((real - np.mean(real)) ** 2) + 1e-6
    r2_raw = 1.0 - mse / var
    return float(np.clip(r2_raw, -1.0, 1.0))   # raw (not sigmoid) for reporting


def _nrmse(sim: np.ndarray, real: np.ndarray) -> float:
    rng = np.ptp(real)
    if rng < 1e-9:
        return 0.0
    return float(np.sqrt(np.mean((sim - real) ** 2)) / rng * 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest_session(
    session_path: str,
    ref_lat: float | None,
    ref_lon: float | None,
    best_lap_only: bool,
) -> tuple[dict, dict, object]:
    """
    Read a CAN CSV session and return (controls, measurements, lap_data).

    If best_lap_only=True, extract laps and return the best (longest) one.
    Otherwise return the full session.
    """
    from telemetry.can_log_reader import CANLogReader
    from telemetry.lap_extractor  import LapExtractor

    print(f"\n[Pipeline] ─── Ingesting: {session_path}")
    reader = CANLogReader(session_path)
    df     = reader.load()

    if best_lap_only:
        # Convert GPS ref from (lat, lon) to local Cartesian if provided
        ref_xy = None
        if ref_lat is not None and ref_lon is not None:
            import math
            # Same flat-Earth projection as CANLogReader._add_cartesian
            lon0 = df["lon"].dropna().iloc[0]
            lat0 = df["lat"].dropna().iloc[0]
            DEG_TO_M_LAT = 111_320.0
            m_per_deg_lon = DEG_TO_M_LAT * math.cos(math.radians(lat0))
            ref_xy = ((ref_lon - lon0) * m_per_deg_lon,
                      (ref_lat - lat0) * DEG_TO_M_LAT)
            print(f"[Pipeline] GPS reference → local ({ref_xy[0]:.1f}, {ref_xy[1]:.1f}) m")

        extractor = LapExtractor(df, reference_xy=ref_xy)
        best = extractor.best_lap()
        if best is None:
            warnings.warn(
                "[Pipeline] No valid laps extracted. Falling back to full session.",
                stacklevel=2)
            controls     = reader.get_controls()
            measurements = reader.get_measurements()
            lap_data = None
        else:
            controls     = best.controls
            measurements = best.measurements
            lap_data     = best
            print(f"[Pipeline] Best lap selected: #{best.lap_index}  "
                  f"{best.duration_s:.1f}s  {best.distance_m:.0f}m")
    else:
        controls     = reader.get_controls()
        measurements = reader.get_measurements()
        lap_data = None

    return controls, measurements, lap_data


# ─────────────────────────────────────────────────────────────────────────────
# §3  Open-loop validation (NumPy/model forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(controls: dict, measurements: dict) -> dict:
    """
    Drive the digital twin open-loop with real controls and compute fidelity.

    Returns a fidelity report dict with the following structure:
    {
       "twin_fidelity_pct": float,      # composite 0–100 score
       "channels": {
           "<name>": {
               "r2": float,
               "nrmse_pct": float,
               "lag_ms": float,         # cross-correlation lag
           }
       }
    }
    """
    print("\n[Pipeline] ─── Running open-loop validation …")
    t0 = time.perf_counter()

    # ── Import JAX inside the function so --validate-only still works even
    #    without the full JAX GPU stack (CPU fallback)
    import jax
    import jax.numpy as jnp
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle,
        compute_equilibrium_suspension,
        make_setup_from_params,          # builds 28-vector from vehicle_params dict
    )
    from config.vehicles.ter26       import vehicle_params as VP
    from config.tire_coeffs          import tire_coeffs   as TC

    vehicle    = DifferentiableMultiBodyVehicle(VP, TC)
    setup_phys = make_setup_from_params(VP).to_vector()   # (28,) float32

    z_eq   = compute_equilibrium_suspension(setup_phys, VP)
    x_init = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=float(measurements["speed"][0]))
    x_init = x_init.at[6:10].set(z_eq)

    N = len(controls["steer"])
    dt_val = float(controls["dt"][0])
    F_DRIVE_MAX = 3000.0
    F_BRAKE_MAX = 4000.0
    R_WHEEL     = 0.2045

    sim_speed    = np.zeros(N, dtype=np.float32)
    sim_yaw_rate = np.zeros(N, dtype=np.float32)
    sim_ay       = np.zeros(N, dtype=np.float32)

    state = x_init
    print(f"[Pipeline]   Simulating {N} steps ({N*dt_val:.1f} s) …")

    for i in range(N):
        delta = float(controls["steer"][i])
        thr   = float(controls["throttle"][i])
        brk   = float(controls["brake"][i])

        F_lon    = thr * F_DRIVE_MAX
        F_hyd    = brk * F_BRAKE_MAX
        T_wheel  = F_lon * R_WHEEL / 4.0
        u = jnp.array([delta, T_wheel, T_wheel, T_wheel, T_wheel, F_hyd])

        state = vehicle.simulate_step(state, u, setup_phys, dt_val)

        sim_speed[i]    = float(state[14])
        sim_yaw_rate[i] = float(state[19])
        sim_ay[i]       = float(state[14]) * float(state[19])

        if i % 500 == 0 and i > 0:
            print(f"[Pipeline]   … {i}/{N} steps ({i*dt_val:.1f}s)")

    # ── Compute per-channel metrics ───────────────────────────────────────────
    def xcorr_lag(sim: np.ndarray, real: np.ndarray, dt: float) -> float:
        """Cross-correlation lag in ms."""
        corr = np.correlate(sim - sim.mean(), real - real.mean(), mode='full')
        lag_idx = int(np.argmax(corr)) - (len(real) - 1)
        return round(lag_idx * dt * 1000.0, 1)   # ms

    channels_report = {}
    pairs = [
        ("speed",    sim_speed,    measurements["speed"]),
        ("yaw_rate", sim_yaw_rate, measurements["yaw_rate"]),
        ("ay",       sim_ay,       measurements["ay"]),
    ]
    fid_components = []
    weights = [0.50, 0.30, 0.20]

    for (name, sim, real), w in zip(pairs, weights):
        r2  = _soft_r2(sim, real)
        nrm = _nrmse(sim, real)
        lag = xcorr_lag(sim, real, dt_val)
        channels_report[name] = {"r2": round(r2, 4), "nrmse_pct": round(nrm, 2), "lag_ms": lag}
        fid_components.append(w * max(0.0, r2))

    twin_fidelity_pct = float(np.clip(sum(fid_components), 0.0, 1.0)) * 100.0
    elapsed = time.perf_counter() - t0

    report = {
        "twin_fidelity_pct": round(twin_fidelity_pct, 2),
        "channels": channels_report,
        "n_steps": N,
        "duration_s": round(N * dt_val, 1),
        "elapsed_wall_s": round(elapsed, 1),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n[Pipeline] ── Validation complete in {elapsed:.1f}s ──")
    print(f"  Twin Fidelity:  {twin_fidelity_pct:.1f}%")
    for name, c in channels_report.items():
        print(f"  {name:<12}: R²={c['r2']:+.3f}  NRMSE={c['nrmse_pct']:.1f}%  "
              f"lag={c['lag_ms']:+.1f}ms")

    return report, {"speed": sim_speed, "yaw_rate": sim_yaw_rate, "ay": sim_ay}


# ─────────────────────────────────────────────────────────────────────────────
# §4  Dashboard figure
# ─────────────────────────────────────────────────────────────────────────────

def save_dashboard(
    controls: dict,
    measurements: dict,
    sim_channels: dict,
    report: dict,
    out_path: Path,
):
    """6-panel MoTeC-style overlay: real (blue) vs sim (red)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        warnings.warn("[Pipeline] matplotlib not available — dashboard skipped.")
        return

    N  = len(measurements["speed"])
    dt = float(controls["dt"][0])
    t  = np.arange(N) * dt

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    _REAL = "#4fc3f7"   # cyan
    _SIM  = "#ef5350"   # red
    _BG   = "#161b22"
    _TXT  = "#e6edf3"
    _GRID = "#30363d"

    channel_pairs = [
        ("Speed [m/s]",    measurements["speed"],    sim_channels["speed"],    0, 0),
        ("Yaw Rate [rad/s]", measurements["yaw_rate"], sim_channels["yaw_rate"], 0, 1),
        ("Lat. Accel [m/s²]", measurements["ay"],     sim_channels["ay"],       1, 0),
        ("Throttle [0-1]", controls["throttle"],     None,                     1, 1),
        ("Brake [0-1]",    controls["brake"],        None,                     2, 0),
        ("Steer [rad]",    controls["steer"],        None,                     2, 1),
    ]

    for label, real_data, sim_data, row, col in channel_pairs:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_TXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, lw=0.5, alpha=0.6)

        ax.plot(t, real_data, color=_REAL, lw=0.8, label="Real", alpha=0.9)
        if sim_data is not None:
            ax.plot(t, sim_data, color=_SIM,  lw=0.8, label="Sim",  alpha=0.9)

        ax.set_title(label, color=_TXT, fontsize=9, pad=4)
        ax.set_xlabel("Time [s]", color=_TXT, fontsize=7)
        ax.legend(loc="upper right", fontsize=7,
                  facecolor=_BG, edgecolor=_GRID, labelcolor=_TXT)

    # Title with fidelity score
    fid = report["twin_fidelity_pct"]
    fig.suptitle(
        f"TeR-Q Digital Twin vs Real Telemetry   |   Fidelity: {fid:.1f}%   "
        f"| Session duration: {report['duration_s']:.0f}s",
        color=_TXT, fontsize=12, fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Pipeline] Dashboard saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# §5  Setup optimisation (telemetry phase)
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(controls: dict, measurements: dict, iterations: int) -> dict:
    """Wire telemetry into MORL-SB-TRPO and run N optimizer iterations."""
    print(f"\n[Pipeline] ─── Starting MORL-SB-TRPO optimisation "
          f"({iterations} iterations, phase='telemetry') …")

    from optimization.evolutionary import MORL_SB_TRPO_Optimizer

    optimizer = MORL_SB_TRPO_Optimizer()
    optimizer.load_telemetry(controls, measurements)

    t0 = time.perf_counter()
    optimizer.run(n_iterations=iterations, phase='telemetry')
    elapsed = time.perf_counter() - t0
    print(f"[Pipeline] Optimisation complete in {elapsed/60:.1f} min")

    # Collect Pareto archive
    results = []
    for i, (setup, grip, stab) in enumerate(
        zip(optimizer.archive_setups,
            optimizer.archive_grips,
            optimizer.archive_stabs)
    ):
        results.append({
            "rank": i,
            "grip": round(float(grip), 4),
            "stab": round(float(stab), 4),
            "setup_28": [round(float(v), 6) for v in setup],
        })

    # Sort by grip (primary)
    results.sort(key=lambda r: r["grip"], reverse=True)
    print(f"[Pipeline] Pareto archive: {len(results)} setups")
    if results:
        best = results[0]
        print(f"[Pipeline] Best grip: {best['grip']:.4f}  stab: {best['stab']:.4f}")
    return {"pareto_setups": results, "elapsed_min": round(elapsed / 60.0, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# §6  main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TeR-Q Digital Twin — Real CAN Telemetry Validation & Setup Optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session", required=True,
                        help="Path to CAN CSV session (e.g. data/raw_can_logs/2.csv)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run validation only; skip setup optimisation")
    parser.add_argument("--optimize", action="store_true",
                        help="Run MORL-SB-TRPO setup optimisation after validation")
    parser.add_argument("--best-lap", action="store_true",
                        help="Extract laps and use only the best one")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of MORL optimizer iterations")
    parser.add_argument("--ref-lat", type=float, default=None,
                        help="Latitude of start/finish line (for lap detection)")
    parser.add_argument("--ref-lon", type=float, default=None,
                        help="Longitude of start/finish line (for lap detection)")
    parser.add_argument("--report-dir", default="reports/twin_fidelity",
                        help="Directory for JSON reports")
    parser.add_argument("--fig-dir", default="figs",
                        help="Directory for figure output")
    args = parser.parse_args()

    _ensure_dirs()

    # ── §6.1 Ingestion ────────────────────────────────────────────────────────
    controls, measurements, lap_data = ingest_session(
        session_path=args.session,
        ref_lat=args.ref_lat,
        ref_lon=args.ref_lon,
        best_lap_only=args.best_lap,
    )

    # ── §6.2 Open-loop validation ─────────────────────────────────────────────
    fidelity_report, sim_channels = run_validation(controls, measurements)

    # Save fidelity JSON
    report_path = ROOT / args.report_dir / "twin_fidelity_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(fidelity_report, f, indent=2)
    print(f"[Pipeline] Fidelity report → {report_path}")

    # Save dashboard PNG
    dash_path = ROOT / args.fig_dir / "real_telemetry_dashboard.png"
    save_dashboard(controls, measurements, sim_channels, fidelity_report, dash_path)

    # ── §6.3 Optimisation (optional) ──────────────────────────────────────────
    if args.optimize and not args.validate_only:
        opt_results = run_optimization(controls, measurements, args.iterations)
        opt_path = ROOT / args.report_dir / "setup_recommendations.json"
        with open(opt_path, "w") as f:
            json.dump(opt_results, f, indent=2)
        print(f"[Pipeline] Setup recommendations → {opt_path}")
    elif args.validate_only:
        print("\n[Pipeline] --validate-only mode: skipping optimisation.")
    else:
        print("\n[Pipeline] Pass --optimize to run setup optimisation.")

    print("\n[Pipeline] ✓ Done.")


if __name__ == "__main__":
    main()
