"""
simulator/telemetry_analysis.py
─────────────────────────────────────────────────────────────────────────────
Post-session analysis of telemetry CSV logs.

Usage:
    python telemetry_analysis.py logs/telemetry_20260304_120000.csv
    python telemetry_analysis.py logs/telemetry_20260304_120000.csv --compare logs/telemetry_20260304_115000.csv

Output:
    · Session summary (lap times, peak G, peak speed, energy)
    · Corner analysis (per-corner entry/apex/exit speeds + slip angles)
    · Setup sensitivity report (correlation of setup params with lap time)
    · PNG exports of key plots (requires matplotlib)
    · W&B upload (optional)
"""

import os
import sys
import argparse
import csv
from typing import Optional, List, Dict, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[Warning] pandas not installed. Basic analysis only.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────

def load_telemetry(path: str) -> Optional['pd.DataFrame']:
    if not HAS_PANDAS:
        print("[Analysis] pandas required for full analysis.")
        return None
    try:
        df = pd.read_csv(path)
        print(f"[Analysis] Loaded {len(df)} frames from {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"[Analysis] Failed to load {path}: {e}")
        return None


def session_summary(df: 'pd.DataFrame') -> Dict:
    """Compute key session statistics."""
    if df is None or len(df) == 0:
        return {}

    # Lap times
    lap_changes = df['lap_number'].diff().fillna(0) != 0
    lap_starts  = df.index[lap_changes].tolist()

    lap_times = []
    for i, start_idx in enumerate(lap_starts):
        end_idx = lap_starts[i+1] if i+1 < len(lap_starts) else len(df)-1
        lap_df  = df.iloc[start_idx:end_idx]
        if len(lap_df) > 10:
            duration = float(lap_df['sim_time'].iloc[-1] - lap_df['sim_time'].iloc[0])
            lap_times.append(duration)

    # Peak values
    peak_lat_g  = float(df['lat_g'].abs().max())
    peak_lon_g  = float(df['lon_g'].abs().max())
    peak_speed  = float(df['speed_kmh'].max())
    total_energy = float(df['energy_kj'].max())

    # Tire temperature summary
    T_cols = ['T_fl', 'T_fr', 'T_rl', 'T_rr']
    T_means = {c: float(df[c].mean()) for c in T_cols if c in df.columns}
    T_maxes = {c: float(df[c].max())  for c in T_cols if c in df.columns}

    # Grip utilisation
    util_f_mean = float(df['grip_util_f'].mean()) if 'grip_util_f' in df.columns else 0.0
    util_r_mean = float(df['grip_util_r'].mean()) if 'grip_util_r' in df.columns else 0.0

    summary = {
        'n_laps'        : len(lap_times),
        'lap_times'     : lap_times,
        'best_lap'      : min(lap_times) if lap_times else 0.0,
        'mean_lap'      : np.mean(lap_times) if lap_times else 0.0,
        'peak_lat_g'    : peak_lat_g,
        'peak_lon_g'    : peak_lon_g,
        'peak_speed_kmh': peak_speed,
        'total_energy_kj': total_energy,
        'T_fl_mean'     : T_means.get('T_fl', 0.0),
        'T_rr_mean'     : T_means.get('T_rr', 0.0),
        'T_fl_max'      : T_maxes.get('T_fl', 0.0),
        'T_rr_max'      : T_maxes.get('T_rr', 0.0),
        'grip_util_f'   : util_f_mean,
        'grip_util_r'   : util_r_mean,
    }
    return summary


def print_summary(summary: Dict, label: str = "Session"):
    print(f"\n{'='*60}")
    print(f"  {label} Summary")
    print(f"{'='*60}")
    if not summary:
        print("  No data.")
        return

    def fmt_t(t): return f"{int(t//60):02d}:{t%60:06.3f}"

    print(f"  Laps completed: {summary['n_laps']}")
    if summary['lap_times']:
        print(f"  Best lap:  {fmt_t(summary['best_lap'])}")
        print(f"  Mean lap:  {fmt_t(summary['mean_lap'])}")
        for i, lt in enumerate(summary['lap_times']):
            star = " ★" if lt == summary['best_lap'] else ""
            print(f"  Lap {i+1:2d}:   {fmt_t(lt)}{star}")

    print(f"\n  Peak lateral G:      {summary['peak_lat_g']:.3f} G")
    print(f"  Peak longitudinal G: {summary['peak_lon_g']:.3f} G")
    print(f"  Peak speed:          {summary['peak_speed_kmh']:.1f} km/h")
    print(f"  Total energy:        {summary['total_energy_kj']:.1f} kJ")
    print(f"\n  Tire temps (mean):   FL {summary['T_fl_mean']:.1f}°C  RR {summary['T_rr_mean']:.1f}°C")
    print(f"  Tire temps (peak):   FL {summary['T_fl_max']:.1f}°C   RR {summary['T_rr_max']:.1f}°C")
    print(f"\n  Mean grip util: Front {summary['grip_util_f']:.3f}  Rear {summary['grip_util_r']:.3f}")


def plot_session(df: 'pd.DataFrame', output_dir: str = '.', label: str = 'session'):
    """Generate diagnostic plots."""
    if not HAS_MATPLOTLIB or df is None:
        print("[Analysis] matplotlib not available — skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    t = df['sim_time'].values

    # ── Figure 1: Speed + G-force trace ─────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Project-GP Telemetry — {label}', fontsize=14)

    axes[0].plot(t, df['speed_kmh'], 'b-', linewidth=0.8, label='Speed [km/h]')
    axes[0].set_ylabel('Speed (km/h)'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t, df['lat_g'],  'r-',  linewidth=0.8, label='Lat G')
    axes[1].plot(t, df['lon_g'],  'g-',  linewidth=0.8, label='Lon G')
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set_ylabel('Acceleration (G)'); axes[1].legend(); axes[1].grid(alpha=0.3)

    for col, color, lbl in [('T_fl', 'blue', 'FL'), ('T_fr', 'cyan', 'FR'),
                              ('T_rl', 'red',  'RL'), ('T_rr', 'orange', 'RR')]:
        if col in df.columns:
            axes[2].plot(t, df[col], color=color, linewidth=0.8, label=f'T_{lbl}')
    axes[2].axhline(80, color='green', linestyle='--', alpha=0.5, label='Optimal')
    axes[2].axhline(110, color='red', linestyle='--', alpha=0.5, label='Hot')
    axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Tire temp (°C)')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(output_dir, f'{label}_speed_g_temp.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis] Saved → {path1}")

    # ── Figure 2: G-G diagram ─────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    lat_g = df['lat_g'].values; lon_g = df['lon_g'].values
    speed = df['speed_kmh'].values
    sc = ax.scatter(lat_g, lon_g, c=speed, cmap='plasma',
                    s=1.0, alpha=0.5, vmin=0, vmax=100)
    circle = plt.Circle((0, 0), 1.45, fill=False, color='gray',
                          linestyle='--', linewidth=1.5, label='μ=1.45')
    ax.add_patch(circle)
    plt.colorbar(sc, ax=ax, label='Speed (km/h)')
    ax.set_xlabel('Lateral G'); ax.set_ylabel('Longitudinal G')
    ax.set_title('G-G Diagram'); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.5, 1.5)
    path2 = os.path.join(output_dir, f'{label}_gg_diagram.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis] Saved → {path2}")

    # ── Figure 3: Suspension & wheel loads ───────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle('Suspension & Wheel Loads')
    for col, color, lbl in [('z_fl','blue','FL'),('z_fr','cyan','FR'),
                              ('z_rl','red','RL'), ('z_rr','orange','RR')]:
        if col in df.columns:
            axes[0].plot(t, df[col] - 0.30, color=color, linewidth=0.8, label=lbl)
    axes[0].set_ylabel('Heave travel (m)'); axes[0].legend(); axes[0].grid(alpha=0.3)

    for col, color, lbl in [('Fz_fl','blue','FL'),('Fz_fr','cyan','FR'),
                              ('Fz_rl','red','RL'),('Fz_rr','orange','RR')]:
        if col in df.columns:
            axes[1].plot(t, df[col], color=color, linewidth=0.8, label=lbl)
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Wheel load (N)')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(output_dir, f'{label}_suspension.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis] Saved → {path3}")

    # ── Figure 4: XY trajectory coloured by speed ────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(df['x'], df['y'], c=df['speed_kmh'], cmap='jet',
               s=1.5, alpha=0.7, vmin=0, vmax=100)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Track Trajectory (coloured by speed)'); ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    path4 = os.path.join(output_dir, f'{label}_trajectory.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis] Saved → {path4}")

    print(f"\n[Analysis] All plots saved to {output_dir}/")


def compare_sessions(df1: 'pd.DataFrame', df2: 'pd.DataFrame',
                     label1: str = 'A', label2: str = 'B',
                     output_dir: str = '.'):
    """
    Side-by-side comparison of two telemetry sessions.
    Useful for setup comparison (e.g., soft vs stiff springs).
    """
    if not HAS_MATPLOTLIB:
        return

    s1 = session_summary(df1)
    s2 = session_summary(df2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Setup Comparison: {label1} vs {label2}', fontsize=14)

    # Speed overlay
    axes[0,0].plot(df1['sim_time'], df1['speed_kmh'],
                   'b-', linewidth=0.8, alpha=0.7, label=label1)
    axes[0,0].plot(df2['sim_time'], df2['speed_kmh'],
                   'r-', linewidth=0.8, alpha=0.7, label=label2)
    axes[0,0].set_title('Speed'); axes[0,0].set_ylabel('km/h')
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    # Lateral G overlay
    axes[0,1].plot(df1['sim_time'], df1['lat_g'],
                   'b-', linewidth=0.8, alpha=0.7, label=label1)
    axes[0,1].plot(df2['sim_time'], df2['lat_g'],
                   'r-', linewidth=0.8, alpha=0.7, label=label2)
    axes[0,1].set_title('Lateral G'); axes[0,1].set_ylabel('G')
    axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    # G-G diagrams
    for ax, df, label, color in [(axes[1,0], df1, label1, 'blue'),
                                   (axes[1,1], df2, label2, 'red')]:
        ax.scatter(df['lat_g'], df['lon_g'], c=color, s=0.8, alpha=0.4)
        circle = plt.Circle((0, 0), 1.45, fill=False, color='gray',
                              linestyle='--', linewidth=1.5)
        ax.add_patch(circle)
        best = s1['best_lap'] if label == label1 else s2['best_lap']
        ax.set_title(f'G-G {label} (best: {best:.3f}s)')
        ax.set_xlim(-2, 2); ax.set_ylim(-2.5, 1.5)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'comparison_{label1}_vs_{label2}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis] Comparison saved → {path}")

    # Delta report
    print(f"\n{'='*50}")
    print(f"  Delta: {label1} vs {label2}")
    print(f"{'='*50}")
    if s1['best_lap'] > 0 and s2['best_lap'] > 0:
        delta = s2['best_lap'] - s1['best_lap']
        print(f"  Best lap delta:    {delta:+.3f}s  ({'A faster' if delta>0 else 'B faster'})")
    if s1['peak_lat_g'] > 0:
        print(f"  Peak lat G delta:  {s2['peak_lat_g']-s1['peak_lat_g']:+.3f} G")
    print(f"  Energy delta:      {s2['total_energy_kj']-s1['total_energy_kj']:+.1f} kJ")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Project-GP Telemetry Analysis')
    parser.add_argument('log',          help='Primary telemetry CSV path')
    parser.add_argument('--compare',    default=None, help='Second CSV for comparison')
    parser.add_argument('--output-dir', default='analysis_output')
    parser.add_argument('--label',      default='session')
    parser.add_argument('--label2',     default='session_b')
    args = parser.parse_args()

    df = load_telemetry(args.log)
    if df is None:
        sys.exit(1)

    summary = session_summary(df)
    print_summary(summary, label=args.label)
    plot_session(df, output_dir=args.output_dir, label=args.label)

    if args.compare:
        df2 = load_telemetry(args.compare)
        if df2 is not None:
            summary2 = session_summary(df2)
            print_summary(summary2, label=args.label2)
            compare_sessions(df, df2, args.label, args.label2, args.output_dir)


if __name__ == '__main__':
    main()
