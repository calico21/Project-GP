"""
telemetry/validation_pipeline.py
─────────────────────────────────
Project-GP Telemetry Validation & Visualization Pipeline.

Consumes CSV outputs from `optimization/ocp_solver.py` (Diff-WMPC Golden Lap)
and `differentiable_lap_sim.py` (Simple-TV / mpQP-TV / WMPC-TV variants).
Produces publication-grade comparative plots plus a real-time TV-controller
latency benchmark.

Generated artifacts (relative to --out, default ./figs/):
  motec_dashboard_<variant>.png    6-panel synchronised time-series
  track_map_yaw.png                Top-down trajectory ribbon, yaw-error coloured
  track_map_temp.png               Top-down ribbon, tire-temperature coloured
  track_map_latency.png            Top-down ribbon, per-step TV latency coloured
  gg_diagram.png                   Friction circle with KDE density + confidence ellipses
  sector_delta.png                 Per-sector time delta vs reference (BONUS)
  phase_portrait.png               β–r stability diagram (BONUS)
  steering_fft.png                 Steering-channel power spectrum (BONUS)
  power_flow.png                   Battery / drive / regen / loss budget (BONUS)
  yaw_tracking.png                 Reference vs actual yaw rate, error band (BONUS)
  latency_histogram.png            Per-method p50/p95/p99 (BONUS)
reports/validation_report.json     Machine-readable metrics summary

CLI:
    python -m telemetry.validation_pipeline \\
        --golden  out/golden_lap.csv      \\
        --simple  out/simple_tv.csv       \\
        --mpqp    out/mpqp_tv.csv         \\
        --wmpc    out/wmpc_tv.csv         \\
        --human   out/human_lap.csv       \\
        --bench-tv                                # opt-in real-time benchmark

Programmatic:
    pipe = ValidationPipeline(theme=MoTeCTheme())
    pipe.add_lap('Golden', golden_df, color='#00D4FF', is_reference=True)
    pipe.add_lap('mpQP',   mpqp_df,   color='#22D67A')
    pipe.render_all(out_dir='./figs')
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from scipy.stats import gaussian_kde

# ════════════════════════════════════════════════════════════════════════════
# §1  Theme & canonical schema
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MoTeCTheme:
    """Dark telemetry palette. Hue spacing tuned for distinguishability against
    bg=#0A0E14 with WCAG-AA legibility on 9-pt sans labels."""
    bg:    str = '#0A0E14'
    panel: str = '#0F1620'
    fg:    str = '#E0E6ED'
    muted: str = '#6B7280'
    grid:  str = '#1A2332'

    speed:    str = '#00D4FF'
    throttle: str = '#22D67A'
    brake:    str = '#FF3344'
    steer:    str = '#FFB300'
    regen:    str = '#B266FF'
    grip:     str = '#FF6F00'
    ref:      str = '#FFFFFF'

    # TV split: cool-side (left) / warm-side (right); light = front, deep = rear.
    # Encodes BOTH axle and side in colour alone, leaving linestyle free.
    tv_fl: str = '#7CCBFF'
    tv_fr: str = '#FFD27A'
    tv_rl: str = '#0099CC'
    tv_rr: str = '#E64B26'

    cmp: tuple = ('#00D4FF', '#FF3344', '#22D67A', '#FFB300', '#B266FF')

    def apply(self) -> None:
        plt.rcParams.update({
            'figure.facecolor':  self.bg,
            'axes.facecolor':    self.panel,
            'savefig.facecolor': self.bg,
            'savefig.dpi':       150,
            'axes.edgecolor':    self.grid,
            'axes.labelcolor':   self.fg,
            'axes.titlecolor':   self.fg,
            'xtick.color':       self.fg,
            'ytick.color':       self.fg,
            'text.color':        self.fg,
            'grid.color':        self.grid,
            'grid.alpha':        0.45,
            'grid.linewidth':    0.5,
            'axes.grid':         True,
            'axes.spines.top':   False,
            'axes.spines.right': False,
            'axes.spines.left':  True,
            'axes.spines.bottom': True,
            'axes.linewidth':    0.6,
            'font.family':       'sans-serif',
            'font.size':         9,
            'axes.titlesize':    10,
            'axes.titleweight':  'bold',
            'legend.frameon':    False,
            'legend.fontsize':   8,
            'lines.linewidth':   1.4,
            'lines.antialiased': True,
        })


# Tolerant column resolution — upstream sources use varying spellings.
_ALIASES: Mapping[str, tuple[str, ...]] = {
    's':         ('s', 'distance', 'arc_length', 'lap_distance'),
    't':         ('t', 'time', 'lap_time'),
    'x':         ('x', 'pos_x', 'X'),
    'y':         ('y', 'pos_y', 'Y'),
    'psi':       ('psi', 'yaw', 'heading'),
    'v':         ('v', 'speed', 'velocity'),
    'vx':        ('vx', 'vel_x'),
    'vy':        ('vy', 'vel_y'),
    'wz':        ('wz', 'yaw_rate', 'r', 'omega_z'),
    'steer':     ('steer', 'delta', 'steering_angle'),
    'throttle':  ('throttle', 'thr', 'pedal_throttle'),
    'brake':     ('brake', 'brk', 'pedal_brake'),
    'accel':     ('accel', 'force', 'u_long'),
    'latG':      ('latG', 'lat_g', 'a_y', 'ay'),
    'lonG':      ('lonG', 'lon_g', 'a_x', 'ax'),
    'kappa':     ('kappa', 'curvature', 'k'),
    'T_fl':      ('T_fl', 'torque_fl', 'tau_fl'),
    'T_fr':      ('T_fr', 'torque_fr', 'tau_fr'),
    'T_rl':      ('T_rl', 'torque_rl', 'tau_rl'),
    'T_rr':      ('T_rr', 'torque_rr', 'tau_rr'),
    'regen':     ('regen', 'alpha_regen', 'regen_alpha'),
    'T_tire_fl': ('T_tire_fl', 'tire_temp_fl', 'temp_fl'),
    'T_tire_fr': ('T_tire_fr', 'tire_temp_fr', 'temp_fr'),
    'T_tire_rl': ('T_tire_rl', 'tire_temp_rl', 'temp_rl'),
    'T_tire_rr': ('T_tire_rr', 'tire_temp_rr', 'temp_rr'),
    'beta':      ('beta', 'sideslip'),
    'P_battery': ('P_battery', 'p_bat', 'P_bat'),
    'wz_ref':    ('wz_ref', 'yaw_rate_ref', 'r_ref'),
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {alias: canon for canon, aliases in _ALIASES.items()
              for alias in aliases if alias in df.columns and alias != canon}
    return df.rename(columns=rename) if rename else df.copy()


# ════════════════════════════════════════════════════════════════════════════
# §2  Domain model — self-healing LapData
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LapData:
    """Telemetry container with optional-channel introspection.

    Performs schema normalisation and lazy derivation of missing channels:
      · (x, y) integrated from (psi, v, dt) when absent
      · throttle/brake split from a signed `accel` channel when absent
      · combinedG = √(latG² + lonG²) always derived
      · kappa from (wz, v) when absent
      · beta from (vx, vy) when absent
    """
    name: str
    df:   pd.DataFrame
    color: str = '#00D4FF'
    is_reference: bool = False
    style: str = '-'

    def __post_init__(self) -> None:
        self.df = _normalise_columns(self.df).reset_index(drop=True).copy()
        self._derive_missing()

    # ── derivations ──────────────────────────────────────────────────────────
    def _derive_missing(self) -> None:
        d = self.df

        if 's' not in d:
            if 'x' in d and 'y' in d:
                d['s'] = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(d['x']), np.diff(d['y'])))])
            else:
                raise ValueError(f"Lap '{self.name}' lacks both 's' and (x,y) — cannot infer arc length.")

        if 'v' not in d and 'vx' in d:
            d['v'] = np.hypot(d['vx'], d.get('vy', 0.0))

        if 'kappa' not in d and {'wz', 'v'}.issubset(d.columns):
            # κ = ω_z / v with floor to suppress division noise at standstill
            d['kappa'] = d['wz'] / np.maximum(d['v'], 0.5)

        if 'latG' not in d and {'v', 'kappa'}.issubset(d.columns):
            d['latG'] = (d['v'] ** 2) * d['kappa'] / 9.81

        if 'lonG' not in d:
            if 't' in d and 'v' in d and len(d) > 1:
                dt = np.diff(d['t'].to_numpy(), prepend=d['t'].iloc[0])
                dt = np.where(dt > 1e-6, dt, 1.0)
                d['lonG'] = np.gradient(d['v'].to_numpy(), edge_order=2) / np.maximum(dt, 1e-3) / 9.81
            else:
                d['lonG'] = 0.0

        d['combinedG'] = np.hypot(d.get('latG', 0.0), d.get('lonG', 0.0))

        # Throttle/brake split from signed accel (Newton-units convention from
        # ocp_solver.py: positive=drive, negative=brake). Normalise to [0,1].
        if 'throttle' not in d and 'accel' in d:
            f = d['accel'].to_numpy()
            f_max = max(float(np.max(np.abs(f))), 1.0)
            d['throttle'] = np.clip(f / f_max,  0.0, 1.0)
            d['brake']    = np.clip(-f / f_max, 0.0, 1.0)

        # Kinematic xy reconstruction from (psi, v, dt). Only when no spatial
        # channel is present — otherwise prefer recorded coordinates.
        if 'x' not in d or 'y' not in d:
            psi = d.get('psi', np.zeros(len(d))).to_numpy()
            v   = d['v'].to_numpy()
            ds  = np.diff(d['s'].to_numpy(), prepend=d['s'].iloc[0])
            x = np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1]) * ds[1:])])
            y = np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1]) * ds[1:])])
            d['x'] = x
            d['y'] = y

        if 'beta' not in d and {'vx', 'vy'}.issubset(d.columns):
            d['beta'] = np.arctan2(d['vy'], np.maximum(d['vx'], 0.5))

    # ── accessors ────────────────────────────────────────────────────────────
    def has(self, *cols: str) -> bool:
        return all(c in self.df.columns for c in cols)

    def get(self, col: str, default: float | np.ndarray | None = None) -> np.ndarray:
        if col in self.df.columns:
            return self.df[col].to_numpy()
        if default is None:
            raise KeyError(f"Lap '{self.name}' missing channel '{col}' with no default.")
        return np.full(len(self.df), default) if np.isscalar(default) else default

    @property
    def n(self) -> int: return len(self.df)

    @property
    def lap_time(self) -> float | None:
        return float(self.df['t'].iloc[-1] - self.df['t'].iloc[0]) if 't' in self.df else None


# ════════════════════════════════════════════════════════════════════════════
# §3  Ingestion
# ════════════════════════════════════════════════════════════════════════════

def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Telemetry CSV not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty CSV: {p}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# §4  MoTeC 6-panel time-series dashboard
# ════════════════════════════════════════════════════════════════════════════

def render_motec_dashboard(
    laps: Sequence[LapData],
    *,
    theme: MoTeCTheme,
    out_path: Path,
    title: str | None = None,
) -> None:
    """6-panel synchronised dashboard with distance on shared X.

    Panels: speed, signed pedal trace, steering, regen blend, TV split, grip-util.
    The signed pedal panel is the diagnostic key — coasting (zero band) is
    visible without computation; each pedal sits on its own sign half-plane so
    overlap on the time axis is impossible.
    """
    theme.apply()
    if not laps:
        raise ValueError("Need at least one lap to render dashboard.")

    fig = plt.figure(figsize=(13, 11))
    gs  = GridSpec(6, 1, figure=fig, hspace=0.18,
                   left=0.07, right=0.985, top=0.955, bottom=0.06)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(6)]

    for ax in axes[:-1]:
        ax.tick_params(axis='x', labelbottom=False)
    axes[-1].set_xlabel('Distance s [m]')

    # ── 1. Speed ─────────────────────────────────────────────────────────────
    ax = axes[0]
    for lap in laps:
        ax.plot(lap.get('s'), lap.get('v') * 3.6,
                color=lap.color, linestyle=lap.style,
                lw=1.6 if lap.is_reference else 1.2,
                alpha=1.0 if lap.is_reference else 0.85,
                label=lap.name)
    ax.set_ylabel('Speed\n[km/h]')
    ax.legend(loc='lower right', ncol=len(laps))

    # ── 2. Signed pedal trace ───────────────────────────────────────────────
    ax = axes[1]
    ref = next((l for l in laps if l.is_reference), laps[0])
    s_ref = ref.get('s')
    thr = ref.get('throttle', 0.0)
    brk = ref.get('brake',    0.0)
    ax.fill_between(s_ref,  thr, 0.0, color=theme.throttle, alpha=0.55, lw=0)
    ax.fill_between(s_ref, -brk, 0.0, color=theme.brake,    alpha=0.55, lw=0)
    # Coasting band: |throttle|<eps ∧ |brake|<eps → hatched neutral
    coast = (np.abs(thr) < 0.02) & (np.abs(brk) < 0.02)
    if coast.any():
        ax.fill_between(s_ref, -0.05, 0.05, where=coast,
                        color=theme.muted, alpha=0.35, lw=0,
                        label='Coast', step='mid')
    ax.axhline(0, color=theme.grid, lw=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f'Pedals\n(thr⁺/brk⁻)\n— {ref.name}')
    ax.legend(loc='upper right')

    # ── 3. Steering ─────────────────────────────────────────────────────────
    ax = axes[2]
    for lap in laps:
        if lap.has('steer'):
            ax.plot(lap.get('s'), np.degrees(lap.get('steer')),
                    color=lap.color, linestyle=lap.style,
                    lw=1.6 if lap.is_reference else 1.0,
                    alpha=1.0 if lap.is_reference else 0.8)
    ax.set_ylabel('Steering\n[deg]')
    ax.axhline(0, color=theme.grid, lw=0.4)

    # ── 4. Regen blend α* ───────────────────────────────────────────────────
    ax = axes[3]
    have_regen = False
    for lap in laps:
        if lap.has('regen'):
            ax.plot(lap.get('s'), np.clip(lap.get('regen'), 0.0, 1.0),
                    color=lap.color, lw=1.2, alpha=0.9)
            have_regen = True
    if not have_regen:
        ax.text(0.5, 0.5, 'no regen channel',
                transform=ax.transAxes, ha='center', va='center',
                color=theme.muted, fontsize=9, style='italic')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Regen α*\n[0–1]')

    # ── 5. TV split ─────────────────────────────────────────────────────────
    ax = axes[4]
    tv_lap = next((l for l in laps if l.has('T_fl', 'T_fr', 'T_rl', 'T_rr')), None)
    if tv_lap:
        s_tv = tv_lap.get('s')
        ax.plot(s_tv, tv_lap.get('T_fl'), color=theme.tv_fl, lw=1.0, label='FL')
        ax.plot(s_tv, tv_lap.get('T_fr'), color=theme.tv_fr, lw=1.0, label='FR')
        ax.plot(s_tv, tv_lap.get('T_rl'), color=theme.tv_rl, lw=1.4, label='RL')
        ax.plot(s_tv, tv_lap.get('T_rr'), color=theme.tv_rr, lw=1.4, label='RR')
        ax.axhline(0, color=theme.grid, lw=0.4)
        ax.legend(loc='lower right', ncol=4)
        ax.set_ylabel(f'TV Torque\n[N·m]\n— {tv_lap.name}')
    else:
        ax.text(0.5, 0.5, 'no TV channels (T_fl/T_fr/T_rl/T_rr)',
                transform=ax.transAxes, ha='center', va='center',
                color=theme.muted, fontsize=9, style='italic')
        ax.set_ylabel('TV Torque\n[N·m]')

    # ── 6. Grip utilisation ─────────────────────────────────────────────────
    ax = axes[5]
    mu_assumed = 1.40
    for lap in laps:
        util = lap.get('combinedG') / mu_assumed * 100.0
        ax.plot(lap.get('s'), util, color=lap.color, linestyle=lap.style,
                lw=1.6 if lap.is_reference else 1.0,
                alpha=1.0 if lap.is_reference else 0.75,
                label=lap.name)
    ax.axhline(100, color=theme.brake, lw=0.6, ls='--', alpha=0.6, label='μ-limit')
    ax.set_ylabel('Grip Util.\n[%]')
    ax.set_ylim(0, 130)

    fig.suptitle(title or 'Project-GP — MoTeC Telemetry Dashboard',
                 fontsize=12, fontweight='bold', y=0.985)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §5  Track-map heatmap
# ════════════════════════════════════════════════════════════════════════════

def _ribbon_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return (N-1, 2, 2) array of consecutive segment endpoints, suitable
    for matplotlib LineCollection. Used as the spine; scatter dots overlay.
    """
    pts = np.column_stack([x, y])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def render_track_map(
    lap: LapData,
    *,
    metric: str,
    theme: MoTeCTheme,
    out_path: Path,
    cmap: str = 'coolwarm',
    title: str | None = None,
    vlim: tuple[float, float] | None = None,
    diverging: bool = False,
    overlay_track: LapData | None = None,
) -> None:
    """Top-down trajectory ribbon coloured by `metric`.

    Two layers: thin ribbon LineCollection (spine continuity) + scatter dots
    (per-sample resolution). User asked for scatter; ribbon improves
    legibility on dense sampling without occluding the dots.

    `diverging`=True snaps the colormap symmetric around zero (TwoSlopeNorm)
    — appropriate for signed metrics like yaw error. Sequential metrics
    (temperature, latency) keep a standard Normalize.
    """
    theme.apply()
    if metric not in lap.df.columns:
        raise KeyError(f"Metric '{metric}' not in lap '{lap.name}'. "
                       f"Available: {sorted(lap.df.columns)}")

    x, y = lap.get('x'), lap.get('y')
    z    = lap.get(metric)

    if vlim is None:
        if diverging:
            m = float(np.nanmax(np.abs(z))) if z.size else 1.0
            vmin, vmax = -m, m
        else:
            vmin, vmax = float(np.nanmin(z)), float(np.nanmax(z))
    else:
        vmin, vmax = vlim

    norm = (TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            if diverging and vmin < 0 < vmax
            else Normalize(vmin=vmin, vmax=vmax))

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor(theme.panel)
    ax.set_aspect('equal', adjustable='datalim')

    # Optional reference track outline (e.g. centerline + bounds) underneath
    if overlay_track is not None:
        ax.plot(overlay_track.get('x'), overlay_track.get('y'),
                color=theme.muted, lw=0.6, alpha=0.5, ls='--', zorder=1)

    # Thin ribbon — continuity spine
    segs = _ribbon_segments(x, y)
    lc = LineCollection(segs, cmap=cmap, norm=norm, alpha=0.45, lw=2.2, zorder=2)
    lc.set_array(0.5 * (z[:-1] + z[1:]))
    ax.add_collection(lc)

    # Per-sample dots
    sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm,
                    s=8, edgecolors='none', zorder=3)

    # Start/finish marker
    ax.plot(x[0], y[0], 'o', mfc=theme.fg, mec=theme.bg,
            mew=1.5, ms=10, zorder=5)
    ax.annotate(' Start', (x[0], y[0]), color=theme.fg,
                fontsize=9, fontweight='bold', zorder=6)

    cb = fig.colorbar(sc, ax=ax, fraction=0.034, pad=0.02)
    cb.set_label(metric, color=theme.fg)
    cb.ax.tick_params(colors=theme.fg)
    cb.outline.set_edgecolor(theme.grid)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(title or f'Track Map — {lap.name} — coloured by {metric}')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §6  G-G friction circle
# ════════════════════════════════════════════════════════════════════════════

def render_gg_diagram(
    laps: Sequence[LapData],
    *,
    theme: MoTeCTheme,
    out_path: Path,
    mu: float = 1.40,
    show_kde: bool = True,
    show_ellipses: bool = True,
) -> None:
    """A_y vs A_x scatter with μ-circle envelope, KDE density, and 1σ/2σ
    confidence ellipses per lap. The KDE reveals not just the operating
    envelope but the *occupancy density* — a Golden Lap should distribute
    energy uniformly along the perimeter; a sub-optimal lap clusters interior.
    """
    theme.apply()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_facecolor(theme.panel)

    # μ-circle
    ax.add_patch(Circle((0, 0), mu, fill=False,
                        edgecolor=theme.brake, lw=1.8, ls='--', label=f'μ = {mu:g}'))
    # 0.5μ inner reference
    ax.add_patch(Circle((0, 0), mu * 0.5, fill=False,
                        edgecolor=theme.muted, lw=0.8, ls=':', alpha=0.6))

    # Per-lap layer
    for lap in laps:
        ax_g, ay_g = lap.get('latG'), lap.get('lonG')
        ax.scatter(ax_g, ay_g, s=4, color=lap.color,
                   alpha=0.55 if not lap.is_reference else 0.75,
                   edgecolors='none', label=lap.name, zorder=3)

        if show_ellipses and lap.n > 30:
            mu_x, mu_y = float(np.mean(ax_g)), float(np.mean(ay_g))
            cov = np.cov(np.stack([ax_g, ay_g]))
            eig_val, eig_vec = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eig_vec[1, 1], eig_vec[0, 1]))
            for k in (1.0, 2.0):
                w, h = 2 * k * np.sqrt(np.maximum(eig_val, 0.0))
                ax.add_patch(Ellipse((mu_x, mu_y), w, h, angle=angle,
                                     edgecolor=lap.color, facecolor='none',
                                     lw=0.8 if k == 1 else 0.5,
                                     alpha=0.7 if k == 1 else 0.4, zorder=2))

    # KDE on the reference lap
    if show_kde:
        ref = next((l for l in laps if l.is_reference), laps[0])
        ax_g, ay_g = ref.get('latG'), ref.get('lonG')
        try:
            kde = gaussian_kde(np.stack([ax_g, ay_g]), bw_method=0.18)
            xx, yy = np.mgrid[-mu*1.2:mu*1.2:120j, -mu*1.2:mu*1.2:120j]
            zz = kde(np.stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=8,
                        cmap=LinearSegmentedColormap.from_list(
                            'kde', [(0, 0, 0, 0), ref.color]),
                        alpha=0.35, zorder=1)
        except Exception as e:
            warnings.warn(f"KDE failed: {e}")

    ax.axhline(0, color=theme.grid, lw=0.4, zorder=0)
    ax.axvline(0, color=theme.grid, lw=0.4, zorder=0)

    lim = mu * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Lateral G  [A_y]')
    ax.set_ylabel('Longitudinal G  [A_x]   (+brake / −drive)')
    ax.set_title('G-G Friction Circle — Operating-Point Envelope')
    ax.legend(loc='upper left', fontsize=8)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §7  Real-time TV-controller latency benchmark
# ════════════════════════════════════════════════════════════════════════════

class TVMethod(Protocol):
    def __call__(self, state: np.ndarray, target_yaw_rate: float) -> np.ndarray: ...


@dataclass
class LatencySample:
    method: str
    s:        np.ndarray  # arc length at sample
    latency_us: np.ndarray  # microseconds


@contextmanager
def _ns_timer():
    """Nanosecond-resolution context manager. Returns a one-element list
    whose value is the elapsed ns after the context exits."""
    out = [0]
    t0 = time.perf_counter_ns()
    try:
        yield out
    finally:
        out[0] = time.perf_counter_ns() - t0


def benchmark_tv_methods(
    methods: Mapping[str, TVMethod],
    reference_lap: LapData,
    *,
    n_warmup: int = 10,
    state_cols: Sequence[str] = ('vx', 'vy', 'wz', 'beta', 'kappa'),
    yaw_ref_col: str = 'wz_ref',
) -> dict[str, LatencySample]:
    """Replays `reference_lap` step-by-step. For each step, calls each TV
    method on the current state, records wall-clock latency. Outputs are
    decoupled from method choice — this is a SHADOW benchmark of latency
    under realistic state distributions, not a closed-loop comparison.

    Discards the first `n_warmup` samples per method (JIT warm-up, allocator
    cache fill). Returns {method_name: LatencySample}.
    """
    s_arr = reference_lap.get('s')
    n     = len(s_arr)

    states = np.column_stack([
        reference_lap.get(c, default=0.0) for c in state_cols
    ])
    yaw_ref = (reference_lap.get(yaw_ref_col, default=0.0)
               if reference_lap.has(yaw_ref_col)
               else reference_lap.get('wz', default=0.0))

    out: dict[str, LatencySample] = {}
    for name, fn in methods.items():
        latencies = np.empty(n, dtype=np.float64)
        for i in range(n):
            with _ns_timer() as t:
                _ = fn(states[i], float(yaw_ref[i]))
            latencies[i] = t[0] / 1_000.0   # → μs

        # Trim warmup; pad with NaN to preserve s-alignment
        warm_mask = np.ones(n, dtype=bool); warm_mask[:n_warmup] = False
        latencies_clean = np.where(warm_mask, latencies, np.nan)
        out[name] = LatencySample(method=name, s=s_arr, latency_us=latencies_clean)
    return out


def render_latency_histogram(
    samples: Mapping[str, LatencySample],
    *,
    theme: MoTeCTheme,
    out_path: Path,
    log_scale: bool = True,
) -> None:
    """Per-method histogram with p50/p95/p99 vlines. Log-x by default
    because TV-method runtimes typically span 2-3 decades (Simple ~5 µs,
    mpQP ~80 µs, WMPC ~2 ms)."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(theme.panel)

    palette = theme.cmp
    for i, (name, s) in enumerate(samples.items()):
        c = palette[i % len(palette)]
        lat = s.latency_us[~np.isnan(s.latency_us)]
        if lat.size == 0:
            continue
        ax.hist(lat, bins=80, color=c, alpha=0.45,
                histtype='stepfilled', label=f'{name}', density=True,
                edgecolor=c, lw=1.0)
        for q, ls, lab_q in zip((50, 95, 99), ('-', '--', ':'),
                                 ('p50', 'p95', 'p99')):
            v = float(np.percentile(lat, q))
            ax.axvline(v, color=c, lw=0.8, ls=ls, alpha=0.7)
            ax.text(v, ax.get_ylim()[1] * (0.92 - 0.07 * i), f' {lab_q}={v:.1f}µs',
                    color=c, fontsize=7, rotation=0)

    if log_scale: ax.set_xscale('log')
    ax.set_xlabel('Latency [µs]')
    ax.set_ylabel('Density')
    ax.set_title('TV Controller Latency — Real-Time Benchmark')
    ax.legend(loc='upper right')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §8  Bonus — sector-delta time analysis
# ════════════════════════════════════════════════════════════════════════════

def _detect_sectors(s: np.ndarray, kappa: np.ndarray,
                    kappa_thresh: float = 0.04, min_arc: float = 8.0) -> list[tuple[int, int, str]]:
    """Auto-segment by signed curvature into {STR, L, R} sectors. Returns
    list of (i_start, i_end, label). Min-arc filter avoids fragmenting on
    sensor noise around κ≈0."""
    cls = np.where(np.abs(kappa) < kappa_thresh, 0,
                   np.where(kappa > 0, 1, -1)).astype(int)
    sectors: list[tuple[int, int, str]] = []
    i = 0
    while i < len(cls):
        j = i + 1
        while j < len(cls) and cls[j] == cls[i]:
            j += 1
        arc = float(s[min(j, len(s) - 1)] - s[i])
        if arc >= min_arc:
            label = {0: 'STR', 1: 'L', -1: 'R'}[int(cls[i])]
            sectors.append((i, j, label))
        i = j
    return sectors


def render_sector_deltas(
    laps: Sequence[LapData],
    *,
    theme: MoTeCTheme,
    out_path: Path,
) -> None:
    """Per-sector time-loss bar chart vs reference. Sectors auto-detected
    from κ(s) of the reference lap. For each sector i, Δt_lap = Σ(ds/v_lap)
    — Σ(ds/v_ref). Positive = time loss vs reference."""
    theme.apply()
    ref = next((l for l in laps if l.is_reference), laps[0])
    s_ref = ref.get('s')
    sectors = _detect_sectors(s_ref, ref.get('kappa', default=0.0))
    if not sectors:
        warnings.warn('Sector detection found no sectors; skipping.')
        return

    others = [l for l in laps if l is not ref]
    if not others:
        warnings.warn('No comparison laps for sector delta; skipping.')
        return

    deltas: dict[str, list[float]] = {l.name: [] for l in others}
    labels: list[str] = []
    for k, (i0, i1, lab) in enumerate(sectors):
        labels.append(f'{lab}{k+1}\n[{s_ref[i0]:.0f}–{s_ref[min(i1, len(s_ref)-1)]:.0f}m]')
        ds = np.diff(s_ref[i0:i1+1])
        v_ref_seg = 0.5 * (ref.get('v')[i0:i1] + ref.get('v')[i0+1:i1+1])
        t_ref = float(np.sum(ds / np.maximum(v_ref_seg, 0.5)))
        for lap in others:
            v_l = np.interp(s_ref[i0:i1+1], lap.get('s'), lap.get('v'))
            v_l_seg = 0.5 * (v_l[:-1] + v_l[1:])
            t_l = float(np.sum(ds / np.maximum(v_l_seg, 0.5)))
            deltas[lap.name].append(t_l - t_ref)

    fig, ax = plt.subplots(figsize=(13, 6))
    n_sec = len(labels)
    n_lap = len(others)
    width = 0.8 / max(n_lap, 1)
    xpos = np.arange(n_sec)
    for i, lap in enumerate(others):
        offset = (i - (n_lap - 1) / 2) * width
        bars = ax.bar(xpos + offset, deltas[lap.name], width,
                      color=lap.color, alpha=0.85, label=lap.name,
                      edgecolor=theme.bg, lw=0.5)
        for b, v in zip(bars, deltas[lap.name]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005 * np.sign(v or 1),
                    f'{v*1000:+.0f}ms',
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=7, color=lap.color)

    ax.axhline(0, color=theme.fg, lw=0.6)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f'Δt vs {ref.name}  [s]   (positive = slower)')
    ax.set_title(f'Per-Sector Time Delta — reference: {ref.name}')
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §9  Bonus — phase portrait (β vs r)
# ════════════════════════════════════════════════════════════════════════════

def render_phase_portrait(
    laps: Sequence[LapData],
    *,
    theme: MoTeCTheme,
    out_path: Path,
    beta_lim_deg: float = 12.0,
    r_lim: float = 1.6,
) -> None:
    """β–r stability portrait. Stable handling envelope is the inner ellipse;
    a controller that pushes outside has lost yaw stability margin."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor(theme.panel)

    ax.add_patch(Ellipse((0, 0), 2 * 0.7 * beta_lim_deg, 2 * 0.7 * r_lim,
                         fill=False, edgecolor=theme.muted, lw=0.8, ls=':',
                         label='nominal envelope'))
    ax.add_patch(Ellipse((0, 0), 2 * beta_lim_deg, 2 * r_lim,
                         fill=False, edgecolor=theme.brake, lw=1.0, ls='--',
                         label='instability bound'))

    for lap in laps:
        if not lap.has('beta', 'wz'):
            continue
        beta_deg = np.degrees(lap.get('beta'))
        r        = lap.get('wz')
        ax.plot(beta_deg, r, color=lap.color,
                lw=0.7, alpha=0.7 if not lap.is_reference else 0.95,
                label=lap.name)

    ax.axhline(0, color=theme.grid, lw=0.4)
    ax.axvline(0, color=theme.grid, lw=0.4)
    ax.set_xlabel('Sideslip β [deg]')
    ax.set_ylabel('Yaw rate r [rad/s]')
    ax.set_title('β–r Phase Portrait — Yaw-Stability Envelope')
    ax.set_xlim(-beta_lim_deg * 1.3, beta_lim_deg * 1.3)
    ax.set_ylim(-r_lim * 1.3, r_lim * 1.3)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §10 Bonus — steering-channel power spectrum
# ════════════════════════════════════════════════════════════════════════════

def render_steering_fft(
    laps: Sequence[LapData],
    *,
    theme: MoTeCTheme,
    out_path: Path,
    fs_hint: float = 200.0,
) -> None:
    """Welch PSD of the steering channel. Aggressive controllers (Simple TV
    correcting on top of human steer) show high-frequency lobes; smooth ones
    (Diff-WMPC) concentrate energy <2 Hz."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(theme.panel)

    for lap in laps:
        if not lap.has('steer', 't'):
            continue
        t = lap.get('t')
        if t.size < 64: continue
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt if dt > 1e-6 else fs_hint
        f, P = welch(lap.get('steer'), fs=fs, nperseg=min(256, lap.n // 2))
        ax.semilogy(f, np.maximum(P, 1e-12),
                    color=lap.color, lw=1.2, alpha=0.85, label=f'{lap.name}  fs={fs:.1f}Hz')

    ax.set_xlim(0, 25)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Steering PSD [rad²/Hz]')
    ax.set_title('Steering Channel — Welch Power Spectral Density')
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §11 Bonus — power flow over arc length
# ════════════════════════════════════════════════════════════════════════════

def render_power_flow(
    lap: LapData,
    *,
    theme: MoTeCTheme,
    out_path: Path,
    eta_drive: float = 0.92,
    eta_regen: float = 0.65,
    m: float = 235.0,
) -> None:
    """Battery / drive / regen / loss power decomposition. Useful for
    endurance: shows where energy is wasted as heat (brake) vs recovered.

    P_kin  = m·v·a_x
    P_drive = max(P_kin, 0) / η_drive             (electrical → wheels)
    P_regen = -min(P_kin, 0) · η_regen            (wheels → battery)
    P_bat   = P_drive − P_regen                   (signed: + = draw, − = recharge)
    """
    theme.apply()
    if not lap.has('v', 's', 'lonG'):
        warnings.warn('Power flow needs v, s, lonG; skipping.')
        return

    s   = lap.get('s')
    v   = lap.get('v')
    a_x = lap.get('lonG') * 9.81
    P_kin    = m * v * a_x
    P_drive  =  np.maximum(P_kin, 0.0) / eta_drive
    P_regen  = -np.minimum(P_kin, 0.0) * eta_regen
    P_bat    = P_drive - P_regen
    P_loss   = np.where(P_kin >= 0, P_drive - P_kin, -P_kin - P_regen)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(s, 0, P_drive / 1e3, color=theme.throttle, alpha=0.55, label='Drive (kW)')
    ax.fill_between(s, 0, -P_regen / 1e3, color=theme.regen,    alpha=0.55, label='Regen (kW, ⁻)')
    ax.plot(s, P_bat  / 1e3, color=theme.fg,    lw=1.2, label='Battery net')
    ax.plot(s, P_loss / 1e3, color=theme.brake, lw=0.8, ls='--', alpha=0.7, label='Heat loss')
    ax.axhline(0, color=theme.grid, lw=0.4)
    ax.set_xlabel('s [m]')
    ax.set_ylabel('Power [kW]')
    ax.set_title(f'Power Flow Decomposition — {lap.name}  '
                 f'(η_drive={eta_drive:.2f}, η_regen={eta_regen:.2f}, m={m:.0f}kg)')
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §12 Bonus — yaw-rate tracking error
# ════════════════════════════════════════════════════════════════════════════

def render_yaw_tracking(
    laps: Sequence[LapData],
    *,
    theme: MoTeCTheme,
    out_path: Path,
) -> None:
    """Reference vs actual yaw rate with shaded tracking-error band per lap.
    Quantifies the model-following accuracy of each TV strategy."""
    theme.apply()
    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.12)
    ax_top = fig.add_subplot(gs[0]); ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ref = next((l for l in laps if l.is_reference), laps[0])
    if ref.has('wz_ref'):
        ax_top.plot(ref.get('s'), ref.get('wz_ref'),
                    color=theme.ref, lw=1.4, ls='--', label='r_ref', alpha=0.8)

    for lap in laps:
        if not lap.has('wz'):
            continue
        s, r = lap.get('s'), lap.get('wz')
        ax_top.plot(s, r, color=lap.color, lw=1.0,
                    alpha=0.9 if lap.is_reference else 0.75, label=lap.name)
        if lap.has('wz_ref'):
            err = r - lap.get('wz_ref')
            ax_bot.fill_between(s, err, 0, color=lap.color, alpha=0.4)

    ax_top.set_ylabel('Yaw rate [rad/s]')
    ax_top.legend(ncol=len(laps), loc='upper right')
    ax_bot.set_ylabel('r − r_ref [rad/s]')
    ax_bot.set_xlabel('s [m]')
    ax_bot.axhline(0, color=theme.grid, lw=0.5)
    ax_top.set_title('Yaw-Rate Tracking — model-following accuracy')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# §13 Orchestrator
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationPipeline:
    theme: MoTeCTheme = field(default_factory=MoTeCTheme)
    laps:  list[LapData] = field(default_factory=list)
    tv_methods: dict[str, TVMethod] = field(default_factory=dict)

    def add_lap(self, name: str, df: pd.DataFrame, *,
                color: str = '#00D4FF', is_reference: bool = False,
                style: str = '-') -> LapData:
        lap = LapData(name=name, df=df, color=color,
                      is_reference=is_reference, style=style)
        self.laps.append(lap)
        return lap

    def add_tv_method(self, name: str, fn: TVMethod) -> None:
        self.tv_methods[name] = fn

    def render_all(self,
                   out_dir: str | Path = './figs',
                   report_path: str | Path | None = './reports/validation_report.json',
                   bench_tv: bool = False) -> dict:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        if not self.laps:
            raise RuntimeError('No laps registered.')

        ref = next((l for l in self.laps if l.is_reference), self.laps[0])

        # 1. Dashboard — one per non-reference lap, plus a combined figure
        render_motec_dashboard(self.laps, theme=self.theme,
                               out_path=out / 'motec_dashboard_combined.png',
                               title='MoTeC Dashboard — All Laps')
        for lap in self.laps:
            render_motec_dashboard([ref, lap] if lap is not ref else [lap],
                                   theme=self.theme,
                                   out_path=out / f'motec_dashboard_{lap.name}.png',
                                   title=f'MoTeC Dashboard — {lap.name} vs {ref.name}'
                                         if lap is not ref
                                         else f'MoTeC Dashboard — {lap.name}')

        # 2. Track maps
        # Yaw error: lap.wz − ref.wz interpolated to lap.s — diverging colormap
        for lap in self.laps:
            if lap.has('wz') and ref.has('wz') and lap is not ref:
                wz_ref_at_lap = np.interp(lap.get('s'), ref.get('s'), ref.get('wz'))
                lap.df['yaw_error'] = lap.get('wz') - wz_ref_at_lap
                render_track_map(lap, metric='yaw_error', theme=self.theme,
                                 out_path=out / f'track_map_yaw_{lap.name}.png',
                                 cmap='coolwarm', diverging=True,
                                 title=f'Yaw-Error Map — {lap.name} vs {ref.name}',
                                 overlay_track=ref)
            if lap.has('T_tire_rl', 'T_tire_rr'):
                lap.df['tire_temp_outer_rear'] = np.maximum(
                    lap.get('T_tire_rl'), lap.get('T_tire_rr'))
                render_track_map(lap, metric='tire_temp_outer_rear', theme=self.theme,
                                 out_path=out / f'track_map_temp_{lap.name}.png',
                                 cmap='inferno',
                                 title=f'Outer-Rear Tire Temp — {lap.name}')

        # 3. G-G
        render_gg_diagram(self.laps, theme=self.theme,
                          out_path=out / 'gg_diagram.png')

        # 4-7. Bonus
        render_sector_deltas(self.laps, theme=self.theme,
                             out_path=out / 'sector_delta.png')
        render_phase_portrait(self.laps, theme=self.theme,
                              out_path=out / 'phase_portrait.png')
        render_steering_fft(self.laps, theme=self.theme,
                            out_path=out / 'steering_fft.png')
        render_power_flow(ref, theme=self.theme,
                          out_path=out / f'power_flow_{ref.name}.png')
        render_yaw_tracking(self.laps, theme=self.theme,
                            out_path=out / 'yaw_tracking.png')

        # 8. TV latency benchmark (opt-in)
        bench_summary: dict = {}
        if bench_tv and self.tv_methods:
            samples = benchmark_tv_methods(self.tv_methods, ref)
            render_latency_histogram(samples, theme=self.theme,
                                     out_path=out / 'latency_histogram.png')
            for nm, sm in samples.items():
                lap = next((l for l in self.laps
                            if l.name.lower() == nm.lower()), ref)
                # Spatial latency map: paint the reference trajectory by
                # per-step latency of *this* method, on the reference's xy.
                lap_for_map = LapData(
                    name=f'{nm}_lat',
                    df=pd.DataFrame({
                        's': sm.s, 'x': ref.get('x'), 'y': ref.get('y'),
                        'latency_us': sm.latency_us,
                    }),
                    color=lap.color)
                render_track_map(lap_for_map, metric='latency_us', theme=self.theme,
                                 out_path=out / f'track_map_latency_{nm}.png',
                                 cmap='magma',
                                 title=f'TV Latency Map — {nm}')
                bench_summary[nm] = {
                    'p50_us': float(np.nanpercentile(sm.latency_us, 50)),
                    'p95_us': float(np.nanpercentile(sm.latency_us, 95)),
                    'p99_us': float(np.nanpercentile(sm.latency_us, 99)),
                    'max_us': float(np.nanmax(sm.latency_us)),
                }

        # Validation report
        report = {
            'reference_lap': ref.name,
            'laps': [
                {
                    'name': l.name,
                    'lap_time': l.lap_time,
                    'samples': l.n,
                    's_max': float(l.get('s')[-1]) if l.n else 0.0,
                    'v_max_kph': float(np.max(l.get('v')) * 3.6),
                    'g_combined_max': float(np.max(l.get('combinedG'))),
                    'g_combined_mean': float(np.mean(l.get('combinedG'))),
                    'pct_above_p9_mu':
                        100.0 * float(np.mean(l.get('combinedG') > 0.9 * 1.40)),
                }
                for l in self.laps
            ],
            'latency_benchmark': bench_summary,
        }
        if report_path:
            rp = Path(report_path); rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2))
        return report


# ════════════════════════════════════════════════════════════════════════════
# §14 CLI
# ════════════════════════════════════════════════════════════════════════════

def _cli() -> int:
    p = argparse.ArgumentParser(description='Project-GP Telemetry Validation Pipeline')
    p.add_argument('--golden', type=str, help='WMPC Golden Lap CSV (reference)')
    p.add_argument('--simple', type=str, help='Simple TV CSV')
    p.add_argument('--mpqp',   type=str, help='mpQP TV CSV')
    p.add_argument('--wmpc',   type=str, help='WMPC TV closed-loop CSV')
    p.add_argument('--human',  type=str, help='Human driver CSV')
    p.add_argument('--out',    type=str, default='./figs')
    p.add_argument('--report', type=str, default='./reports/validation_report.json')
    p.add_argument('--bench-tv', action='store_true',
                   help='Run real-time TV latency benchmark (requires importable methods).')
    args = p.parse_args()

    pipe = ValidationPipeline()
    palette = iter(MoTeCTheme().cmp)

    inputs = [
        ('Golden', args.golden, True),
        ('Simple', args.simple, False),
        ('mpQP',   args.mpqp,   False),
        ('WMPC',   args.wmpc,   False),
        ('Human',  args.human,  False),
    ]
    added = 0
    for name, path, is_ref in inputs:
        if not path: continue
        df = load_csv(path)
        pipe.add_lap(name, df, color=next(palette), is_reference=is_ref)
        added += 1
    if added == 0:
        p.error('At least one lap CSV must be provided.')

    # Stub TV methods for bench demo if explicitly enabled but no real ones
    # are wired. User should overwrite via programmatic `add_tv_method`.
    if args.bench_tv and not pipe.tv_methods:
        warnings.warn('--bench-tv enabled but no TV methods registered. '
                      'Wire real methods via pipe.add_tv_method() in a script.')

    rep = pipe.render_all(out_dir=args.out, report_path=args.report,
                          bench_tv=args.bench_tv)
    print(json.dumps({k: v for k, v in rep.items() if k != 'laps'}, indent=2))
    return 0


if __name__ == '__main__':   # pragma: no cover
    raise SystemExit(_cli())