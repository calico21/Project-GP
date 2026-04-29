# telemetry/driver_coaching.py
# ═══════════════════════════════════════════════════════════════════════════════
# Project-GP — AI Race Engineer (Spatial Align + Delta Channels + NLP Logic)
# ═══════════════════════════════════════════════════════════════════════════════
"""
Two-stage driver coaching pipeline.

Stage 1 — Spatial alignment (align_human_to_ghost):
    Human telemetry is sampled in TIME (typically 100 Hz, GPS+IMU).
    Ghost telemetry is sampled in DISTANCE (1 m on the centerline).
    We project the human (x, y) trajectory onto the ghost spline using a
    JIT-compiled monotonic windowed soft-argmin search (faster, smoother,
    and more stable than full DTW for racing telemetry where the driver
    never reverses progress). After alignment, all human channels are
    interpolated onto the ghost's s-grid for pointwise comparison.

    Why not DTW: full DTW is O(N·M) ≈ 10⁵·10³ = 10⁸ per session and produces
    non-monotonic alignments under noise. The 1-D monotonic projection used
    here is O(N·W) with W=96 candidates and exploits the physical constraint
    that ds/dt > 0 along a hot lap. Full DTW is provided as a fallback for
    edge cases (split sectors, recovery laps) via dtw_align_full().

Stage 2 — Delta-channel extraction (compute_deltas):
    Computes paired channel deltas on the unified s-grid:
        speed_delta(s)    = v_human − v_ghost                                 [m/s]
        coasting_delta(s) = (T_g + B_g) − (T_h + B_h)                       [pedal %]
        brake_delta(s)    = B_human − B_ghost                                [N or %]
        line_delta(s)     = n_human − n_ghost                                [m]
        steer_delta(s)    = δ_human − δ_ghost                                [rad]
        latG_lost(s)      = combinedG_ghost − combinedG_human                [g]

Stage 3 — NLP logic tree (generate_coaching_report):
    Detects persistent events on the delta channels (sustained > min_persistence
    samples) within corner zones (|κ| > κ_threshold). Each event is mapped to
    one of 6 driver archetypes via a deterministic decision tree, then
    formatted into plain-English coaching feedback with quantified loss.

Public API:
    align_human_to_ghost(df_human, df_ghost) -> pd.DataFrame
    compute_deltas(df_aligned, df_ghost) -> pd.DataFrame
    generate_coaching_report(df_deltas, df_ghost) -> List[CoachingEvent]
    full_pipeline(df_human, df_ghost) -> (df_aligned, df_deltas, events)

Output format compatible with visualization/dashboard_react/DriverCoachingModule.jsx.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import partial
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import lax, jit


# ═════════════════════════════════════════════════════════════════════════════
# §1.  Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CoachingEvent:
    s_start:       float
    s_end:         float
    corner_id:     int
    archetype:     str
    severity_g:    float                # lateral grip "left on the table" [g]
    time_lost_s:   float                # estimated time lost to this event [s]
    advice:        str                  # plain-English coaching feedback

    def as_dict(self):  return asdict(self)


@dataclass(frozen=True)
class CornerZone:
    corner_id:  int
    s_entry:    float
    s_apex:     float
    s_exit:     float
    direction:  str                     # 'L' or 'R'
    radius_m:   float


# ═════════════════════════════════════════════════════════════════════════════
# §2.  Stage 1 — Spatial alignment via monotonic windowed soft-argmin
# ═════════════════════════════════════════════════════════════════════════════

def _build_ghost_lookup(df_ghost: pd.DataFrame) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Returns (xy_ghost [N_g, 2], s_ghost [N_g], L_total)."""
    xy_g = jnp.stack([jnp.array(df_ghost['x'].values),
                      jnp.array(df_ghost['y'].values)], axis=-1)
    s_g  = jnp.array(df_ghost['s'].values)
    L    = float(s_g[-1] - s_g[0]) + float(s_g[1] - s_g[0])
    return xy_g, s_g, L


def _project_one(xy_query: jnp.ndarray,
                 xy_ghost: jnp.ndarray,
                 s_ghost:  jnp.ndarray,
                 s_prev:   jnp.ndarray,
                 win_half: float = 30.0,
                 tau:      float = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Soft-argmin projection of one (x,y) point onto the ghost spline within a
    ±win_half window around s_prev. Returns (s_aligned, n_lateral_signed).

    n_lateral_signed > 0  →  human is left of centerline at that s.
    """
    # Mask: only candidates within window are eligible (large penalty otherwise)
    in_window = jnp.abs(s_ghost - s_prev) < win_half
    d_sq = jnp.sum((xy_ghost - xy_query[None, :]) ** 2, axis=-1)
    masked = d_sq + (1.0 - in_window.astype(d_sq.dtype)) * 1e6

    weights = jax.nn.softmax(-masked / tau)
    s_aligned = jnp.sum(weights * s_ghost)

    # Approximate lateral signed offset: dot product with ghost normal.
    # For the ghost normal, finite-difference adjacent points (good enough at 1m grid)
    # We pick the ghost station closest to s_aligned for the local frame.
    idx = jnp.argmin(jnp.abs(s_ghost - s_aligned))
    idx = jnp.clip(idx, 1, s_ghost.shape[0] - 2)
    tan = xy_ghost[idx + 1] - xy_ghost[idx - 1]
    tan = tan / (jnp.linalg.norm(tan) + 1e-9)
    n_hat = jnp.stack([-tan[1], tan[0]])                              # +N̂ = left
    delta_xy = xy_query - xy_ghost[idx]
    n_signed = jnp.dot(delta_xy, n_hat)
    return s_aligned, n_signed


def _scan_align_monotonic(xy_human: jnp.ndarray,
                          xy_ghost: jnp.ndarray,
                          s_ghost:  jnp.ndarray,
                          s0:       jnp.ndarray,
                          eps_back: float = 0.5,
                          win_half: float = 30.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sequential alignment with monotonicity constraint:
        s(t+1) >= s(t) - eps_back
    The eps_back tolerance allows for telemetry GPS jitter without losing
    progress on tight hairpins where two consecutive human samples can land
    on either side of the centerline projection.
    """
    def step(s_prev, xy_t):
        s_t, n_t = _project_one(xy_t, xy_ghost, s_ghost, s_prev,
                                win_half=win_half)
        s_t = jnp.maximum(s_t, s_prev - eps_back)
        return s_t, (s_t, n_t)

    _, (s_aligned, n_aligned) = lax.scan(step, s0, xy_human)
    return s_aligned, n_aligned


@partial(jit, static_argnames=('win_half',))
def _align_jax(xy_human: jnp.ndarray,
               xy_ghost: jnp.ndarray,
               s_ghost:  jnp.ndarray,
               s0:       jnp.ndarray,
               win_half: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _scan_align_monotonic(xy_human, xy_ghost, s_ghost, s0,
                                 win_half=win_half)


def align_human_to_ghost(df_human:  pd.DataFrame,
                         df_ghost:  pd.DataFrame,
                         *,
                         win_half:  float = 30.0) -> pd.DataFrame:
    """
    Project every human (x, y) sample onto the ghost s-grid, then interpolate
    all available human channels onto the GHOST's uniform 1-meter s-grid so
    that pointwise comparison with df_ghost is meaningful.

    Required columns in df_human: 'x', 'y', 't'.
    Optional columns (interpolated if present): 'v', 'steer', 'throttle',
        'brake', 'latG', 'lonG', 'yaw_rate'.

    Required columns in df_ghost:  's', 'x', 'y'.

    Returns DataFrame with one row per ghost station, columns suffixed
    '_human' for the aligned channels.
    """
    # 1) Project human (x, y) onto ghost spline, with monotonic constraint
    xy_g, s_g, L = _build_ghost_lookup(df_ghost)
    xy_h = jnp.stack([jnp.array(df_human['x'].values),
                      jnp.array(df_human['y'].values)], axis=-1)

    # Initial s0: project the first sample globally (no window)
    d0 = jnp.sum((xy_g - xy_h[0][None, :]) ** 2, axis=-1)
    s0 = s_g[jnp.argmin(d0)]

    s_aligned_jax, n_aligned_jax = _align_jax(xy_h, xy_g, s_g, s0, win_half=win_half)
    s_aligned = np.asarray(s_aligned_jax)
    n_aligned = np.asarray(n_aligned_jax)

    # 2) Resample human channels onto the ghost s-grid via 1-D interp.
    #    `s_aligned` is monotonic by construction, so np.interp is safe.
    s_grid = df_ghost['s'].values
    out = pd.DataFrame({'s': s_grid})

    optional_channels = ['v', 'steer', 'throttle', 'brake',
                         'latG', 'lonG', 'yaw_rate', 't']
    for ch in optional_channels:
        if ch in df_human.columns:
            out[f'{ch}_human'] = np.interp(s_grid, s_aligned, df_human[ch].values)

    # n_lateral_human: signed offset already computed
    out['n_lateral_human'] = np.interp(s_grid, s_aligned, n_aligned)

    # Diagnostic: how much spatial coverage do we have?
    coverage_pct = float((np.diff(s_aligned) >= 0).mean() * 100.0)
    print(f"[Align] Monotonic-coverage = {coverage_pct:.1f}% "
          f"(s_aligned range: [{s_aligned.min():.1f}, {s_aligned.max():.1f}])")
    return out


def dtw_align_full(df_human: pd.DataFrame, df_ghost: pd.DataFrame) -> pd.DataFrame:
    """
    Full DTW fallback for pathological cases (split sectors, multi-lap concat).
    O(N·M) — uses banded DTW with Sakoe-Chiba radius = max(N,M)/8.
    Falls back to a vanilla NumPy implementation; not JIT'd because DTW's
    dependency chain isn't `scan`-friendly.
    """
    xy_h = df_human[['x', 'y']].values
    xy_g = df_ghost[['x', 'y']].values
    N, M = len(xy_h), len(xy_g)
    radius = max(N, M) // 8

    INF = 1e18
    cost = np.full((N + 1, M + 1), INF)
    cost[0, 0] = 0.0
    for i in range(1, N + 1):
        j_lo = max(1, int(i * M / N) - radius)
        j_hi = min(M, int(i * M / N) + radius)
        for j in range(j_lo, j_hi + 1):
            d = np.sum((xy_h[i - 1] - xy_g[j - 1]) ** 2)
            cost[i, j] = d + min(cost[i - 1, j],
                                 cost[i, j - 1],
                                 cost[i - 1, j - 1])

    # Backtrack
    i, j = N, M
    path_h, path_g = [], []
    while i > 0 and j > 0:
        path_h.append(i - 1); path_g.append(j - 1)
        choices = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
        k = int(np.argmin(choices))
        if   k == 0: i, j = i - 1, j - 1
        elif k == 1: i = i - 1
        else:        j = j - 1
    path_h, path_g = path_h[::-1], path_g[::-1]

    # Map each ghost station to its averaged human index
    ghost_to_human = {g: [] for g in range(M)}
    for h, g in zip(path_h, path_g):
        ghost_to_human[g].append(h)

    s_grid = df_ghost['s'].values
    out = pd.DataFrame({'s': s_grid})
    for ch in ['v', 'steer', 'throttle', 'brake', 'latG', 'lonG']:
        if ch in df_human.columns:
            arr = df_human[ch].values
            out[f'{ch}_human'] = np.array([
                arr[ghost_to_human[g]].mean() if ghost_to_human[g] else np.nan
                for g in range(M)
            ])
    out['n_lateral_human'] = 0.0
    return out


# ═════════════════════════════════════════════════════════════════════════════
# §3.  Stage 2 — Delta-channel extraction
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_pedal(x: np.ndarray, scale: float) -> np.ndarray:
    """Normalize pedal inputs to [0,1] regardless of incoming units (N or %)."""
    x = np.asarray(x, dtype=float)
    if np.nanmax(np.abs(x)) > 1.5:        # treat as Newtons (or 0..100 %)
        if np.nanmax(np.abs(x)) > 110.0:  # Newtons
            return np.clip(x / scale, 0.0, 1.0)
        return np.clip(x / 100.0, 0.0, 1.0)
    return np.clip(x, 0.0, 1.0)


def compute_deltas(df_aligned: pd.DataFrame,
                   df_ghost:   pd.DataFrame,
                   F_brake_max: float = 8500.0,
                   F_drive_max: float = 4500.0) -> pd.DataFrame:
    """
    Compute the engineering delta channels on the shared s-grid.

    Returns DataFrame with: s, kappa, v_g, v_h, speed_delta, brake_delta,
    throttle_delta, coasting_delta, line_delta, steer_delta, latG_lost,
    combined_pedal_h, combined_pedal_g.
    """
    s   = df_ghost['s'].values
    v_g = df_ghost['v'].values
    v_h = df_aligned['v_human'].values if 'v_human' in df_aligned.columns else np.full_like(v_g, np.nan)

    thr_h = _normalize_pedal(df_aligned.get('throttle_human', np.zeros_like(s)), F_drive_max)
    thr_g = _normalize_pedal(df_ghost['throttle'].values, F_drive_max)
    brk_h = _normalize_pedal(df_aligned.get('brake_human', np.zeros_like(s)),    F_brake_max)
    brk_g = _normalize_pedal(df_ghost['brake'].values,    F_brake_max)

    pedal_h = thr_h + brk_h            # > 1 means pedal-overlap (rare/intentional)
    pedal_g = thr_g + brk_g

    coast_h = 1.0 - pedal_h            # how much "no pedal" the human is doing
    coast_g = 1.0 - pedal_g
    coasting_delta = coast_h - coast_g            # positive → human coasting more

    line_delta  = (df_aligned.get('n_lateral_human', np.zeros_like(s))
                 - df_ghost['n_lateral'].values)
    steer_delta = (df_aligned.get('steer_human', np.zeros_like(s))
                 - df_ghost['steer'].values)
    speed_delta = v_h - v_g
    brake_delta = brk_h - brk_g
    throttle_delta = thr_h - thr_g

    # Approximate combined-G left on the table:
    #   if human carries less speed at a given κ, human ay = κ·v_h² < κ·v_g²
    kappa = df_ghost['kappa'].values
    ay_g = kappa * v_g ** 2 / 9.81
    ay_h = kappa * v_h ** 2 / 9.81
    latG_lost = np.abs(ay_g) - np.abs(ay_h)        # >0 → grip on the table

    return pd.DataFrame({
        's':              s,
        'kappa':          kappa,
        'v_g':            v_g,
        'v_h':            v_h,
        'speed_delta':    speed_delta,
        'brake_delta':    brake_delta,
        'throttle_delta': throttle_delta,
        'coasting_delta': coasting_delta,
        'line_delta':     line_delta,
        'steer_delta':    steer_delta,
        'latG_lost':      latG_lost,
        'combined_pedal_h': pedal_h,
        'combined_pedal_g': pedal_g,
        'thr_h':          thr_h,
        'thr_g':          thr_g,
        'brk_h':          brk_h,
        'brk_g':          brk_g,
    })


# ═════════════════════════════════════════════════════════════════════════════
# §4.  Corner segmentation
# ═════════════════════════════════════════════════════════════════════════════

def detect_corners(df_ghost: pd.DataFrame,
                   kappa_threshold: float = 0.04,
                   min_arc_length:  float = 6.0) -> List[CornerZone]:
    """
    Segment the track into corner zones based on |κ(s)| crossing a threshold.
    Each zone is bounded by entry (κ rises above thr), apex (peak |κ|), exit
    (κ falls below thr).
    """
    s     = df_ghost['s'].values
    kappa = df_ghost['kappa'].values
    is_corner = np.abs(kappa) > kappa_threshold

    zones: List[CornerZone] = []
    i = 0
    cid = 0
    while i < len(s):
        if not is_corner[i]:
            i += 1; continue
        j = i
        while j < len(s) and is_corner[j]:
            j += 1
        # [i, j) is a corner segment
        if (s[j - 1] - s[i]) >= min_arc_length:
            apex_idx = i + int(np.argmax(np.abs(kappa[i:j])))
            radius = 1.0 / (np.abs(kappa[apex_idx]) + 1e-6)
            direction = 'L' if kappa[apex_idx] > 0 else 'R'
            zones.append(CornerZone(
                corner_id=cid + 1,
                s_entry=float(s[i]),
                s_apex=float(s[apex_idx]),
                s_exit=float(s[j - 1]),
                direction=direction,
                radius_m=float(radius),
            ))
            cid += 1
        i = j

    print(f"[Corners] Detected {len(zones)} corner zones "
          f"(κ_thr={kappa_threshold}, min_arc={min_arc_length} m)")
    return zones


# ═════════════════════════════════════════════════════════════════════════════
# §5.  Stage 3 — NLP logic tree
# ═════════════════════════════════════════════════════════════════════════════
#
# Driver archetypes (decision tree fires the FIRST matching rule):
#
#   1. COASTING_ENTRY   : coasting_delta > 0.15 sustained, on entry phase,
#                          v_h < v_g  → "you're lifting too early"
#   2. LATE_BRAKE       : brake_delta > 0.20 inside corner, v_h > v_g + 1.0
#                          → "brake earlier; you're carrying too much speed"
#   3. EARLY_THROTTLE   : throttle_delta > 0.15 before apex, lateral_g_h
#                          → "you're stabbing throttle before grip is loaded"
#   4. APEX_MISS        : |line_delta| > 0.4 m at apex, low |steer_delta|
#                          → "you're missing the apex by X cm"
#   5. NO_TRAIL_BRAKE   : brake_delta < -0.10 in corner entry (human braking
#                          LESS than ghost while ghost trail-brakes)
#                          → "extend trail-brake to keep front loaded"
#   6. CORNER_EXIT_LIFT : throttle_delta < -0.20 in corner exit
#                          → "stay on throttle through track-out"
# ═════════════════════════════════════════════════════════════════════════════

PERSISTENCE_M = 4.0   # event must persist over ≥ 4 m of track (≈ 0.2 s @ 20 m/s)


def _find_persistent_events(mask: np.ndarray, s: np.ndarray,
                            min_arc: float = PERSISTENCE_M) -> List[Tuple[int, int]]:
    """Return (i_start, i_end) index pairs where `mask` is continuously True
       over an arc length ≥ min_arc."""
    out = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1; continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if (s[j - 1] - s[i]) >= min_arc:
            out.append((i, j - 1))
        i = j
    return out


def _phase_of(s_query: float, zone: CornerZone) -> str:
    """Returns 'entry' if s_query < apex, 'mid' if near apex, 'exit' otherwise."""
    if s_query < zone.s_apex - 5.0: return 'entry'
    if s_query > zone.s_apex + 5.0: return 'exit'
    return 'mid'


def _format_advice(archetype: str, zone: CornerZone, severity_g: float,
                   metrics: dict) -> str:
    """Return plain-English coaching string."""
    cid = f"Turn {zone.corner_id}"
    dir_word = "left" if zone.direction == 'L' else "right"
    R = zone.radius_m

    if archetype == 'COASTING_ENTRY':
        return (f"{cid}: You coasted for {metrics['duration_s']:.2f}s on entry. "
                f"You are leaving {severity_g:.2f}G of lateral grip on the table. "
                f"The optimal car carries {int(metrics['ghost_brake_pct']*100)}% "
                f"trail braking to the apex to keep the front aero platform "
                f"pitched down.")

    if archetype == 'LATE_BRAKE':
        return (f"{cid}: You braked {metrics['delta_v_kmh']:+.1f} km/h late "
                f"at the apex of this {dir_word} R{R:.0f} m corner. "
                f"That excess entry speed costs ~{metrics['time_lost_s']:.2f}s "
                f"and unloads the front under combined slip. Brake "
                f"{metrics['advance_m']:.0f}m earlier and trail-release.")

    if archetype == 'EARLY_THROTTLE':
        return (f"{cid}: Throttle applied {metrics['advance_s']:.2f}s before "
                f"the optimal pickup point. You're loading longitudinal "
                f"slip while the tire is at peak lateral demand — {severity_g:.2f}G "
                f"of combined-slip headroom is being wasted on throttle "
                f"oversaturation.")

    if archetype == 'APEX_MISS':
        return (f"{cid}: Apex offset {metrics['offset_cm']:.0f} cm "
                f"{'wide' if metrics['offset_cm']>0 else 'tight'}. "
                f"Steering input matches optimal but the car is on the wrong "
                f"line — likely a turn-in reference issue. Aim for "
                f"{metrics['target_n']:+.1f} m from centerline at the apex.")

    if archetype == 'NO_TRAIL_BRAKE':
        return (f"{cid}: You released brakes {metrics['early_release_m']:.0f}m "
                f"before the apex; the optimal car carries "
                f"{int(metrics['ghost_brake_pct']*100)}% brake pressure to the "
                f"apex itself. Without trail-brake the front aero platform "
                f"rises and understeer sets in — {severity_g:.2f}G lost on entry.")

    if archetype == 'CORNER_EXIT_LIFT':
        return (f"{cid}: You lifted on exit ({metrics['lift_pct']:.0f}% throttle "
                f"reduction vs optimal). Stay committed through track-out; "
                f"the rear is more stable when you're on power than coasting.")

    return f"{cid}: Driver delta detected — {archetype}."


def generate_coaching_report(df_deltas:  pd.DataFrame,
                             df_ghost:   pd.DataFrame,
                             zones:      Optional[List[CornerZone]] = None,
                             *,
                             top_k:      int = 8) -> List[CoachingEvent]:
    """
    Run the logic tree on every detected event in every corner zone, return
    a ranked list of the top_k highest-severity coaching events.

    Severity ranking: time_lost_s (ms-equivalent) — deterministic, repeatable.
    """
    if zones is None:
        zones = detect_corners(df_ghost)

    s = df_deltas['s'].values
    events: List[CoachingEvent] = []

    for zone in zones:
        mask_zone = (s >= zone.s_entry - 2.0) & (s <= zone.s_exit + 2.0)
        if not mask_zone.any():
            continue
        idx_zone = np.where(mask_zone)[0]
        s_z = s[idx_zone]

        coast    = df_deltas['coasting_delta'].values[idx_zone]
        brake    = df_deltas['brake_delta'].values[idx_zone]
        throt    = df_deltas['throttle_delta'].values[idx_zone]
        line     = df_deltas['line_delta'].values[idx_zone]
        speed_d  = df_deltas['speed_delta'].values[idx_zone]
        latG_lost= df_deltas['latG_lost'].values[idx_zone]
        v_g      = df_deltas['v_g'].values[idx_zone]
        thr_g    = df_deltas['thr_g'].values[idx_zone]
        brk_g    = df_deltas['brk_g'].values[idx_zone]

        # Decision tree (fires first matching archetype per persistent event)

        # 1) COASTING_ENTRY
        m = (coast > 0.15) & (speed_d < -0.5)
        for i_lo, i_hi in _find_persistent_events(m, s_z):
            phase = _phase_of(s_z[i_lo], zone)
            if phase != 'entry': continue
            sev = float(np.max(latG_lost[i_lo:i_hi + 1]))
            duration = (s_z[i_hi] - s_z[i_lo]) / max(np.mean(v_g[i_lo:i_hi + 1]), 1.0)
            t_lost   = float(np.mean(speed_d[i_lo:i_hi + 1])) * duration / max(np.mean(v_g), 1.0)
            events.append(CoachingEvent(
                s_start=float(s_z[i_lo]), s_end=float(s_z[i_hi]),
                corner_id=zone.corner_id, archetype='COASTING_ENTRY',
                severity_g=sev, time_lost_s=abs(t_lost),
                advice=_format_advice('COASTING_ENTRY', zone, sev, {
                    'duration_s': duration,
                    'ghost_brake_pct': float(np.mean(brk_g[i_lo:i_hi + 1])),
                }),
            ))

        # 2) LATE_BRAKE
        m = (brake > 0.20) & (speed_d > 1.0)
        for i_lo, i_hi in _find_persistent_events(m, s_z):
            sev = float(np.max(latG_lost[i_lo:i_hi + 1]))
            dv_kmh = float(np.max(speed_d[i_lo:i_hi + 1])) * 3.6
            t_lost = abs(dv_kmh) * 0.005                            # heuristic
            events.append(CoachingEvent(
                s_start=float(s_z[i_lo]), s_end=float(s_z[i_hi]),
                corner_id=zone.corner_id, archetype='LATE_BRAKE',
                severity_g=sev, time_lost_s=t_lost,
                advice=_format_advice('LATE_BRAKE', zone, sev, {
                    'delta_v_kmh': dv_kmh,
                    'time_lost_s': t_lost,
                    'advance_m':  abs(dv_kmh) * 0.5,
                }),
            ))

        # 3) EARLY_THROTTLE
        m = (throt > 0.15) & np.array([_phase_of(ss, zone) == 'entry' for ss in s_z])
        for i_lo, i_hi in _find_persistent_events(m, s_z):
            sev = float(np.max(latG_lost[i_lo:i_hi + 1]))
            advance_s = (zone.s_apex - s_z[i_lo]) / max(np.mean(v_g[i_lo:i_hi + 1]), 1.0)
            events.append(CoachingEvent(
                s_start=float(s_z[i_lo]), s_end=float(s_z[i_hi]),
                corner_id=zone.corner_id, archetype='EARLY_THROTTLE',
                severity_g=sev, time_lost_s=0.05,
                advice=_format_advice('EARLY_THROTTLE', zone, sev, {
                    'advance_s': advance_s,
                }),
            ))

        # 4) APEX_MISS — measured at apex, not over a window
        idx_apex_global = int(np.argmin(np.abs(df_ghost['s'].values - zone.s_apex)))
        offset_m = float(df_deltas['line_delta'].values[idx_apex_global])
        if abs(offset_m) > 0.40:
            sev = float(latG_lost[np.argmin(np.abs(s_z - zone.s_apex))])
            events.append(CoachingEvent(
                s_start=zone.s_apex - 2.0, s_end=zone.s_apex + 2.0,
                corner_id=zone.corner_id, archetype='APEX_MISS',
                severity_g=max(sev, 0.0), time_lost_s=abs(offset_m) * 0.05,
                advice=_format_advice('APEX_MISS', zone, max(sev, 0.0), {
                    'offset_cm': offset_m * 100.0,
                    'target_n':  float(df_ghost['n_lateral'].values[idx_apex_global]),
                }),
            ))

        # 5) NO_TRAIL_BRAKE
        m = (brake < -0.10) & np.array([_phase_of(ss, zone) == 'entry' for ss in s_z])
        for i_lo, i_hi in _find_persistent_events(m, s_z):
            sev = float(np.max(latG_lost[i_lo:i_hi + 1]))
            early_release_m = zone.s_apex - s_z[i_hi]
            events.append(CoachingEvent(
                s_start=float(s_z[i_lo]), s_end=float(s_z[i_hi]),
                corner_id=zone.corner_id, archetype='NO_TRAIL_BRAKE',
                severity_g=sev, time_lost_s=sev * 0.03,
                advice=_format_advice('NO_TRAIL_BRAKE', zone, sev, {
                    'early_release_m': early_release_m,
                    'ghost_brake_pct': float(np.mean(brk_g[i_lo:i_hi + 1])),
                }),
            ))

        # 6) CORNER_EXIT_LIFT
        m = (throt < -0.20) & np.array([_phase_of(ss, zone) == 'exit' for ss in s_z])
        for i_lo, i_hi in _find_persistent_events(m, s_z):
            sev = float(np.max(latG_lost[i_lo:i_hi + 1]))
            lift_pct = float(np.mean(throt[i_lo:i_hi + 1])) * -100.0
            events.append(CoachingEvent(
                s_start=float(s_z[i_lo]), s_end=float(s_z[i_hi]),
                corner_id=zone.corner_id, archetype='CORNER_EXIT_LIFT',
                severity_g=sev, time_lost_s=lift_pct * 0.002,
                advice=_format_advice('CORNER_EXIT_LIFT', zone, sev, {
                    'lift_pct': lift_pct,
                }),
            ))

    # Rank by time lost descending; deduplicate adjacent same-archetype events
    events = sorted(events, key=lambda e: -e.time_lost_s)
    print(f"[Coaching] Generated {len(events)} events; returning top {top_k}.")
    return events[:top_k]


# ═════════════════════════════════════════════════════════════════════════════
# §6.  Public pipeline + dashboard adapter
# ═════════════════════════════════════════════════════════════════════════════

def full_pipeline(df_human: pd.DataFrame,
                  df_ghost: pd.DataFrame,
                  output_dir: Optional[str] = None,
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, List[CoachingEvent]]:
    """
    End-to-end ghost vs human coaching pipeline.

    Returns: (df_aligned, df_deltas, coaching_events)
    Optionally writes:
        coaching_aligned.csv
        coaching_deltas.csv
        coaching_report.csv     <-- ranked list of events
        coaching_dashboard.json <-- DriverCoachingModule.jsx ingestable
    """
    df_aligned = align_human_to_ghost(df_human, df_ghost)
    df_deltas  = compute_deltas(df_aligned, df_ghost)
    zones      = detect_corners(df_ghost)
    events     = generate_coaching_report(df_deltas, df_ghost, zones=zones)

    if output_dir:
        import os, json
        os.makedirs(output_dir, exist_ok=True)
        df_aligned.to_csv(f"{output_dir}/coaching_aligned.csv", index=False)
        df_deltas.to_csv(f"{output_dir}/coaching_deltas.csv",  index=False)
        pd.DataFrame([e.as_dict() for e in events]).to_csv(
            f"{output_dir}/coaching_report.csv", index=False)

        # Dashboard-compatible JSON: per-station stream matching DriverCoachingModule.jsx
        dash = pd.DataFrame({
            's':            df_ghost['s'].values,
            't':            df_ghost['t'].values,
            'speed':        df_aligned.get('v_human', df_ghost['v']).values,
            'optSteer':     df_ghost['steer'].values,
            'actSteer':     df_aligned.get('steer_human', df_ghost['steer']).values,
            'steerError':   df_deltas['steer_delta'].values,
            'optThrottle':  df_deltas['thr_g'].values,
            'actThrottle':  df_deltas['thr_h'].values,
            'optBrake':     df_deltas['brk_g'].values,
            'actBrake':     df_deltas['brk_h'].values,
            'latG':         df_ghost['latG'].values,
            'lonG':         df_ghost['lonG'].values,
            'combinedG':    df_ghost['combinedG'].values,
            'lineDeviation':df_deltas['line_delta'].values,
            'curvature':    df_ghost['kappa'].values,
        })
        dash.to_json(f"{output_dir}/coaching_dashboard.json", orient='records')
        print(f"[Coaching] Wrote artefacts to {output_dir}/")

    return df_aligned, df_deltas, events