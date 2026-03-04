"""
simulator/track_builder.py
─────────────────────────────────────────────────────────────────────────────
Procedural FSAE track generation.

Generates track centreline, boundaries, and timing lines from:
  · Named event presets (skidpad, autocross layouts)
  · Programmatic segment sequences (straight/arc)
  · Serialised JSON layouts (for team-shared track files)

Output: Track object containing:
  · Centreline (x, y, psi, kappa) arrays at 1m spacing
  · Left / right boundary offsets
  · Cone positions for visualisation
  · TimingLine objects for LapTimer
  · Sector waypoints

Uses the same track layouts as lap_simulator.py (FS_TRACKS) but adds 2D
geometry, cone positions, and per-corner track widths.
"""

import numpy as np
import json
import os
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field

try:
    from simulator.lap_timer import TimingLine, LapTimer
except ImportError:
    from lap_timer import TimingLine, LapTimer


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Cone:
    x     : float
    y     : float
    color : str   # 'yellow' | 'blue' | 'orange' | 'big_orange'


@dataclass
class Track:
    """Complete track descriptor."""
    name         : str
    # Centreline at 1m intervals
    cx           : np.ndarray    # x  [m]
    cy           : np.ndarray    # y  [m]
    cpsi         : np.ndarray    # heading [rad]
    ck           : np.ndarray    # curvature [1/m]
    # Boundaries (offset from centreline normal)
    width_left   : np.ndarray    # [m]  positive = left
    width_right  : np.ndarray    # [m]  positive = right (stored positive)
    # Track length
    total_length : float         # [m]
    # Cones
    cones_left   : List[Cone]    = field(default_factory=list)
    cones_right  : List[Cone]    = field(default_factory=list)
    cones_orange : List[Cone]    = field(default_factory=list)   # S/F line
    # Timing
    finish_line  : Optional[TimingLine] = None
    sector_lines : List[TimingLine]     = field(default_factory=list)
    # Start position
    start_x      : float = 0.0
    start_y      : float = 0.0
    start_yaw    : float = 0.0

    def get_start_pose(self) -> Tuple[float, float, float]:
        return self.start_x, self.start_y, self.start_yaw

    def get_closest_point(self, x: float, y: float) -> Tuple[int, float]:
        """Return (index, distance) of closest centreline point."""
        dx = self.cx - x; dy = self.cy - y
        dists = np.sqrt(dx*dx + dy*dy)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def get_local_curvature(self, x: float, y: float) -> float:
        idx, _ = self.get_closest_point(x, y)
        return float(self.ck[idx])

    def to_json(self, path: str):
        d = {
            'name': self.name,
            'cx': self.cx.tolist(), 'cy': self.cy.tolist(),
            'cpsi': self.cpsi.tolist(), 'ck': self.ck.tolist(),
            'width_left': self.width_left.tolist(),
            'width_right': self.width_right.tolist(),
            'total_length': self.total_length,
            'start_x': self.start_x, 'start_y': self.start_y,
            'start_yaw': self.start_yaw,
            'cones_left':  [{'x':c.x,'y':c.y,'color':c.color} for c in self.cones_left],
            'cones_right': [{'x':c.x,'y':c.y,'color':c.color} for c in self.cones_right],
            'cones_orange':[{'x':c.x,'y':c.y,'color':c.color} for c in self.cones_orange],
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
        print(f"[TrackBuilder] Saved → {path}")

    @classmethod
    def from_json(cls, path: str) -> 'Track':
        with open(path) as f:
            d = json.load(f)
        track = cls(
            name         = d['name'],
            cx           = np.array(d['cx']),
            cy           = np.array(d['cy']),
            cpsi         = np.array(d['cpsi']),
            ck           = np.array(d['ck']),
            width_left   = np.array(d['width_left']),
            width_right  = np.array(d['width_right']),
            total_length = d['total_length'],
            start_x      = d.get('start_x', 0.0),
            start_y      = d.get('start_y', 0.0),
            start_yaw    = d.get('start_yaw', 0.0),
            cones_left   = [Cone(**c) for c in d.get('cones_left',  [])],
            cones_right  = [Cone(**c) for c in d.get('cones_right', [])],
            cones_orange = [Cone(**c) for c in d.get('cones_orange',[])],
        )
        return track


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

class TrackBuilder:
    """
    Build a Track from a sequence of segment primitives.

    Segment types:
        straight(length_m)
        arc(radius_m, angle_deg, direction)  direction: 'L' or 'R'
        chicane(width_m, length_m)           double-apex S-curve
    """

    CONE_SPACING_STRAIGHT = 5.0   # m between cones on straights
    CONE_SPACING_CORNER   = 3.0   # m between cones in corners
    DS = 0.5                       # centreline discretisation step [m]

    def __init__(self, name: str = "custom",
                 default_half_width: float = 3.5,
                 start_x: float = 0.0, start_y: float = 0.0,
                 start_yaw: float = 0.0):
        self.name             = name
        self.default_hw       = default_half_width
        self.start_x          = start_x
        self.start_y          = start_y
        self.start_yaw        = start_yaw

        # Builder state
        self._x    = start_x
        self._y    = start_y
        self._psi  = start_yaw

        # Accumulated centreline
        self._pts_x   : List[float] = [start_x]
        self._pts_y   : List[float] = [start_y]
        self._pts_psi : List[float] = [start_yaw]
        self._pts_k   : List[float] = [0.0]
        self._pts_wl  : List[float] = [default_half_width]
        self._pts_wr  : List[float] = [default_half_width]

        # Sector waypoints (arc_length, label)
        self._sector_arclens : List[float] = []
        self._current_arclen  = 0.0

    # ── Primitives ─────────────────────────────────────────────────────────────

    def straight(self, length: float, half_width: Optional[float] = None) -> 'TrackBuilder':
        """Add a straight of given length [m]."""
        hw = half_width or self.default_hw
        n  = max(2, int(length / self.DS))
        for _ in range(n):
            self._x += self.DS * np.cos(self._psi)
            self._y += self.DS * np.sin(self._psi)
            self._pts_x.append(self._x);   self._pts_y.append(self._y)
            self._pts_psi.append(self._psi); self._pts_k.append(0.0)
            self._pts_wl.append(hw);        self._pts_wr.append(hw)
            self._current_arclen += self.DS
        return self

    def arc(self, radius: float, angle_deg: float, direction: str = 'L',
            half_width: Optional[float] = None) -> 'TrackBuilder':
        """
        Add a circular arc.
        radius    : [m] — inner cone radius ≈ radius − half_width
        angle_deg : total angle swept [deg]
        direction : 'L' (left/CCW) or 'R' (right/CW)
        """
        hw   = half_width or self.default_hw
        sign = 1.0 if direction.upper() == 'L' else -1.0
        k    = sign / max(radius, 0.5)
        arc_len = abs(np.deg2rad(angle_deg)) * radius
        n    = max(2, int(arc_len / self.DS))
        dpsi = (np.deg2rad(angle_deg) * sign) / n

        for _ in range(n):
            self._psi += dpsi
            self._x   += self.DS * np.cos(self._psi - dpsi / 2)
            self._y   += self.DS * np.sin(self._psi - dpsi / 2)
            self._pts_x.append(self._x);    self._pts_y.append(self._y)
            self._pts_psi.append(self._psi); self._pts_k.append(k)
            self._pts_wl.append(hw);         self._pts_wr.append(hw)
            self._current_arclen += self.DS
        return self

    def chicane(self, offset: float = 3.0, length: float = 20.0,
                half_width: Optional[float] = None) -> 'TrackBuilder':
        """
        Add a slalom chicane: slight left-right curvature over `length` metres
        with a lateral `offset` at the apex.

        Geometry: chord = length/2 (half chicane), sagitta = offset.
        Radius of curvature: r = ((L/2)² + d²) / (2d)
        Half-subtended angle:  a = arcsin((L/2) / r)   [always ≤ 1 by construction]
        """
        hw     = half_width or self.default_hw
        half_L = length / 2.0
        # Correct chord-sagitta formula (was length²/8 — wrong by factor 2)
        r      = (half_L ** 2 + offset ** 2) / (2.0 * max(offset, 0.1))
        r      = max(r, half_L * 1.01)          # guarantee r > half-chord → arcsin safe
        sin_a  = min(half_L / r, 1.0)           # clamp for float safety
        a      = np.degrees(np.arcsin(sin_a))
        self.arc(r, a, 'L', hw).arc(r, a * 2, 'R', hw).arc(r, a, 'L', hw)
        return self

    def add_sector_split(self) -> 'TrackBuilder':
        """Mark the current arc-length as a sector boundary."""
        self._sector_arclens.append(self._current_arclen)
        return self

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self) -> Track:
        cx   = np.array(self._pts_x)
        cy   = np.array(self._pts_y)
        cpsi = np.array(self._pts_psi)
        ck   = np.array(self._pts_k)
        wl   = np.array(self._pts_wl)
        wr   = np.array(self._pts_wr)

        total_length = float(len(cx)) * self.DS

        # Smooth curvature with Gaussian filter to avoid artefacts at segment joints
        from scipy.ndimage import gaussian_filter1d
        ck = gaussian_filter1d(ck, sigma=3.0)

        # ── Generate cone positions ────────────────────────────────────────
        cones_left  : List[Cone] = []
        cones_right : List[Cone] = []
        cones_orange: List[Cone] = []
        last_cone_s = -99.0

        for i in range(len(cx)):
            k_abs   = abs(ck[i])
            spacing = self.CONE_SPACING_CORNER if k_abs > 0.03 else self.CONE_SPACING_STRAIGHT
            s = i * self.DS
            if s - last_cone_s >= spacing:
                psi_n = cpsi[i] + np.pi / 2
                lx = cx[i] + wl[i] * np.cos(psi_n)
                ly = cy[i] + wl[i] * np.sin(psi_n)
                rx = cx[i] - wr[i] * np.cos(psi_n)
                ry = cy[i] - wr[i] * np.sin(psi_n)
                cones_left.append(Cone(lx, ly, 'yellow'))
                cones_right.append(Cone(rx, ry, 'blue'))
                last_cone_s = s

        # ── Finish line (across the track at s=0) ─────────────────────────
        psi_n_sf = cpsi[0] + np.pi / 2
        fl = TimingLine(
            x0 = cx[0] + wl[0] * np.cos(psi_n_sf),
            y0 = cy[0] + wl[0] * np.sin(psi_n_sf),
            x1 = cx[0] - wr[0] * np.cos(psi_n_sf),
            y1 = cy[0] - wr[0] * np.sin(psi_n_sf),
            label = "S/F",
        )
        # Big orange cones at S/F
        cones_orange.append(Cone(fl.x0, fl.y0, 'big_orange'))
        cones_orange.append(Cone(fl.x1, fl.y1, 'big_orange'))

        # ── Sector lines ──────────────────────────────────────────────────
        sector_lines = []
        for arc_s in self._sector_arclens:
            i = min(int(arc_s / self.DS), len(cx) - 1)
            psi_n_s = cpsi[i] + np.pi / 2
            sl = TimingLine(
                x0 = cx[i] + wl[i] * np.cos(psi_n_s),
                y0 = cy[i] + wl[i] * np.sin(psi_n_s),
                x1 = cx[i] - wr[i] * np.cos(psi_n_s),
                y1 = cy[i] - wr[i] * np.sin(psi_n_s),
                label = f"S{len(sector_lines)+1}",
            )
            sector_lines.append(sl)
            cones_orange.append(Cone(sl.x0, sl.y0, 'big_orange'))
            cones_orange.append(Cone(sl.x1, sl.y1, 'big_orange'))

        track = Track(
            name          = self.name,
            cx            = cx, cy = cy, cpsi = cpsi, ck = ck,
            width_left    = wl, width_right = wr,
            total_length  = total_length,
            cones_left    = cones_left,
            cones_right   = cones_right,
            cones_orange  = cones_orange,
            finish_line   = fl,
            sector_lines  = sector_lines,
            start_x       = self.start_x,
            start_y       = self.start_y,
            start_yaw     = self.start_yaw,
        )
        return track


# ─────────────────────────────────────────────────────────────────────────────
# Named event presets
# ─────────────────────────────────────────────────────────────────────────────

def build_skidpad() -> Track:
    """
    FSAE Skidpad — two concentric circles, R=7.625m centreline.
    Standard layout per FSAE rules.
    """
    R  = 9.125   # outer circle centreline (inner = 7.625, outer = 9.125, w=1.5m)
    b = TrackBuilder("Skidpad", default_half_width=1.5,
                     start_x=0.0, start_y=0.0, start_yaw=np.pi/2)
    # Entry straight
    b.straight(15.0)
    # Two left loops (4 laps total in comp but we do 2 for timing)
    b.arc(R, 360, 'L', 1.5)
    b.arc(R, 360, 'L', 1.5)
    # Cross to right circles
    b.straight(2.0)
    b.arc(R, 360, 'R', 1.5)
    b.arc(R, 360, 'R', 1.5)
    # Exit straight
    b.straight(15.0)
    return b.build()


def build_fsg_autocross() -> Track:
    """
    FS Germany autocross layout approximation.
    Based on published 2023 event maps: ~750m, tight hairpins, one fast sweeper.
    3 sectors: sector 1 = hairpin zone, sector 2 = slalom, sector 3 = sweeper.
    """
    b = TrackBuilder("FSG Autocross", default_half_width=3.5,
                     start_x=0.0, start_y=0.0, start_yaw=0.0)
    b.straight(30)
    b.arc(9, 180, 'L')        # hairpin 1
    b.straight(20)
    b.arc(18, 90, 'R')         # right sweeper
    b.straight(12)
    b.arc(9, 90, 'L')
    b.straight(10)
    b.arc(8, 90, 'R')
    b.straight(10)
    b.add_sector_split()       # Sector 1 boundary
    b.chicane(3.0, 18.0)       # slalom section
    b.straight(15)
    b.chicane(3.5, 20.0)
    b.straight(12)
    b.add_sector_split()       # Sector 2 boundary
    b.arc(25, 120, 'L', 4.0)  # fast sweeper
    b.straight(40)
    b.arc(9, 90, 'R')
    b.straight(10)
    b.arc(9, 90, 'L')
    b.straight(25)
    return b.build()


def build_endurance_lap() -> Track:
    """
    Generic 1km FSAE endurance lap — balanced mix of speeds.
    Used for multi-lap wear simulation.
    """
    b = TrackBuilder("Endurance Lap", default_half_width=4.0,
                     start_x=0.0, start_y=0.0, start_yaw=0.0)
    b.straight(80)
    b.arc(18, 90, 'L')
    b.straight(12)
    b.arc(18, 90, 'R')
    b.straight(70)
    b.arc(10, 90, 'L')
    b.straight(10)
    b.arc(10, 90, 'R')
    b.add_sector_split()
    b.straight(80)
    b.arc(12, 90, 'L')
    b.straight(10)
    b.arc(12, 90, 'L')
    b.straight(80)
    b.arc(8,  180, 'R')        # tight hairpin
    b.add_sector_split()
    b.straight(80)
    b.arc(16, 90, 'L')
    b.straight(14)
    b.arc(16, 90, 'R')
    b.straight(80)
    return b.build()


def build_acceleration() -> Track:
    """75m acceleration event — straight line only."""
    b = TrackBuilder("Acceleration 75m", default_half_width=5.0,
                     start_x=0.0, start_y=0.0, start_yaw=0.0)
    b.straight(80)
    return b.build()


# ── Registry ────────────────────────────────────────────────────────────────
TRACK_REGISTRY = {
    'skidpad'         : build_skidpad,
    'fsg_autocross'   : build_fsg_autocross,
    'endurance_lap'   : build_endurance_lap,
    'acceleration'    : build_acceleration,
}


def get_track(name: str) -> Track:
    """Load a named preset or a JSON file path."""
    if name in TRACK_REGISTRY:
        track = TRACK_REGISTRY[name]()
        print(f"[TrackBuilder] Built '{name}': {track.total_length:.0f}m | "
              f"{len(track.cones_left)+len(track.cones_right)} cones | "
              f"{len(track.sector_lines)} sectors")
        return track
    elif os.path.isfile(name):
        return Track.from_json(name)
    else:
        raise ValueError(f"Unknown track '{name}'. "
                         f"Options: {list(TRACK_REGISTRY.keys())} or a .json path")


if __name__ == "__main__":
    # Build and save all presets
    for nm, fn in TRACK_REGISTRY.items():
        t = fn()
        out = f"tracks/{nm}.json"
        os.makedirs("tracks", exist_ok=True)
        t.to_json(out)
        print(f"  {nm}: {t.total_length:.0f}m, {len(t.cones_left)} left cones")
