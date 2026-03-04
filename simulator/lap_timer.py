"""
simulator/lap_timer.py
─────────────────────────────────────────────────────────────────────────────
Real-time lap timer with sector splits and best-lap tracking.

A "lap" is defined by a start/finish line crossing (configurable position +
heading tolerance).  Sectors are defined by intermediate waypoints.

The timer works purely from the vehicle's (X, Y) position — no external
trigger required.  It detects line crossings via signed-distance transitions.
"""

import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SectorSplit:
    sector_idx  : int
    split_time  : float   # time since lap start at sector crossing [s]
    sim_time    : float   # absolute simulation time [s]
    delta_best  : Optional[float] = None   # diff vs best lap's sector time [s]


@dataclass
class LapRecord:
    lap_number  : int
    lap_time    : float    # [s]
    sector_times: List[float] = field(default_factory=list)
    max_lat_g   : float   = 0.0
    max_lon_g   : float   = 0.0
    max_speed   : float   = 0.0
    energy_kj   : float   = 0.0
    valid       : bool    = True


@dataclass
class TimingLine:
    """A line segment that triggers a lap or sector crossing."""
    x0: float; y0: float   # start of line segment
    x1: float; y1: float   # end of line segment
    label: str = "S/F"

    @property
    def normal(self) -> np.ndarray:
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        L = max(np.sqrt(dx*dx + dy*dy), 1e-9)
        return np.array([-dy / L, dx / L])   # normal pointing "into" track

    def signed_distance(self, px: float, py: float) -> float:
        n = self.normal
        return n[0] * (px - self.x0) + n[1] * (py - self.y0)

    def check_crossing(self, prev_x, prev_y, curr_x, curr_y) -> bool:
        """
        Returns True if the car crossed from negative to positive side.
        Uses signed distance sign change + proximity check.
        """
        d_prev = self.signed_distance(prev_x, prev_y)
        d_curr = self.signed_distance(curr_x, curr_y)
        if d_prev < 0 <= d_curr:
            # Verify crossing is within line segment bounds
            cx = (prev_x + curr_x) / 2
            cy = (prev_y + curr_y) / 2
            # Project onto line direction
            dx, dy = self.x1 - self.x0, self.y1 - self.y0
            L = max(np.sqrt(dx*dx + dy*dy), 1e-9)
            t = ((cx - self.x0) * dx + (cy - self.y0) * dy) / (L*L)
            return -0.2 <= t <= 1.2   # small margin for near-miss detection
        return False


class LapTimer:
    """
    Real-time lap and sector timing.

    Detects start/finish and sector line crossings from (X, Y) position.
    Maintains best-lap history and computes delta-T vs best lap.
    """

    MIN_LAP_TIME = 20.0    # [s] ignore crossings within this of last start
    SECTOR_COOLDOWN = 5.0  # [s] minimum time between same-sector triggers

    def __init__(self, finish_line: Optional[TimingLine] = None,
                 sector_lines: Optional[List[TimingLine]] = None):
        """
        Args:
            finish_line:  TimingLine for start/finish. If None, timing is
                          inactive until set_track() is called.
            sector_lines: List of intermediate sector lines (in track order).
        """
        self.finish_line  = finish_line
        self.sector_lines = sector_lines or []
        self.n_sectors    = len(self.sector_lines) + 1

        # State
        self._lap_start_sim : Optional[float] = None
        self._sector_start  : Optional[float] = None
        self._current_sector: int             = 0
        self._sector_splits : List[SectorSplit] = []
        self._last_crossing : float           = -999.0

        # Position history for crossing detection
        self._prev_x : float = 0.0
        self._prev_y : float = 0.0
        self._initialised : bool = False

        # Records
        self.lap_records   : List[LapRecord] = []
        self.best_lap      : Optional[LapRecord] = None
        self.current_lap_n : int = 0

        # Current lap accumulators
        self._max_lat_g  : float = 0.0
        self._max_lon_g  : float = 0.0
        self._max_speed  : float = 0.0
        self._energy_start_kj : float = 0.0

        self._sector_cooldowns : List[float] = [-999.0] * max(1, len(sector_lines or []))

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_track(self, finish_line: TimingLine,
                  sector_lines: Optional[List[TimingLine]] = None):
        self.finish_line  = finish_line
        self.sector_lines = sector_lines or []
        self.n_sectors    = len(self.sector_lines) + 1
        self._sector_cooldowns = [-999.0] * len(self.sector_lines)
        self.reset()

    def reset(self):
        """Reset all timing state."""
        self._lap_start_sim    = None
        self._sector_start     = None
        self._current_sector   = 0
        self._sector_splits    = []
        self._last_crossing    = -999.0
        self._initialised      = False
        self._max_lat_g        = 0.0
        self._max_lon_g        = 0.0
        self._max_speed        = 0.0
        self._sector_cooldowns = [-999.0] * len(self.sector_lines)

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, x: float, y: float, sim_time: float,
               speed_kmh: float = 0.0, lat_g: float = 0.0,
               lon_g: float = 0.0, energy_kj: float = 0.0) -> dict:
        """
        Call once per physics frame.

        Returns a dict with:
            lap_time    : float — current lap time (0 if not running)
            lap_number  : int   — completed laps
            sector      : int   — current sector (0-indexed)
            new_lap     : bool  — True on the frame a new lap starts
            new_sector  : bool  — True on the frame a new sector starts
            best_lap_s  : float — best lap time so far (0 if < 1 lap done)
            delta_best  : float — delta vs best lap at this sector (-=faster)
        """
        result = {
            'lap_time'  : 0.0,
            'lap_number': self.current_lap_n,
            'sector'    : self._current_sector,
            'new_lap'   : False,
            'new_sector': False,
            'best_lap_s': self.best_lap.lap_time if self.best_lap else 0.0,
            'delta_best': 0.0,
        }

        if self.finish_line is None:
            return result

        # Accumulate per-lap stats
        self._max_lat_g  = max(self._max_lat_g,  abs(lat_g))
        self._max_lon_g  = max(self._max_lon_g,  abs(lon_g))
        self._max_speed  = max(self._max_speed,  speed_kmh)

        # ── Check sector crossings ─────────────────────────────────────────
        if self._lap_start_sim is not None:
            for i, sec_line in enumerate(self.sector_lines):
                if (sim_time - self._sector_cooldowns[i]) < self.SECTOR_COOLDOWN:
                    continue
                if self._initialised and sec_line.check_crossing(
                        self._prev_x, self._prev_y, x, y):
                    self._sector_cooldowns[i] = sim_time
                    split_t = sim_time - self._lap_start_sim
                    delta_b = self._sector_delta_vs_best(i, split_t)
                    split   = SectorSplit(
                        sector_idx=i, split_time=split_t,
                        sim_time=sim_time, delta_best=delta_b,
                    )
                    self._sector_splits.append(split)
                    self._current_sector = i + 1
                    result['new_sector'] = True
                    result['delta_best'] = delta_b or 0.0
                    break

        # ── Check finish line crossing ─────────────────────────────────────
        if self._initialised and self.finish_line.check_crossing(
                self._prev_x, self._prev_y, x, y):

            if self._lap_start_sim is not None:
                lap_t = sim_time - self._lap_start_sim

                if lap_t >= self.MIN_LAP_TIME:
                    # Record completed lap
                    sector_times = [s.split_time for s in self._sector_splits]
                    rec = LapRecord(
                        lap_number   = self.current_lap_n,
                        lap_time     = lap_t,
                        sector_times = sector_times,
                        max_lat_g    = self._max_lat_g,
                        max_lon_g    = self._max_lon_g,
                        max_speed    = self._max_speed,
                        energy_kj    = energy_kj - self._energy_start_kj,
                    )
                    self.lap_records.append(rec)
                    if self.best_lap is None or lap_t < self.best_lap.lap_time:
                        self.best_lap = rec
                    self.current_lap_n += 1
                    result['new_lap']   = True
                    result['lap_number']= self.current_lap_n
                    result['best_lap_s']= self.best_lap.lap_time
                    self._print_lap_summary(rec)

            # Reset for new lap
            self._lap_start_sim     = sim_time
            self._sector_start      = sim_time
            self._current_sector    = 0
            self._sector_splits     = []
            self._last_crossing     = sim_time
            self._max_lat_g         = 0.0
            self._max_lon_g         = 0.0
            self._max_speed         = 0.0
            self._energy_start_kj   = energy_kj
            self._sector_cooldowns  = [-999.0] * len(self.sector_lines)

        elif self._lap_start_sim is None and not self._initialised:
            # Arm: start timing on first finish-line crossing
            self._lap_start_sim    = sim_time
            self._sector_start     = sim_time
            self._energy_start_kj  = energy_kj

        # Current lap time
        if self._lap_start_sim is not None:
            result['lap_time'] = sim_time - self._lap_start_sim
        result['sector'] = self._current_sector

        # Update position history
        self._prev_x      = x
        self._prev_y      = y
        self._initialised = True
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sector_delta_vs_best(self, sector_idx: int, current_split: float) -> Optional[float]:
        if self.best_lap is None:
            return None
        best_splits = self.best_lap.sector_times
        if sector_idx < len(best_splits):
            return current_split - best_splits[sector_idx]
        return None

    def _print_lap_summary(self, rec: LapRecord):
        stars = " ★ BEST LAP ★" if (self.best_lap and rec.lap_time == self.best_lap.lap_time) else ""
        print(f"[LapTimer] Lap {rec.lap_number:3d}: {rec.lap_time:6.3f}s{stars}")
        if rec.sector_times:
            splits = " | ".join(f"S{i+1}: {s:.3f}s" for i, s in enumerate(rec.sector_times))
            print(f"           Sectors: {splits}")
        print(f"           Peak: {rec.max_lat_g:.2f}G lat | {rec.max_speed:.1f} km/h | {rec.energy_kj:.1f} kJ")

    def get_display_string(self) -> str:
        """Compact timing string for HUD overlay."""
        if self._lap_start_sim is None:
            return "  --:--.--- | Lap --"
        from time import gmtime, strftime
        def fmt(t): return f"{int(t//60):02d}:{t%60:06.3f}"
        lines = [f"  LAP {self.current_lap_n+1:3d} | {fmt(self._get_current_lap_time())}"]
        if self.best_lap:
            lines.append(f"  BEST:     {fmt(self.best_lap.lap_time)}")
        if self._sector_splits:
            last = self._sector_splits[-1]
            d    = last.delta_best
            if d is not None:
                sign = "+" if d >= 0 else ""
                lines.append(f"  S{last.sector_idx+1} Δ: {sign}{d:.3f}s")
        return "\n".join(lines)

    def _get_current_lap_time(self) -> float:
        # Can't use sim_time directly, so approximate with wall clock
        # In practice, called with sim_time in update()
        return 0.0

    @property
    def timing_active(self) -> bool:
        return self._lap_start_sim is not None
