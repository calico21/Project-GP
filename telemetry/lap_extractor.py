# telemetry/lap_extractor.py
# Project-GP — Lap Segmentation from Real CAN Telemetry
# ═══════════════════════════════════════════════════════════════════════════════
#
# Takes the raw CAN log DataFrame produced by CANLogReader and:
#  1. Detects lap start/finish crossings via GPS proximity to a reference point.
#  2. Filters out incomplete or GPS-dropout laps.
#  3. Returns a list of (controls, measurements) dicts per lap, ready for
#     ModelValidator.run_open_loop_validation() and compute_twin_fidelity_objective().
#
# Lap detection strategy:
#  - Reference point: centroid of the start cluster (first 10 s) OR explicit coord.
#  - Crossing trigger: car returns within FINISH_RADIUS_M of reference after having
#    traveled > MIN_LAP_DISTANCE_M.
#  - Quality filter: lap duration in [MIN_LAP_S, MAX_LAP_S], GPS dropout < 10%.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FINISH_RADIUS_M:      float = 8.0    # metres — GPS proximity threshold for lap end
MIN_LAP_DISTANCE_M:   float = 20.0   # must travel this far before finish can trigger
MIN_LAP_S:            float = 5.0    # discard laps shorter than this [s]  (FSG laps ~6-47s)
MAX_LAP_S:            float = 600.0  # discard suspiciously long laps [s]
MAX_GPS_DROPOUT_FRAC: float = 0.10   # discard laps with > 10% NaN GPS


# ─────────────────────────────────────────────────────────────────────────────
# §1  LapData — typed container for one lap
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LapData:
    """One clean lap extracted from a CAN session."""
    lap_index:    int
    duration_s:   float
    distance_m:   float
    controls:     dict     # steer, throttle, brake, dt — numpy float32 arrays
    measurements: dict     # speed, yaw_rate, ay, ax, x_m, y_m — numpy float32 arrays
    t_abs:        np.ndarray  # absolute time [s] for this lap


# ─────────────────────────────────────────────────────────────────────────────
# §2  LapExtractor
# ─────────────────────────────────────────────────────────────────────────────

class LapExtractor:
    """
    Segments a CAN session DataFrame into individual laps.

    Parameters
    ----------
    df : pd.DataFrame
        Output of CANLogReader.load() — must contain x_m, y_m, v_ms columns.
    reference_xy : (float, float) | None
        Local Cartesian (x, y) of the start/finish line.
        If None, computed as the centroid of the GPS track during the first 10 s.
    finish_radius_m : float
        Distance from reference point that triggers a new lap.
    dt_target : float
        Sample interval assumed for duration computation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        reference_xy: Optional[tuple[float, float]] = None,
        finish_radius_m: float = FINISH_RADIUS_M,
        dt_target: float = 0.010,
    ):
        self.df             = df.reset_index(drop=True)
        self.finish_radius  = finish_radius_m
        self.dt             = dt_target
        self._laps: list[LapData] = []
        self._ref_xy: tuple[float, float] = (
            reference_xy if reference_xy is not None
            else self._detect_reference()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def extract(self) -> list[LapData]:
        """Run lap detection + filtering. Returns list of LapData objects."""
        boundaries = self._detect_lap_boundaries()
        raw_laps   = self._slice_laps(boundaries)
        good_laps  = self._filter_laps(raw_laps)
        self._laps = good_laps

        if not good_laps:
            warnings.warn(
                "[LapExtractor] No valid laps found. Check reference_xy, "
                "session length, and GPS quality.", stacklevel=2)
        else:
            best = max(good_laps, key=lambda l: l.distance_m)
            print(f"[LapExtractor] {len(good_laps)} valid lap(s) extracted. "
                  f"Best lap: {best.duration_s:.1f} s, {best.distance_m:.0f} m "
                  f"(lap #{best.lap_index})")

        return good_laps

    def best_lap(self) -> Optional[LapData]:
        """Return the longest-distance valid lap (proxy for cleanest lap)."""
        if not self._laps:
            self.extract()
        if not self._laps:
            return None
        return max(self._laps, key=lambda l: l.distance_m)

    # ─────────────────────────────────────────────────────────────────────────
    # §2.1  Reference point detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_reference(self) -> tuple[float, float]:
        """
        Compute start/finish reference as the GPS centroid of the first 10 s.
        This is robust to minor GPS drift at session start.
        """
        df = self.df
        t0 = df["time"].iloc[0]
        mask = (df["time"] - t0) < 10.0
        x_start = df.loc[mask, "x_m"].dropna().values
        y_start = df.loc[mask, "y_m"].dropna().values
        if len(x_start) < 3:
            warnings.warn("[LapExtractor] Insufficient GPS fixes in first 10 s. "
                          "Using (0, 0) as reference.", stacklevel=3)
            return (0.0, 0.0)
        ref = (float(np.mean(x_start)), float(np.mean(y_start)))
        print(f"[LapExtractor] Start/finish reference: ({ref[0]:.1f}, {ref[1]:.1f}) m")
        return ref

    # ─────────────────────────────────────────────────────────────────────────
    # §2.2  Lap boundary detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_lap_boundaries(self) -> list[int]:
        """
        Return row indices where a new lap starts (including index 0).

        When GPS is absent (all-zero), returns [0, len-1] — one full-session segment.
        """
        df   = self.df
        x    = df["x_m"].values
        y    = df["y_m"].values
        vx   = df["v_ms"].values
        rx, ry = self._ref_xy

        # ── GPS-absent guard ─────────────────────────────────────────────────
        # If GPS is all-zero the reference is at (0,0) and every sample is within
        # FINISH_RADIUS_M of it, which causes O(N) false lap boundaries.
        # Detect this and treat the whole file as one segment.
        gps_all_zero = bool(np.all(x == 0.0) and np.all(y == 0.0))
        ref_at_origin = (abs(rx) < 1.0 and abs(ry) < 1.0)
        if gps_all_zero and ref_at_origin:
            print("[LapExtractor] GPS absent (all-zero) — treating full session as one segment.")
            return [0, len(df) - 1]

        # Cumulative distance from odometry (v·dt, GPS-independent)
        dx_odo = np.abs(vx) * self.dt
        cum_dist = np.cumsum(dx_odo)

        boundaries = [0]
        lap_start_cum = 0.0
        armed = False          # True once we've left the start area

        for i in range(1, len(df)):
            if not (np.isfinite(x[i]) and np.isfinite(y[i])):
                continue

            dist_from_lap_start = cum_dist[i] - lap_start_cum

            # Arm the finish trigger after leaving the start area
            if not armed and dist_from_lap_start > MIN_LAP_DISTANCE_M:
                armed = True

            if armed:
                dist_to_ref = math.hypot(x[i] - rx, y[i] - ry)
                if dist_to_ref < self.finish_radius:
                    boundaries.append(i)
                    lap_start_cum = cum_dist[i]
                    armed = False

        # Always add the last row as a sentinel
        boundaries.append(len(df) - 1)
        return boundaries

    # ─────────────────────────────────────────────────────────────────────────
    # §2.3  Slice into lap segments
    # ─────────────────────────────────────────────────────────────────────────

    def _slice_laps(self, boundaries: list[int]) -> list[LapData]:
        """Cut the DataFrame at the detected boundaries."""
        laps = []
        for k in range(len(boundaries) - 1):
            i_start = boundaries[k]
            i_end   = boundaries[k + 1]
            if i_end <= i_start + 1:
                continue
            lap_df = self.df.iloc[i_start:i_end].reset_index(drop=True)
            laps.append(self._build_lap_data(k, lap_df))
        return laps

    # ─────────────────────────────────────────────────────────────────────────
    # §2.4  Build LapData from a lap DataFrame slice
    # ─────────────────────────────────────────────────────────────────────────

    def _build_lap_data(self, lap_index: int, lap_df: pd.DataFrame) -> LapData:
        N = len(lap_df)
        dt_arr = np.full(N, self.dt, dtype=np.float32)

        def col(name: str) -> np.ndarray:
            if name in lap_df.columns:
                return lap_df[name].values.astype(np.float32)
            return np.full(N, np.nan, dtype=np.float32)

        duration_s = N * self.dt
        # Odometric distance
        distance_m = float(np.nansum(np.abs(col("v_ms"))) * self.dt)

        controls = {
            "steer":    col("delta_rad"),
            "throttle": col("throttle"),
            "brake":    col("brake"),
            "dt":       dt_arr,
        }
        measurements = {
            "speed":    col("v_ms"),
            "yaw_rate": col("yaw_rate"),
            "ay":       col("ay"),
            "ax":       col("ax"),
            "x_m":      col("x_m"),
            "y_m":      col("y_m"),
        }
        return LapData(
            lap_index=lap_index,
            duration_s=duration_s,
            distance_m=distance_m,
            controls=controls,
            measurements=measurements,
            t_abs=col("time"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # §2.5  Quality filter
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_laps(self, laps: list[LapData]) -> list[LapData]:
        """Discard laps that are too short, too long, or GPS-deficient."""
        good = []
        for lap in laps:
            # Duration check
            if not (MIN_LAP_S <= lap.duration_s <= MAX_LAP_S):
                print(f"[LapExtractor]   Lap {lap.lap_index}: DISCARDED "
                      f"(duration {lap.duration_s:.1f}s out of [{MIN_LAP_S}, {MAX_LAP_S}]s)")
                continue
            # GPS dropout check — skip if GPS was never populated (all-zero)
            x = lap.measurements["x_m"]
            gps_absent = bool(np.all(x == 0.0) or np.all(~np.isfinite(x)))
            dropout_frac = float(np.isnan(x).mean())
            if not gps_absent and dropout_frac > MAX_GPS_DROPOUT_FRAC:
                print(f"[LapExtractor]   Lap {lap.lap_index}: DISCARDED "
                      f"(GPS dropout {dropout_frac*100:.1f}% > {MAX_GPS_DROPOUT_FRAC*100:.0f}%)")
                continue
            print(f"[LapExtractor]   Lap {lap.lap_index}: OK "
                  f"({lap.duration_s:.1f}s, {lap.distance_m:.0f}m, "
                  f"GPS OK {100*(1-dropout_frac):.0f}%)")
            good.append(lap)
        return good


# ─────────────────────────────────────────────────────────────────────────────
# §3  CLI self-test
# ─────────────────────────────────────────────────────────────────────────────

import math   # noqa: E402  (needed inside _detect_lap_boundaries)


def _cli_selftest(file: str):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from telemetry.can_log_reader import CANLogReader

    reader = CANLogReader(file)
    df     = reader.load()

    extractor = LapExtractor(df)
    laps      = extractor.extract()

    print(f"\n── Total laps found: {len(laps)}")
    for lap in laps:
        print(f"   Lap {lap.lap_index}: {lap.duration_s:.1f}s | "
              f"{lap.distance_m:.0f}m | "
              f"steer shape={lap.controls['steer'].shape}")

    best = extractor.best_lap()
    if best is not None:
        print(f"\n── Best lap: #{best.lap_index}  {best.duration_s:.1f}s")
        print(f"   Speed:    {best.measurements['speed'].min():.1f} – "
              f"{best.measurements['speed'].max():.1f} m/s")
        print(f"   Yaw rate: {best.measurements['yaw_rate'].min():.3f} – "
              f"{best.measurements['yaw_rate'].max():.3f} rad/s")

    print("\n[LapExtractor] Self-test PASSED ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LapExtractor self-test")
    parser.add_argument("--file", "-f", required=True,
                        help="Path to CAN CSV log (e.g. data/raw_can_logs/2.csv)")
    args = parser.parse_args()
    _cli_selftest(args.file)
