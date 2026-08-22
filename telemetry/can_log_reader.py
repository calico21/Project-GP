# telemetry/can_log_reader.py
# Project-GP — Real CAN Log Reader
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reads the real CAN CSV files from data/raw_can_logs/ and produces a clean
# pd.DataFrame on a uniform 10 ms time grid with canonical channel names.
#
# Schema mapping (CSV column → canonical name → physical unit):
#   Time       → time        → s (Unix timestamp)
#   Latitude   → lat         → deg
#   Longitude  → lon         → deg
#   speed      → v_kph       → km/h  → then v_ms (m/s)
#   ANGLE      → delta_sw_deg → deg  → then delta_rad (rad, at wheel)
#   Yaw_Rate_z → yaw_rate    → °/s  → then rad/s
#   a_x        → ax_raw      → m/s²
#   a_y        → ay_raw      → m/s²
#   APPS_AV    → throttle_pct → 0-100 → then throttle (0-1)
#   BPPS       → brake_pct   → 0-100 → then brake (0-1)
#   rlRPM      → rpm_rl      → RPM
#   rrRPM      → rpm_rr      → RPM
#   rlTRQ      → trq_rl      → N·m
#   rrTRQ      → trq_rr      → N·m
#
# GPS→Cartesian: WGS84 flat-Earth projection anchored at the first valid GPS fix.
# IMU lag correction: 65 ms (FIFO latency from CAN bus), applied to ax, ay, yaw_rate.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate

# ─────────────────────────────────────────────────────────────────────────────
# Column mapping: (csv_name → canonical_name)
# The reader tries each alias in order; first match wins.
# ─────────────────────────────────────────────────────────────────────────────

_COLUMN_MAP: dict[str, list[str]] = {
    # Canonical      Possible CSV names (case-insensitive)
    "time":          ["Time", "time", "Timestamp", "timestamp", "t"],
    "lat":           ["Latitude", "latitude", "lat", "GPS_Lat"],
    "lon":           ["Longitude", "longitude", "lon", "GPS_Lon"],
    "v_kph":         ["speed", "Speed", "v_kph", "velocity_kmh", "GPS_Speed"],
    "delta_sw_deg":  ["ANGLE", "angle", "steer_angle", "SteeringAngle", "steering_deg"],
    "yaw_rate_dps":  ["Yaw_Rate_z", "yaw_rate_z", "YawRate", "yaw_rate", "wz_dps"],
    "ax_raw":        ["a_x", "accel_x", "ax", "AccelX", "a_x_g"],
    "ay_raw":        ["a_y", "accel_y", "ay", "AccelY", "a_y_g"],
    "throttle_pct":  ["APPS_AV", "apps_av", "throttle_pct", "APPS", "throttle"],
    "brake_pct":     ["BPPS", "bpps", "brake_pct", "brake", "BrakePressure"],
    "rpm_rl":        ["rlRPM", "rl_rpm", "rpm_rl", "RPM_RL"],
    "rpm_rr":        ["rrRPM", "rr_rpm", "rpm_rr", "RPM_RR"],
    "trq_rl":        ["rlTRQ", "rl_trq", "trq_rl", "TRQ_RL"],
    "trq_rr":        ["rrTRQ", "rr_trq", "trq_rr", "TRQ_RR"],
}

# Channels that are REQUIRED for validation.  Missing optional channels get NaN.
_REQUIRED_CHANNELS = {"time", "lat", "lon", "v_kph", "delta_sw_deg"}

# Steering rack ratio (steering wheel deg → front wheel deg)
# Typical FS value: ~3.5:1 (adjust per car geometry in config)
_STEER_RACK_RATIO: float = 3.5

# IMU CAN FIFO latency to correct [s]
_IMU_LAG_S: float = 0.065   # 65 ms (2 CAN frames @ 20 ms + 25 ms processing)

# Resample target [s]
_DT_TARGET: float = 0.010   # 10 ms

# Constant for GPS→Cartesian projection
_DEG_TO_M_LAT: float = 111_320.0          # metres per degree of latitude
# metres per degree of longitude (latitude-dependent, computed at anchor)


# ─────────────────────────────────────────────────────────────────────────────
# §1  CANLogReader
# ─────────────────────────────────────────────────────────────────────────────

class CANLogReader:
    """
    Reads one real CAN CSV session file and returns a clean DataFrame.

    Usage
    -----
    >>> reader = CANLogReader('data/raw_can_logs/2.csv')
    >>> df = reader.load()       # returns uniform-grid DataFrame
    >>> x_cart, y_cart = df['x_m'].values, df['y_m'].values
    """

    def __init__(
        self,
        csv_path: str | Path,
        steer_rack_ratio: float = _STEER_RACK_RATIO,
        imu_lag_s: float = _IMU_LAG_S,
        dt_target: float = _DT_TARGET,
        ax_unit: str = "g",    # 'g' if a_x is in g, 'm/s2' otherwise
        ay_unit: str = "g",
    ):
        self.path            = Path(csv_path)
        self.steer_ratio     = steer_rack_ratio
        self.imu_lag_s       = imu_lag_s
        self.dt_target       = dt_target
        self.ax_unit         = ax_unit
        self.ay_unit         = ay_unit
        self._df_raw: Optional[pd.DataFrame] = None
        self._df_out: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Full pipeline: parse → map → resample → convert → add Cartesian."""
        raw        = self._parse_csv()
        mapped     = self._map_columns(raw)
        resampled  = self._resample(mapped)
        converted  = self._convert_units(resampled)
        with_cart  = self._add_cartesian(converted)
        self._df_out = with_cart
        print(f"[CANLogReader] {self.path.name}: {len(with_cart)} rows "
              f"@ {self.dt_target*1000:.0f} ms grid | "
              f"channels: {sorted(with_cart.columns.tolist())}")
        return with_cart

    # ─────────────────────────────────────────────────────────────────────────
    # §1.1  CSV parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_csv(self) -> pd.DataFrame:
        """Read raw CSV, tolerating trailing commas, mixed dtypes, NaN rows."""
        df = pd.read_csv(
            self.path,
            on_bad_lines="skip",
            engine="python",   # C engine crashes on buffer overflow in real CAN CSVs
            dtype=str,         # read all as str; coerce_to_numeric done per-column below
        )
        # Drop fully-empty rows
        df = df.dropna(how="all")
        self._df_raw = df
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # §1.2  Column mapping
    # ─────────────────────────────────────────────────────────────────────────

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map CSV column names → canonical names. Warns on missing channels."""
        available = {c.strip(): c for c in df.columns}
        available_lower = {c.lower(): orig for c, orig in available.items()}

        out: dict[str, pd.Series] = {}
        missing_required = []

        for canonical, aliases in _COLUMN_MAP.items():
            found = False
            for alias in aliases:
                # exact match first, then case-insensitive
                if alias in available:
                    out[canonical] = df[available[alias]]
                    found = True
                    break
                if alias.lower() in available_lower:
                    out[canonical] = df[available_lower[alias.lower()]]
                    found = True
                    break
            if not found:
                if canonical in _REQUIRED_CHANNELS:
                    missing_required.append(canonical)
                else:
                    out[canonical] = pd.Series(np.nan, index=df.index)
                    warnings.warn(
                        f"[CANLogReader] Optional channel '{canonical}' not found "
                        f"in {self.path.name}; filled with NaN.", stacklevel=2)

        if missing_required:
            raise ValueError(
                f"[CANLogReader] Required channels missing in {self.path.name}: "
                f"{missing_required}\nAvailable columns: {list(available.keys())[:20]}")

        mapped = pd.DataFrame(out)

        # Coerce all to numeric (non-numeric → NaN)
        for col in mapped.columns:
            mapped[col] = pd.to_numeric(mapped[col], errors="coerce")

        # Sort by time
        mapped = mapped.sort_values("time").reset_index(drop=True)

        # Drop rows where time itself is NaN
        mapped = mapped.dropna(subset=["time"]).reset_index(drop=True)

        return mapped

    # ─────────────────────────────────────────────────────────────────────────
    # §1.3  Resample to uniform grid
    # ─────────────────────────────────────────────────────────────────────────

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate all channels onto a uniform dt_target grid."""
        t_raw = df["time"].values.astype(float)
        t0, t1 = t_raw[0], t_raw[-1]
        t_grid = np.arange(t0, t1, self.dt_target)

        out: dict[str, np.ndarray] = {"time": t_grid}
        for col in df.columns:
            if col == "time":
                continue
            y = df[col].values.astype(float)
            # Fill forward for NaN before interpolation to avoid extrapolation gaps
            valid = np.isfinite(y)
            if valid.sum() < 4:
                out[col] = np.full(len(t_grid), np.nan)
                continue
            # 1-D linear interpolation (fast, no oscillation near GPS dropouts)
            interp_fn = interpolate.interp1d(
                t_raw[valid], y[valid],
                kind="linear",
                bounds_error=False,
                fill_value=(y[valid][0], y[valid][-1]),
            )
            out[col] = interp_fn(t_grid)

        return pd.DataFrame(out)

    # ─────────────────────────────────────────────────────────────────────────
    # §1.4  Unit conversion + IMU lag correction
    # ─────────────────────────────────────────────────────────────────────────

    def _convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Speed: km/h → m/s
        df["v_ms"] = df["v_kph"] / 3.6

        # Steering: wheel angle → front tyre steer angle [rad]
        df["delta_rad"] = np.deg2rad(df["delta_sw_deg"] / self.steer_ratio)

        # Yaw rate: °/s → rad/s
        df["yaw_rate"] = np.deg2rad(df["yaw_rate_dps"])

        # Accelerations: g → m/s²  (or pass through if already m/s²)
        g = 9.81
        df["ax"] = df["ax_raw"] * (g if self.ax_unit == "g" else 1.0)
        df["ay"] = df["ay_raw"] * (g if self.ay_unit == "g" else 1.0)

        # Throttle / brake: % → fraction
        df["throttle"] = df["throttle_pct"] / 100.0
        df["brake"]    = df["brake_pct"]    / 100.0

        # ── IMU lag correction ────────────────────────────────────────────────
        # The IMU data (ax, ay, yaw_rate) arrives 65 ms late on the CAN bus.
        # Correct by time-advancing the IMU channels (shift left by lag/dt steps).
        lag_steps = round(self.imu_lag_s / self.dt_target)
        for ch in ("ax", "ay", "yaw_rate"):
            raw = df[ch].values.copy()
            df[ch] = np.roll(raw, -lag_steps)
            df.loc[df.index[-lag_steps:], ch] = raw[-1]  # hold last value

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # §1.5  GPS → local Cartesian
    # ─────────────────────────────────────────────────────────────────────────

    def _add_cartesian(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project GPS (lat, lon) to local (x_m, y_m) anchored at first fix."""
        lat = df["lat"].values
        lon = df["lon"].values

        # Anchor at first valid fix
        valid_mask = np.isfinite(lat) & np.isfinite(lon)
        if not valid_mask.any():
            warnings.warn("[CANLogReader] No valid GPS fixes found. x_m/y_m set to NaN.")
            df["x_m"] = np.nan
            df["y_m"] = np.nan
            return df

        lat0 = lat[valid_mask][0]
        lon0 = lon[valid_mask][0]

        m_per_deg_lat = _DEG_TO_M_LAT
        m_per_deg_lon = _DEG_TO_M_LAT * math.cos(math.radians(lat0))

        x_m = (lon - lon0) * m_per_deg_lon
        y_m = (lat - lat0) * m_per_deg_lat

        df = df.copy()
        df["x_m"] = x_m
        df["y_m"] = y_m
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # §1.6  Convenience accessors
    # ─────────────────────────────────────────────────────────────────────────

    def get_controls(self) -> dict:
        """
        Return control channels as numpy arrays, ready for the digital twin.
        Must call load() first.
        """
        if self._df_out is None:
            raise RuntimeError("Call .load() before .get_controls()")
        df = self._df_out
        return {
            "steer":    df["delta_rad"].values.astype(np.float32),
            "throttle": df["throttle"].values.astype(np.float32),
            "brake":    df["brake"].values.astype(np.float32),
            "dt":       np.full(len(df), self.dt_target, dtype=np.float32),
        }

    def get_measurements(self) -> dict:
        """
        Return measurement channels as numpy arrays for fidelity scoring.
        Must call load() first.
        """
        if self._df_out is None:
            raise RuntimeError("Call .load() before .get_measurements()")
        df = self._df_out
        return {
            "speed":    df["v_ms"].values.astype(np.float32),
            "yaw_rate": df["yaw_rate"].values.astype(np.float32),
            "ay":       df["ay"].values.astype(np.float32),
            "ax":       df["ax"].values.astype(np.float32),
            "x_m":      df["x_m"].values.astype(np.float32),
            "y_m":      df["y_m"].values.astype(np.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# §2  CLI self-test
# ─────────────────────────────────────────────────────────────────────────────

def _cli_selftest(file: str):
    reader = CANLogReader(file)
    df = reader.load()

    print(f"\n── Shape: {df.shape}")
    print(f"── Duration: {df['time'].iloc[-1] - df['time'].iloc[0]:.1f} s")
    print(f"── Speed range: {df['v_ms'].min():.1f} – {df['v_ms'].max():.1f} m/s")
    print(f"── Yaw rate range: {df['yaw_rate'].min():.3f} – {df['yaw_rate'].max():.3f} rad/s")
    print(f"── Steer range: {df['delta_rad'].min():.3f} – {df['delta_rad'].max():.3f} rad")
    print(f"── GPS: lat0={df['lat'].dropna().iloc[0]:.6f}, lon0={df['lon'].dropna().iloc[0]:.6f}")
    print(f"── Cartesian extent: x=[{df['x_m'].min():.1f}, {df['x_m'].max():.1f}] m  "
          f"y=[{df['y_m'].min():.1f}, {df['y_m'].max():.1f}] m")
    print(f"── NaN pct per channel:")
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            print(f"     {col}: {pct:.1f}%")

    controls = reader.get_controls()
    meas     = reader.get_measurements()
    print(f"\n── Controls keys: {list(controls.keys())}")
    print(f"── Measurement keys: {list(meas.keys())}")
    print("\n[CANLogReader] Self-test PASSED ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CANLogReader self-test")
    parser.add_argument("--file", "-f", required=True,
                        help="Path to CAN CSV log (e.g. data/raw_can_logs/2.csv)")
    args = parser.parse_args()
    _cli_selftest(args.file)
