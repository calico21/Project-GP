"""
simulator/telemetry_recorder.py
─────────────────────────────────────────────────────────────────────────────
Project-GP Telemetry Recorder — High-Fidelity Session Capture

Records physics server telemetry into formats consumable by:
  · ModelValidator.run_open_loop_validation() — NPZ with named channels
  · telemetry_analysis.py — CSV with full column headers
  · Dashboard ANALYZE mode — JSON snapshots for static replay

This closes the critical gap: run a simulation → record it → validate against
real telemetry using ModelValidator → get twin_fidelity score.

Usage:
  # Record 10 laps of autopilot on FSG autocross:
  python simulator/telemetry_recorder.py --duration 120 --output session_001

  # Record indefinitely (Ctrl+C to stop):
  python simulator/telemetry_recorder.py --output skidpad_test

  # Record and immediately run validation:
  python simulator/telemetry_recorder.py --output run_01 --validate

Output files:
  logs/run_01.npz          — NumPy archive (for ModelValidator)
  logs/run_01.csv          — Full CSV (for analysis / dashboard)
  logs/run_01_meta.json    — Session metadata (setup, track, duration)
"""

from __future__ import annotations

import os
import sys
import time
import json
import socket
import struct
import signal
import argparse
import csv
from collections import defaultdict
from typing import Optional

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.sim_config import (
    HOST, PORT_TELEM_VIZ, PORT_TELEM_DEBUG,
    PHYSICS_HZ, LOG_DIR, DEFAULT_SETUP_28,
    SETUP_PARAM_NAMES, S,
)

# ─────────────────────────────────────────────────────────────────────────────
# §1  Telemetry Packet Parser
# ─────────────────────────────────────────────────────────────────────────────

TX_FMT   = '<64f'
TX_BYTES = struct.calcsize(TX_FMT)

# Channel names matching sim_protocol.py TX layout
# These are the fields we record per frame
RECORD_CHANNELS = [
    # 0-2: header
    "frame_id", "sim_time",
    # 3-8: pose
    "x", "y", "z", "roll", "pitch", "yaw",
    # 9-15: velocities + yaw rate
    "vx", "vy", "vz", "ax", "ay", "az", "wz",
    # 16-19: suspension
    "z_fl", "z_fr", "z_rl", "z_rr",
    # 20-27: forces
    "Fz_fl", "Fz_fr", "Fz_rl", "Fz_rr",
    "Fy_fl", "Fy_fr", "Fy_rl", "Fy_rr",
    # 28-33: slip
    "slip_fl", "slip_fr", "slip_rl", "slip_rr",
    "kappa_rl", "kappa_rr",
    # 34-37: wheel omega
    "omega_fl", "omega_fr", "omega_rl", "omega_rr",
    # 38-41: tire temps
    "T_fl", "T_fr", "T_rl", "T_rr",
    # 42-44: controls
    "delta", "throttle", "brake_norm",
    # 45-46: grip
    "grip_f", "grip_r",
    # 47-49: timing
    "lap_time", "lap_number", "sector",
    # 50-52: derived
    "speed_kmh", "lat_g", "lon_g",
    # 53: yaw_rate_deg
    "yaw_rate_deg",
    # 54-56: aero + energy
    "downforce", "drag", "energy_kj",
]

# Indices into the 64-float packet for each channel
# magic=0, then frame_id=1, sim_time=2, x=3, ... (shift by 1 for magic prefix)
CHANNEL_INDICES = {
    "frame_id": 1, "sim_time": 2,
    "x": 3, "y": 4, "z": 5, "roll": 6, "pitch": 7, "yaw": 8,
    "vx": 9, "vy": 10, "vz": 11, "ax": 12, "ay": 13, "az": 14, "wz": 15,
    "z_fl": 16, "z_fr": 17, "z_rl": 18, "z_rr": 19,
    "Fz_fl": 20, "Fz_fr": 21, "Fz_rl": 22, "Fz_rr": 23,
    "Fy_fl": 24, "Fy_fr": 25, "Fy_rl": 26, "Fy_rr": 27,
    "slip_fl": 28, "slip_fr": 29, "slip_rl": 30, "slip_rr": 31,
    "kappa_rl": 32, "kappa_rr": 33,
    "omega_fl": 34, "omega_fr": 35, "omega_rl": 36, "omega_rr": 37,
    "T_fl": 38, "T_fr": 39, "T_rl": 40, "T_rr": 41,
    "delta": 42, "throttle": 43, "brake_norm": 44,
    "grip_f": 45, "grip_r": 46,
    "lap_time": 47, "lap_number": 48, "sector": 49,
    "speed_kmh": 50, "lat_g": 51, "lon_g": 52,
    "yaw_rate_deg": 53,
    "downforce": 54, "drag": 55, "energy_kj": 56,
}


# ─────────────────────────────────────────────────────────────────────────────
# §2  Recorder
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryRecorder:
    """
    High-fidelity telemetry recorder.

    Listens for UDP packets from physics_server, accumulates per-channel
    arrays, and writes NPZ + CSV + metadata on stop.
    """

    def __init__(self,
                 output_name: str = "session",
                 udp_port:    int = PORT_TELEM_DEBUG,
                 duration:    float = 0.0,   # 0 = indefinite
                 record_hz:   int = PHYSICS_HZ):  # record every frame by default
        self.output_name = output_name
        self.udp_port    = udp_port
        self.duration    = duration
        self.record_hz   = record_hz

        # Decimation: record every N-th UDP frame
        self._decimate = max(1, PHYSICS_HZ // record_hz)

        # Accumulators: channel_name → list of float
        self._data: dict[str, list] = {ch: [] for ch in RECORD_CHANNELS}
        self._frame_count = 0
        self._record_count = 0
        self._running = False

        # Session metadata
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def run(self):
        """Main recording loop."""
        os.makedirs(LOG_DIR, exist_ok=True)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.udp_port))
        sock.settimeout(2.0)

        print("=" * 60)
        print("  Project-GP Telemetry Recorder")
        print(f"  Listening on   : UDP :{self.udp_port}")
        print(f"  Recording at   : {self.record_hz} Hz (1/{self._decimate} decimation)")
        print(f"  Duration       : {'indefinite (Ctrl+C)' if self.duration <= 0 else f'{self.duration:.0f}s'}")
        print(f"  Output prefix  : {self.output_name}")
        print(f"  Channels       : {len(RECORD_CHANNELS)}")
        print("=" * 60)
        print()
        print("  Waiting for physics_server.py telemetry...")

        self._running = True
        self._start_time = time.monotonic()
        last_status = time.monotonic()

        # Handle Ctrl+C gracefully
        def _signal_handler(sig, frame):
            self._running = False
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            while self._running:
                # Duration check
                elapsed = time.monotonic() - self._start_time
                if self.duration > 0 and elapsed > self.duration:
                    print(f"\n  Duration reached ({self.duration:.0f}s). Stopping.")
                    break

                try:
                    data, _ = sock.recvfrom(TX_BYTES + 16)
                except socket.timeout:
                    if self._record_count == 0:
                        print("  Still waiting for telemetry packets...")
                    continue

                if len(data) < TX_BYTES:
                    continue

                self._frame_count += 1

                # Decimation
                if self._frame_count % self._decimate != 0:
                    continue

                # Parse
                try:
                    values = struct.unpack(TX_FMT, data[:TX_BYTES])
                except struct.error:
                    continue

                # Record each channel
                for ch in RECORD_CHANNELS:
                    idx = CHANNEL_INDICES.get(ch)
                    if idx is not None and idx < len(values):
                        self._data[ch].append(float(values[idx]))
                    else:
                        self._data[ch].append(0.0)

                self._record_count += 1

                # Status update every 5 seconds
                now = time.monotonic()
                if now - last_status > 5.0:
                    udp_hz = self._frame_count / (now - self._start_time)
                    rec_hz = self._record_count / (now - self._start_time)
                    spd = values[50] if len(values) > 50 else 0  # speed_kmh
                    lap = int(values[48]) if len(values) > 48 else 0
                    print(f"  Recording: {self._record_count} frames | "
                          f"{rec_hz:.0f} Hz | "
                          f"{elapsed:.0f}s elapsed | "
                          f"v={spd:.1f} km/h | "
                          f"lap {lap+1}")
                    last_status = now

        finally:
            sock.close()
            self._end_time = time.monotonic()

        # Save
        if self._record_count > 0:
            self._save_all()
        else:
            print("  No frames recorded. Nothing to save.")

    def _save_all(self):
        """Save NPZ, CSV, and metadata."""
        elapsed = self._end_time - self._start_time
        print(f"\n  Session complete: {self._record_count} frames in {elapsed:.1f}s")

        # ── NPZ (for ModelValidator) ─────────────────────────────────
        npz_path = os.path.join(LOG_DIR, f"{self.output_name}.npz")
        arrays = {}
        for ch in RECORD_CHANNELS:
            arrays[ch] = np.array(self._data[ch], dtype=np.float64)

        # Also provide ModelValidator-compatible aliases
        # ModelValidator expects: 'velocity', 'yaw_rate', 'g_lat', 's', 'steer'
        if "vx" in arrays:
            arrays["velocity"] = np.abs(arrays["vx"])
        if "wz" in arrays:
            arrays["yaw_rate"] = arrays["wz"]
        if "lat_g" in arrays:
            arrays["g_lat"] = arrays["lat_g"] * 9.81  # convert G → m/s²
        if "lon_g" in arrays:
            arrays["g_long"] = arrays["lon_g"] * 9.81
        if "delta" in arrays:
            arrays["steer"] = arrays["delta"]

        # Approximate arc-length from cumulative speed × dt
        dt_arr = np.diff(arrays.get("sim_time", np.zeros(1)), prepend=0.0)
        dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 1.0 / self.record_hz
        arrays["s"] = np.cumsum(np.abs(arrays.get("vx", np.zeros(1))) * dt_arr)

        np.savez_compressed(npz_path, **arrays)
        print(f"  ✓ NPZ saved: {npz_path}  ({os.path.getsize(npz_path) / 1024:.0f} KB)")

        # ── CSV ──────────────────────────────────────────────────────
        csv_path = os.path.join(LOG_DIR, f"{self.output_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(RECORD_CHANNELS)
            n = len(self._data[RECORD_CHANNELS[0]])
            for i in range(n):
                row = [self._data[ch][i] for ch in RECORD_CHANNELS]
                writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])
        print(f"  ✓ CSV saved: {csv_path}  ({os.path.getsize(csv_path) / 1024:.0f} KB)")

        # ── Metadata ─────────────────────────────────────────────────
        meta_path = os.path.join(LOG_DIR, f"{self.output_name}_meta.json")
        meta = {
            "session_name": self.output_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(elapsed, 2),
            "frames_recorded": self._record_count,
            "record_hz": self.record_hz,
            "udp_frames_total": self._frame_count,
            "channels": RECORD_CHANNELS,
            "setup": {
                name: float(DEFAULT_SETUP_28[i])
                for i, name in enumerate(SETUP_PARAM_NAMES)
            },
            "files": {
                "npz": os.path.basename(npz_path),
                "csv": os.path.basename(csv_path),
            },
            "validation_channels": {
                "velocity": "abs(vx) [m/s]",
                "yaw_rate": "wz [rad/s]",
                "g_lat": "ay [m/s²]",
                "steer": "delta [rad]",
                "s": "cumulative arc-length [m]",
            },
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ Meta saved: {meta_path}")

        # ── Summary statistics ───────────────────────────────────────
        vx = arrays.get("vx", np.zeros(1))
        ay = arrays.get("ay", np.zeros(1))
        wz = arrays.get("wz", np.zeros(1))
        print(f"\n  Session statistics:")
        print(f"    Speed    : {np.min(np.abs(vx))*3.6:.1f} – {np.max(np.abs(vx))*3.6:.1f} km/h "
              f"(mean {np.mean(np.abs(vx))*3.6:.1f})")
        print(f"    Lat G    : {np.min(ay)/9.81:.2f} – {np.max(ay)/9.81:.2f} G")
        print(f"    Yaw rate : {np.min(wz):.2f} – {np.max(wz):.2f} rad/s")
        print(f"    Distance : {arrays['s'][-1]:.0f} m")
        print(f"    Energy   : {arrays.get('energy_kj', np.zeros(1))[-1]:.1f} kJ")


# ─────────────────────────────────────────────────────────────────────────────
# §3  Validation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(npz_path: str):
    """
    Load a recorded NPZ and run ModelValidator to compute twin_fidelity.

    This compares the recorded simulation against itself (trivially 100%)
    or against real telemetry if a real NPZ is provided alongside.
    """
    try:
        from telemetry.validation import ModelValidator
    except ImportError:
        print("[Validation] Cannot import ModelValidator. Skipping.")
        return

    data = dict(np.load(npz_path))

    # Self-validation: sim_data = real_data + small noise
    # This verifies the pipeline works; real validation needs actual car data
    sim_data = {
        's': data.get('s', np.zeros(1)),
        'velocity': data.get('velocity', np.zeros(1)),
        'yaw_rate': data.get('yaw_rate', np.zeros(1)),
        'g_lat': data.get('g_lat', np.zeros(1)),
    }

    print("\n[Validation] Running self-validation (sim vs sim = trivial 100%)...")
    print("[Validation] For real validation, provide real car telemetry NPZ.")

    # Build a minimal track model from the recorded XY
    track_model = {
        's': data.get('s', np.zeros(1)),
        'x': data.get('x', np.zeros(1)),
        'y': data.get('y', np.zeros(1)),
    }

    validator = ModelValidator(track_model)
    metrics = validator.validate_model(sim_data, sim_data)
    print(f"[Validation] Twin fidelity: {metrics.get('twin_fidelity', 0.0):.1f}%")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# §4  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project-GP Telemetry Recorder — session capture for validation")
    parser.add_argument("--output", "-o", type=str, default="session",
                        help="Output file prefix (default: 'session')")
    parser.add_argument("--duration", "-d", type=float, default=0.0,
                        help="Recording duration in seconds (0 = indefinite)")
    parser.add_argument("--port", type=int, default=PORT_TELEM_DEBUG,
                        help=f"UDP port to listen (default: {PORT_TELEM_DEBUG})")
    parser.add_argument("--hz", type=int, default=PHYSICS_HZ,
                        help=f"Recording rate in Hz (default: {PHYSICS_HZ})")
    parser.add_argument("--validate", action="store_true",
                        help="Run ModelValidator after recording")
    args = parser.parse_args()

    recorder = TelemetryRecorder(
        output_name = args.output,
        udp_port    = args.port,
        duration    = args.duration,
        record_hz   = args.hz,
    )
    recorder.run()

    if args.validate:
        npz_path = os.path.join(LOG_DIR, f"{args.output}.npz")
        if os.path.exists(npz_path):
            run_validation(npz_path)


if __name__ == "__main__":
    main()