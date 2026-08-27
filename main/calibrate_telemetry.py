#!/usr/bin/env python3
# main/calibrate_telemetry.py — Gradient-Based Parameter Calibration
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.calibrate_mu_from_telemetry import main as calibrate_main


def main():
    parser = argparse.ArgumentParser(description="Calibración por gradiente contra telemetría CAN")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_can_logs"))
    parser.add_argument("--dbc", type=Path, default=Path("TER.dbc"))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--steer-sign", type=float, default=1.0)
    args = parser.parse_args()

    sys.argv = [
        sys.argv[0],
        "--data-dir", str(args.data_dir),
        "--dbc", str(args.dbc),
        "--dt", str(args.dt),
        "--steps", str(args.steps),
        "--lr", str(args.lr),
        "--steer-sign", str(args.steer_sign),
    ]
    calibrate_main()


if __name__ == "__main__":
    main()