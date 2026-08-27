#!/usr/bin/env python3
# main/run_backtest.py — CAN Telemetry Replay & Correlation Engine
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.run_can_backtest import main as backtest_main


def main():
    parser = argparse.ArgumentParser(description="Replay y evaluación de correlación de flota")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_can_logs"))
    parser.add_argument("--dbc", type=Path, default=Path("TER.dbc"))
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--lag-samples", type=int, default=14)
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()

    argv_pass = [
        sys.argv[0],
        "--data-dir", str(args.data_dir),
        "--dbc", str(args.dbc),
        "--dt", str(args.dt),
        "--lag-samples", str(args.lag_samples),
    ]
    if args.no_calibration:
        argv_pass.append("--no-calibration")

    sys.argv = argv_pass
    backtest_main()


if __name__ == "__main__":
    main()