# scripts/run_ter27_design_exploration.py
# Project-GP  —  Ter27 Suspension Geometry Design Exploration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the MORL optimizer with progressive parameter freeze/unfreeze.
#
# USAGE:
#   # Phase 1: everything free (concept exploration)
#   python scripts/run_ter27_design_exploration.py --phase 1
#
#   # Phase 2: hardpoints locked, alignment still free
#   python scripts/run_ter27_design_exploration.py --phase 2
#
#   # Phase 3: only track-tuning params free
#   python scripts/run_ter27_design_exploration.py --phase 3
#
#   # Custom freeze: lock specific params from a previous run
#   python scripts/run_ter27_design_exploration.py \
#       --freeze castor_f=5.5 anti_dive_f=0.35 anti_squat=0.30
#
#   # Quick screening (BO only, ~5 min)
#   python scripts/run_ter27_design_exploration.py --phase 1 --bo-only
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import jax_config  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from data.configs.car_config import (
    get_car_config,
    get_design_bounds,
    format_optimizer_output,
)
from data.configs.design_freeze import DesignFreeze, install_freeze
from models.vehicle_dynamics import (
    SuspensionSetup,
    SETUP_NAMES,
    SETUP_DIM,
    make_setup_from_params,
)


def _install_vehicle_params(car_id: str):
    """Swap vehicle_params module dict to selected car."""
    import data.configs.vehicle_params as vp_module
    cfg = get_car_config(car_id)
    vp_module.vehicle_params.clear()
    vp_module.vehicle_params.update(cfg['vehicle_params'])
    print(f"[Config] Vehicle params → {car_id.upper()} "
          f"({cfg['vehicle_params']['total_mass']} kg, "
          f"{cfg['vehicle_params'].get('drivetrain_mode', 'rwd')})")


def parse_freeze_args(freeze_strs: list[str]) -> dict[str, float]:
    """Parse 'param=value' strings into a freeze dict."""
    frozen = {}
    for s in freeze_strs:
        if '=' not in s:
            raise ValueError(f"Expected 'param=value', got '{s}'")
        name, val = s.split('=', 1)
        name = name.strip()
        if name not in SETUP_NAMES:
            raise ValueError(f"Unknown param '{name}'. Valid: {SETUP_NAMES}")
        frozen[name] = float(val)
    return frozen


def main():
    parser = argparse.ArgumentParser(
        description='Ter27 Suspension Geometry Design Exploration'
    )
    parser.add_argument('--car', default='ter27', choices=['ter26', 'ter27'])
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                        help='Design phase: 1=all free, 2=hardpoints locked, '
                             '3=track tuning only')
    parser.add_argument('--freeze', nargs='*', default=[],
                        help='Custom freeze: param=value pairs '
                             '(e.g., --freeze castor_f=5.5 anti_dive_f=0.35)')
    parser.add_argument('--bo-only', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='50 gradient steps instead of full run')
    parser.add_argument('--ensemble', type=int, default=20)
    parser.add_argument('--output-dir', default='data/design_exploration')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build freeze config ───────────────────────────────────────────────
    if args.freeze:
        # Custom freeze from CLI
        frozen_dict = parse_freeze_args(args.freeze)
        freeze = DesignFreeze(frozen_dict)
        phase_label = "custom"
    else:
        # Preset phases
        if args.phase == 1:
            freeze = DesignFreeze.all_free()
        elif args.phase == 2:
            freeze = DesignFreeze.for_ter27_phase2()
        elif args.phase == 3:
            freeze = DesignFreeze.for_ter27_phase3()
        phase_label = f"phase{args.phase}"

    print("=" * 62)
    print(f"  PROJECT-GP  ·  DESIGN EXPLORATION  ·  "
          f"{args.car.upper()}  ·  {phase_label.upper()}")
    print("=" * 62)

    # ── Install car params + freeze ───────────────────────────────────────
    _install_vehicle_params(args.car)

    # Set default setup from car config
    import models.vehicle_dynamics as vd
    cfg = get_car_config(args.car)
    vd.DEFAULT_SETUP = make_setup_from_params(cfg['vehicle_params']).to_vector()

    # Install freeze (modifies SETUP_LB / SETUP_UB in vehicle_dynamics)
    grad_mask = install_freeze(freeze, args.car)

    # ── Import optimizer AFTER bounds are installed ───────────────────────
    from optimization.evolutionary import MORL_SB_TRPO_Optimizer

    # ── Run ───────────────────────────────────────────────────────────────
    optimizer = MORL_SB_TRPO_Optimizer(ensemble_size=args.ensemble)
    t0 = time.time()

    print(f"\n[Phase 1/2] Bayesian Optimization cold-start "
          f"({freeze.n_free}D effective search)...")
    optimizer.run_bo_phase()

    if not args.bo_only:
        n_steps = 50 if args.quick else optimizer.N_GRADIENT_STEPS
        print(f"\n[Phase 2/2] MORL gradient optimization ({n_steps} steps)...")
        optimizer.run_gradient_phase(n_steps=n_steps)

    elapsed = time.time() - t0
    print(f"\n[Done] Total time: {elapsed:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────
    pareto = optimizer.get_pareto_front()
    tag = f"{args.car}_{phase_label}"

    pareto_path = os.path.join(args.output_dir, f'{tag}_pareto_front.csv')
    pareto.to_csv(pareto_path, index=False)
    print(f"\n[Saved] Pareto front → {pareto_path}")

    if len(pareto) > 0:
        best_idx = pareto['grip'].idxmax()
        best_setup = jnp.array(
            [pareto.iloc[best_idx][name] for name in SETUP_NAMES],
            dtype=jnp.float32
        )

        report = format_optimizer_output(best_setup, args.car)
        report_path = os.path.join(args.output_dir, f'{tag}_geometry_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"[Saved] Geometry report → {report_path}")

        # ── Summary table ─────────────────────────────────────────────────
        print(f"\n{'═' * 62}")
        print(f"  GEOMETRY TARGETS  ·  {tag.upper()}")
        print(f"  {freeze.n_frozen} frozen  ·  {freeze.n_free} explored")
        print(f"{'═' * 62}")

        geom_params = [
            'camber_f', 'camber_r', 'toe_f', 'toe_r', 'castor_f',
            'anti_squat', 'anti_dive_f', 'anti_dive_r', 'anti_lift',
            'brake_bias_f', 'bump_steer_f', 'bump_steer_r',
        ]

        print(f"\n  {'Parameter':<18s} {'Status':<8s} {'Min':>8s} "
              f"{'Best':>8s} {'Max':>8s}")
        print(f"  {'─' * 58}")

        for name in geom_params:
            idx = SETUP_NAMES.index(name)
            status = "🔒" if freeze.is_frozen(name) else "🔓"
            best_val = float(best_setup[idx])

            if name in pareto.columns and not freeze.is_frozen(name):
                vals = pareto[name]
                print(f"  {name:<18s} {status:<8s} {vals.min():>+8.3f} "
                      f"{best_val:>+8.3f} {vals.max():>+8.3f}")
            else:
                print(f"  {name:<18s} {status:<8s} {'─':>8s} "
                      f"{best_val:>+8.3f} {'─':>8s}")

        print(f"\n  Best grip: {float(pareto.iloc[best_idx]['grip']):.4f} G")
        print(f"{'═' * 62}")


if __name__ == '__main__':
    main()