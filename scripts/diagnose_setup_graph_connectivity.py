#!/usr/bin/env python3
# scripts/diagnose_setup_graph_connectivity.py
# Project-GP — Setup Parameter Graph Connectivity Diagnostic
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM:
#   `jax.grad(simulate_full_lap)(setup)` returns exact-zero gradient for 20 of
#   the 28 parameters. This means those parameters are not connected to the
#   computational graph that produces `effective_lap_time`.
#
# This script diagnoses the dead-end by running gradients against a series
# of intermediate physics outputs, so we can localise where the routing breaks:
#
#   setup_vector ──┬──→ compute_equilibrium_suspension  ← stage 1
#                  ├──→ vehicle.simulate_step (1 step)   ← stage 2
#                  ├──→ tire force per wheel             ← stage 3
#                  ├──→ yaw moment from diff             ← stage 4
#                  └──→ 10-step rollout v_x sum          ← stage 5
#
# For each stage, we report which parameters have nonzero gradient. The first
# stage where a parameter becomes nonzero is where it enters the graph; if a
# parameter is zero at all stages, it's truly disconnected.
#
# USAGE:
#   python scripts/diagnose_setup_graph_connectivity.py
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

try:
    import jax_config  # noqa: F401
except ImportError:
    pass

import jax
import jax.numpy as jnp
import numpy as np

from models.vehicle_dynamics import (
    DifferentiableMultiBodyVehicle, build_default_setup_28,
    compute_equilibrium_suspension, SETUP_NAMES,
)
from config.vehicles.ter26 import vehicle_params as VP
from config.tire_coeffs import tire_coeffs as TC


def zero_report(grad_vec: jax.Array, stage: str) -> dict:
    """Report which parameters have zero gradient at a given stage."""
    g = np.array(grad_vec)
    connected = [SETUP_NAMES[i] for i in range(28) if abs(g[i]) > 0]
    disconnected = [SETUP_NAMES[i] for i in range(28) if abs(g[i]) == 0]
    print(f"\n── Stage: {stage} ──")
    print(f"  CONNECTED  ({len(connected):2d}/28): {connected}")
    print(f"  DEAD  ENDS ({len(disconnected):2d}/28): {disconnected}")
    return {"stage": stage, "connected": connected, "disconnected": disconnected}


def main():
    print("=" * 72)
    print("  SETUP PARAMETER GRAPH CONNECTIVITY DIAGNOSTIC")
    print("=" * 72)

    default_setup = build_default_setup_28(VP)
    vehicle = DifferentiableMultiBodyVehicle(VP, TC)

    reports = []

    # ── Stage 1: static equilibrium ──────────────────────────────────────
    # compute_equilibrium_suspension depends on spring rates, heave springs,
    # and (through gravity) sprung mass distribution. Dampers/geometry should
    # NOT show up here — they don't affect static equilibrium.
    def f1(setup):
        z = compute_equilibrium_suspension(setup, VP)
        return jnp.sum(z)  # scalar sum of static travel
    grad1 = jax.grad(f1)(default_setup)
    reports.append(zero_report(grad1, "compute_equilibrium_suspension"))

    # ── Stage 2: single physics step, return sum of state ────────────────
    # A single 1ms step should reveal damper and geometry dependencies
    # (forces act in the first dt, affecting all derivative terms).
    x0 = jnp.zeros(46).at[14].set(15.0)  # rolling at 15 m/s
    z_eq = compute_equilibrium_suspension(default_setup, VP)
    x0 = x0.at[6:10].set(z_eq)
    x0 = x0.at[28:38].set(jnp.array([85., 85., 85., 85., 80.,
                                      85., 85., 85., 85., 80.]))
    u = jnp.array([0.05, 2000.0])  # slight steering + throttle

    def f2(setup):
        x1 = vehicle.simulate_step(x0, u, setup, 0.005)
        return jnp.sum(x1)  # scalar: sum across all 46 state derivatives
    grad2 = jax.grad(f2)(default_setup)
    reports.append(zero_report(grad2, "vehicle.simulate_step(1 step) → Σx"))

    # ── Stage 3: 10-step rollout, return final v_x ──────────────────────
    # This exercises damper rate dependencies via suspension velocity
    # evolution over multiple steps.
    def f3(setup):
        def step_fn(x, _):
            return vehicle.simulate_step(x, u, setup, 0.005), None
        x_final, _ = jax.lax.scan(step_fn, x0, None, length=10)
        return x_final[14]  # final v_x
    grad3 = jax.grad(f3)(default_setup)
    reports.append(zero_report(grad3, "10-step rollout → final v_x"))

    # ── Stage 4: 10-step rollout, return mean |suspension velocity| ─────
    # Dampers SHOULD matter here: they dissipate energy from suspension DOFs.
    def f4(setup):
        def step_fn(x, _):
            x_new = vehicle.simulate_step(x, u, setup, 0.005)
            return x_new, jnp.sum(jnp.abs(x_new[20:24]))  # Σ|ż_susp|
        _, susp_vel_hist = jax.lax.scan(step_fn, x0, None, length=10)
        return jnp.mean(susp_vel_hist)
    grad4 = jax.grad(f4)(default_setup)
    reports.append(zero_report(grad4, "10-step rollout → mean |suspension velocity|"))

    # ── Stage 5: 10-step rollout, return max yaw rate ────────────────────
    # Brake bias, diff lock, ARBs should matter via yaw moment.
    u_hard = jnp.array([0.15, 3000.0])  # harder cornering input
    def f5(setup):
        def step_fn(x, _):
            return vehicle.simulate_step(x, u_hard, setup, 0.005), x[19]
        _, yaw_rate_hist = jax.lax.scan(step_fn, x0, None, length=20)
        return jnp.max(jnp.abs(yaw_rate_hist))
    grad5 = jax.grad(f5)(default_setup)
    reports.append(zero_report(grad5, "20-step rollout (hard turn) → max |yaw rate|"))

    # ── Summary: cross-reference matrix ──────────────────────────────────
    print("\n" + "=" * 72)
    print("  CONNECTIVITY MATRIX")
    print("=" * 72)
    stages = [r["stage"] for r in reports]
    # Truncate stage names for display
    short = [s[:22] for s in stages]
    header = f"  {'Parameter':<22} " + " ".join(f"{s:^24}" for s in short)
    print(header)
    print("  " + "─" * (22 + len(short) * 25))

    for i, name in enumerate(SETUP_NAMES):
        row = f"  {name:<22} "
        for r in reports:
            g = 1 if name in r["connected"] else 0
            sym = "●" if g else "·"
            row += f"{sym:^24} "
        print(row)

    # ── Dead-end analysis ────────────────────────────────────────────────
    all_dead = set(SETUP_NAMES)
    for r in reports:
        all_dead -= set(r["connected"])

    print("\n" + "=" * 72)
    if all_dead:
        print(f"  GLOBALLY DISCONNECTED PARAMETERS ({len(all_dead)}):")
        for name in sorted(all_dead):
            print(f"    · {name}")
        print("\n  These parameters are never read by any physics in the rollout.")
        print("  Likely cause: the code reads these from VP (config) rather than")
        print("  from the setup_vector. Fix: route them through setup_vector in")
        print("  compute_suspension_forces / compute_tire_forces / etc.")
    else:
        print("  ALL parameters appear in at least one physics stage.")
        print("  The lap-time dead-end must be in the reduction:")
        print("  effective_lap_time = track.total_length / mean(v_x)")
        print("  → damper effects on v_x may be smaller than float32 precision.")
    print("=" * 72)


if __name__ == "__main__":
    main()