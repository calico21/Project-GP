#!/usr/bin/env python3
# scripts/benchmark_socp_latency.py
# Project-GP — SOCP Allocator Deployment Latency Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Validate the 5 ms real-time budget claim (Test 16) by measuring:
#     1. Cold-start latency (T_warmstart = zeros)
#     2. Warm-start latency (T_warmstart = previous solution)
#     3. Sequential warm-start chain (simulates 200 Hz deployment loop)
#     4. Full powertrain_step() latency (all 13 stages)
#
#   Produces a JSON report suitable for the FSG Digital Twin Award submission.
#
# USAGE:
#   python scripts/benchmark_socp_latency.py [--n-warmup 5] [--n-trials 200]
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import statistics

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

try:
    import jax_config  # noqa: F401
except ImportError:
    pass

import jax
import jax.numpy as jnp
import numpy as np


def benchmark_socp_allocator(n_warmup: int = 5, n_trials: int = 200) -> dict:
    """Benchmark solve_torque_allocation cold vs warm latency."""
    from powertrain.modes.advanced.torque_vectoring import (
        solve_torque_allocation, TVGeometry, AllocatorWeights,
    )

    geo = TVGeometry()
    w = AllocatorWeights()

    # Realistic operating scenarios (not just the easy straight-line case)
    scenarios = {
        "straight_accel": dict(
            Fx_target=3000.0, Mz_target=0.0, delta=0.0,
            Fz=[700., 700., 800., 800.], Fy=[0., 0., 0., 0.],
        ),
        "hard_cornering": dict(
            Fx_target=500.0, Mz_target=120.0, delta=0.12,
            Fz=[500., 900., 550., 950.], Fy=[450., 820., 480., 870.],
        ),
        "trail_braking": dict(
            Fx_target=-2000.0, Mz_target=80.0, delta=0.08,
            Fz=[850., 850., 550., 550.], Fy=[300., 300., 200., 200.],
        ),
        "hairpin_exit": dict(
            Fx_target=4000.0, Mz_target=-60.0, delta=-0.15,
            Fz=[600., 550., 900., 850.], Fy=[-500., -400., -700., -600.],
        ),
    }

    results = {}

    for name, sc in scenarios.items():
        Fx_target = jnp.array(sc["Fx_target"])
        Mz_target = jnp.array(sc["Mz_target"])
        delta = jnp.array(sc["delta"])
        Fz = jnp.array(sc["Fz"])
        Fy = jnp.array(sc["Fy"])
        mu = jnp.full(4, 1.4)
        omega_w = jnp.full(4, 15.0 / 0.2032)
        T_min = jnp.full(4, -400.0)
        T_max = jnp.full(4, 450.0)
        P_max = jnp.array(80000.0)

        # ── Cold-start benchmark ────────────────────────────────────────
        T_cold = jnp.zeros(4)

        # JIT warmup
        for _ in range(n_warmup):
            T_out = solve_torque_allocation(
                T_cold, T_cold, Fx_target, Mz_target, delta,
                Fz, Fy, mu, omega_w, T_min, T_max, P_max,
                jnp.full(4, 25.0), geo, w,
            )
            _ = float(T_out[0])  # force materialisation

        cold_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            T_out = solve_torque_allocation(
                T_cold, T_cold, Fx_target, Mz_target, delta,
                Fz, Fy, mu, omega_w, T_min, T_max, P_max,
                jnp.full(4, 25.0), geo, w,
            )
            _ = float(T_out[0])
            cold_times.append((time.perf_counter() - t0) * 1000)

        # ── Warm-start benchmark (sequential chain) ─────────────────────
        warm_times = []
        T_prev = T_cold
        for _ in range(n_trials):
            t0 = time.perf_counter()
            T_out = solve_torque_allocation(
                T_prev, T_prev, Fx_target, Mz_target, delta,
                Fz, Fy, mu, omega_w, T_min, T_max, P_max,
                jnp.full(4, 25.0), geo, w,
            )
            _ = float(T_out[0])
            warm_times.append((time.perf_counter() - t0) * 1000)
            T_prev = T_out  # chain: each call warm-starts from previous

        # ── Convergence quality: compare cold vs warm final cost ────────
        from powertrain.modes.advanced.torque_vectoring import allocator_cost
        T_cold_sol = solve_torque_allocation(
            T_cold, T_cold, Fx_target, Mz_target, delta,
            Fz, Fy, mu, omega_w, T_min, T_max, P_max,
            jnp.full(4, 25.0), geo, w,
        )
        T_warm_sol = solve_torque_allocation(
            T_prev, T_prev, Fx_target, Mz_target, delta,
            Fz, Fy, mu, omega_w, T_min, T_max, P_max,
            jnp.full(4, 25.0), geo, w,
        )
        cost_cold = float(allocator_cost(
            T_cold_sol, T_cold, Fx_target, Mz_target, delta,
            Fz, Fy, mu, omega_w, T_min, T_max, P_max,
            jnp.full(4, 25.0), geo, w,
        ))
        cost_warm = float(allocator_cost(
            T_warm_sol, T_prev, Fx_target, Mz_target, delta,
            Fz, Fy, mu, omega_w, T_min, T_max, P_max,
            jnp.full(4, 25.0), geo, w,
        ))

        results[name] = {
            "cold_start_ms": {
                "mean": statistics.mean(cold_times),
                "median": statistics.median(cold_times),
                "p95": np.percentile(cold_times, 95),
                "p99": np.percentile(cold_times, 99),
                "std": statistics.stdev(cold_times),
            },
            "warm_start_ms": {
                "mean": statistics.mean(warm_times),
                "median": statistics.median(warm_times),
                "p95": np.percentile(warm_times, 95),
                "p99": np.percentile(warm_times, 99),
                "std": statistics.stdev(warm_times),
            },
            "cost_cold": cost_cold,
            "cost_warm": cost_warm,
            "cost_improvement_pct": (cost_cold - cost_warm) / (abs(cost_cold) + 1e-8) * 100,
            "T_cold": [float(x) for x in T_cold_sol],
            "T_warm": [float(x) for x in T_warm_sol],
        }

    return results


def benchmark_full_powertrain_step(n_warmup: int = 3, n_trials: int = 100) -> dict:
    """Benchmark the full 13-stage powertrain_step() JIT latency."""
    try:
        from powertrain.powertrain_manager import (
            powertrain_step, make_manager_state, PowertrainConfig,
        )
        from powertrain.motor_model import MotorParams, BatteryParams
        from powertrain.modes.advanced.torque_vectoring import TVGeometry

        config = PowertrainConfig()
        state = make_manager_state()
        geo = TVGeometry()
        mp = MotorParams()
        bp = BatteryParams()

        # Minimal inputs
        throttle = jnp.array(0.7)
        brake = jnp.array(0.0)
        delta = jnp.array(0.05)
        vx = jnp.array(15.0)
        vy = jnp.array(0.3)
        wz = jnp.array(0.8)
        ax = jnp.array(2.0)
        omega_wheel = jnp.full(4, 15.0 / 0.2032)
        Fz = jnp.array([700., 700., 800., 800.])
        Fy = jnp.array([200., 200., 300., 300.])
        mu_est = jnp.array(1.4)
        curvature = jnp.array(0.02)
        gp_sigma = jnp.array(0.05)
        launch_button = jnp.array(0.0)
        dt = jnp.array(0.005)

        # JIT warmup
        for _ in range(n_warmup):
            out, new_state = powertrain_step(
                throttle, brake, delta, vx, vy, wz, ax,
                omega_wheel, Fz, Fy, mu_est, curvature,
                gp_sigma, launch_button, dt,
                state, config, geo, mp, bp,
            )
            _ = float(out.T_wheel[0])

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            out, new_state = powertrain_step(
                throttle, brake, delta, vx, vy, wz, ax,
                omega_wheel, Fz, Fy, mu_est, curvature,
                gp_sigma, launch_button, dt,
                state, config, geo, mp, bp,
            )
            _ = float(out.T_wheel[0])
            times.append((time.perf_counter() - t0) * 1000)
            state = new_state  # chain states

        return {
            "powertrain_step_ms": {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "p95": np.percentile(times, 95),
                "p99": np.percentile(times, 99),
                "std": statistics.stdev(times),
            },
            "within_5ms_budget": np.percentile(times, 99) < 5.0,
            "n_trials": n_trials,
        }
    except ImportError as e:
        return {"error": f"Could not import powertrain_manager: {e}"}


def main():
    parser = argparse.ArgumentParser(description="SOCP Latency Benchmark")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--output", type=str, default="reports/socp_latency_report.json")
    args = parser.parse_args()

    print("=" * 72)
    print("  PROJECT-GP  ·  SOCP DEPLOYMENT LATENCY BENCHMARK")
    print("=" * 72)

    # ── Platform info ─────────────────────────────────────────────────────
    platform_info = {
        "jax_version": jax.__version__,
        "backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(f"\n  JAX {platform_info['jax_version']} on {platform_info['backend']}")
    print(f"  Devices: {platform_info['devices']}")

    # ── SOCP allocator benchmark ──────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Phase 1: SOCP Allocator (12-iter projected gradient)")
    print(f"{'─' * 72}")

    socp_results = benchmark_socp_allocator(args.n_warmup, args.n_trials)

    for name, r in socp_results.items():
        cs = r["cold_start_ms"]
        ws = r["warm_start_ms"]
        print(f"\n  [{name}]")
        print(f"    Cold:  {cs['mean']:.3f} ms (p95={cs['p95']:.3f}, p99={cs['p99']:.3f})")
        print(f"    Warm:  {ws['mean']:.3f} ms (p95={ws['p95']:.3f}, p99={ws['p99']:.3f})")
        print(f"    Cost:  cold={r['cost_cold']:.2f}  warm={r['cost_warm']:.2f}  "
              f"(Δ={r['cost_improvement_pct']:+.1f}%)")
        speedup = cs['mean'] / (ws['mean'] + 1e-8)
        print(f"    Speedup: {speedup:.1f}×")

    # ── Full powertrain step benchmark ────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Phase 2: Full powertrain_step() (13 stages)")
    print(f"{'─' * 72}")

    pt_results = benchmark_full_powertrain_step(args.n_warmup, min(args.n_trials, 100))

    if "error" not in pt_results:
        ps = pt_results["powertrain_step_ms"]
        budget_ok = pt_results["within_5ms_budget"]
        print(f"\n    Mean:   {ps['mean']:.3f} ms")
        print(f"    p95:    {ps['p95']:.3f} ms")
        print(f"    p99:    {ps['p99']:.3f} ms")
        print(f"    Budget: {'✓ PASS' if budget_ok else '✗ FAIL'} (p99 < 5 ms)")
    else:
        print(f"    {pt_results['error']}")

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "platform": platform_info,
        "socp_allocator": socp_results,
        "powertrain_step": pt_results,
        "summary": {
            "socp_warm_mean_ms": statistics.mean(
                r["warm_start_ms"]["mean"] for r in socp_results.values()
            ),
            "socp_cold_mean_ms": statistics.mean(
                r["cold_start_ms"]["mean"] for r in socp_results.values()
            ),
            "all_scenarios_within_budget": all(
                r["warm_start_ms"]["p99"] < 5.0 for r in socp_results.values()
            ),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
