# powertrain/modes/registry.py
# Project-GP — Mode Registry & Pipeline Dispatcher
# ═══════════════════════════════════════════════════════════════════════════════
#
# SIMPLE vs ADVANCED pipeline contracts differ in fundamental ways:
#
#   SIMPLE pipeline (real-time embedded, target: <1 ms/step):
#     T_base  = simple_dyc_torque_vectoring(vx, wz, delta, Fx, T_min, T_max)
#     T_final = tc_simple(T_base, omega, vx, state, params, dt)
#     → Direct torque correction. No SOCP. No CBF. No DESC.
#
#   ADVANCED pipeline (offline twin, target: <5 ms/step post-JIT):
#     tc_out        = tc_step(vx, vy, ..., state, dt)          # DESC + kappa*
#     T_alloc       = solve_torque_allocation(kappa_star, ...)  # SOCP 12-iter
#     T_final       = cbf_safety_filter(T_alloc, ...)          # CBF 3-iter
#     → kappa* references feed the SOCP. SOCP output goes through CBF.
#
# The registry does NOT force identical signatures across modes.
# Callers select a mode and use the returned pipeline accordingly.
# The powertrain_manager.py (ADVANCED) and any future simple_manager.py
# (SIMPLE) are the canonical callers — they know their pipeline shape.
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
from enum import Enum
from typing import NamedTuple, Callable


# ─────────────────────────────────────────────────────────────────────────────
# §1  Mode Enum
# ─────────────────────────────────────────────────────────────────────────────

class ControlMode(str, Enum):
    SIMPLE   = "simple"
    ADVANCED = "advanced"

    @staticmethod
    def from_env(default: str = "advanced") -> "ControlMode":
        """Read GP_CONTROL_MODE env var; fall back to default."""
        raw = os.environ.get("GP_CONTROL_MODE", default).lower().strip()
        try:
            return ControlMode(raw)
        except ValueError:
            raise ValueError(
                f"GP_CONTROL_MODE='{raw}' is invalid. "
                f"Valid values: {[m.value for m in ControlMode]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Pipeline Containers (typed, not Protocol — avoids fake signature parity)
# ─────────────────────────────────────────────────────────────────────────────

class SimplePipeline(NamedTuple):
    """
    Callables for SIMPLE mode. Both are JIT-compiled at construction.

    tv_allocate: simple_dyc_torque_vectoring
        (vx, wz, delta, Fx_driver, T_min, T_max, Kp_yaw?, geo?) → T (4,)

    tc_correct: tc_simple
        (T_requested, omega_wheel, vx, state, params, dt) → (T_out, state, diag)
    """
    tv_allocate: Callable
    tc_correct: Callable
    mode: ControlMode = ControlMode.SIMPLE


class AdvancedPipeline(NamedTuple):
    """
    Callables for ADVANCED mode. All are JIT-compiled at construction.

    tc_step: DESC + kappa* fusion
        (vx, vy, ax, ay, omega_wheel, alpha_t, Fz, T_applied, T_tire,
         mu_est, gp_sigma, tc_state, dt, desc_params, tc_weights, r_w)
        → (TCOutput, TCState)

    solve_allocation: projected-gradient SOCP (12 iters)
        (T_warmstart, T_prev, Fx_target, Mz_target, delta, Fz, Fy, mu,
         omega_wheel, T_min, T_max, P_max, geo, w) → T (4,)

    cbf_filter: input-delay CBF safety projection (3 iters)
        (T_alloc, T_prev, vx, vy, wz, Fz, Fy_total, mu_est,
         omega_wheel, T_min, T_max, gp_sigma, geo, cbf) → T (4,)
    """
    tc_step: Callable
    solve_allocation: Callable
    cbf_filter: Callable
    mode: ControlMode = ControlMode.ADVANCED


# ─────────────────────────────────────────────────────────────────────────────
# §3  Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_powertrain(mode: ControlMode) -> SimplePipeline | AdvancedPipeline:
    """
    Instantiate a mode-specific pipeline.

    Imports are deferred to this call so that each mode's dependencies are
    only loaded when actually needed. Both modes' callables are JIT-compiled
    at construction time — the first call is the trace, not the hot path.

    Args:
        mode: ControlMode.SIMPLE or ControlMode.ADVANCED

    Returns:
        SimplePipeline or AdvancedPipeline (NamedTuple of JIT-compiled fns)
    """
    import jax

    if mode is ControlMode.SIMPLE:
        from powertrain.modes.simple.torque_vectoring import simple_dyc_torque_vectoring
        from powertrain.modes.simple.traction_control import tc_simple

        return SimplePipeline(
            tv_allocate=jax.jit(simple_dyc_torque_vectoring),
            tc_correct=jax.jit(tc_simple),
        )

    elif mode is ControlMode.ADVANCED:
        from powertrain.modes.advanced.torque_vectoring import (
            solve_torque_allocation, cbf_safety_filter,
        )
        from powertrain.modes.advanced.traction_control import tc_step

        return AdvancedPipeline(
            tc_step=jax.jit(tc_step),
            solve_allocation=jax.jit(solve_torque_allocation),
            cbf_filter=jax.jit(cbf_safety_filter),
        )

    else:
        raise ValueError(f"Unknown ControlMode: {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# §4  Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline() -> SimplePipeline | AdvancedPipeline:
    """Build pipeline from GP_CONTROL_MODE env var (default: advanced)."""
    return build_powertrain(ControlMode.from_env())