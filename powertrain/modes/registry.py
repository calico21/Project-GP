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
    SIMPLE       = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"

    @staticmethod
    def from_env(default: str = "advanced") -> "ControlMode":
        """Read GP_CONTROL_MODE env var; fall back to default."""
        # Valid values: simple | intermediate | advanced
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
    Callables for SIMPLE mode.

    tv_allocate: simple_dyc_torque_vectoring
        (..., is_rwd=pipeline.is_rwd) → T (4,)
        Callers must forward pipeline.is_rwd as a keyword arg.
    tc_correct: tc_simple
        (T_requested, omega_wheel, vx, state, params, dt) → (T_out, state, diag)
    """
    tv_allocate: Callable
    tc_correct: Callable
    is_rwd: bool = False
    mode: ControlMode = ControlMode.SIMPLE


class IntermediatePipeline(NamedTuple):
    """
    Callables for INTERMEDIATE mode.

    tv_allocate: intermediate_tv_step  (already @partial(jax.jit, static_argnames=...))
        (vx, wz, delta, ax, Fx_driver, mu_est, T_min_hw, T_max_hw,
         tv_state, dt, geo?, params?, is_rwd?) → (IntermediateTVOutput, IntermediateTVState)

    tc_correct: tc_simple  (reused from SIMPLE — PI slip correction, stateful)
        (T_requested, omega_wheel, vx, state, params, dt) → (T_out, state, diag)

    Note: tv_allocate is NOT wrapped with jax.jit() here because intermediate_tv_step
    carries static_argnames — double-wrapping drops those, recompiling on every call.
    """
    tv_allocate: Callable
    tc_correct: Callable
    is_rwd: bool = False
    mode: ControlMode = ControlMode.INTERMEDIATE


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
    is_rwd: bool = False
    mode: ControlMode = ControlMode.ADVANCED


# ─────────────────────────────────────────────────────────────────────────────
# §3  Factory
# ─────────────────────────────────────────────────────────────────────────────

def _is_rwd_from_env(default: bool = False) -> bool:
    """Read GP_DRIVE_CONFIG env var. Values: 'rwd' → True, 'awd' → False."""
    raw = os.environ.get("GP_DRIVE_CONFIG", "rwd" if default else "awd").lower().strip()
    if raw == "rwd":
        return True
    if raw == "awd":
        return False
    raise ValueError(
        f"GP_DRIVE_CONFIG='{raw}' is invalid. Valid values: 'rwd', 'awd'"
    )


def build_powertrain(
    mode: ControlMode,
    is_rwd: bool = False,
) -> SimplePipeline | IntermediatePipeline | AdvancedPipeline:
    """
    Instantiate a mode-specific pipeline.

    Args:
        mode:   ControlMode.SIMPLE | INTERMEDIATE | ADVANCED
        is_rwd: True  → Ter26 RWD (driven=[RL,RR], front bounds zeroed in SOCP/CBF)
                False → Ter27 AWD (driven=[FL,FR,RL,RR], full 4-wheel allocation)

    is_rwd is stored on the returned pipeline so callers can read it via
    pipeline.is_rwd rather than tracking it separately.
    The value is a Python bool — each (mode, is_rwd) pair compiles a distinct XLA graph.
    """
    import jax

    if mode is ControlMode.SIMPLE:
        from powertrain.modes.simple.torque_vectoring import simple_dyc_torque_vectoring
        from powertrain.modes.simple.traction_control import tc_simple

        return SimplePipeline(
            tv_allocate=simple_dyc_torque_vectoring,  # static_argnums=(6,7,8); do not re-jit
            tc_correct=jax.jit(tc_simple),
            is_rwd=is_rwd,
        )

    elif mode is ControlMode.INTERMEDIATE:
        from powertrain.modes.intermediate.torque_vectoring import intermediate_tv_step
        from powertrain.modes.simple.traction_control import tc_simple

        return IntermediatePipeline(
            tv_allocate=intermediate_tv_step,  # static_argnames=(...,'is_rwd'); do not re-jit
            tc_correct=jax.jit(tc_simple),
            is_rwd=is_rwd,
        )

    elif mode is ControlMode.ADVANCED:
        from powertrain.modes.advanced.torque_vectoring import (
            solve_torque_allocation, cbf_safety_filter,
        )
        from powertrain.modes.advanced.traction_control import tc_step

        return AdvancedPipeline(
            tc_step=jax.jit(tc_step),
            solve_allocation=solve_torque_allocation,  # static_argnames=('is_rwd',); do not re-jit
            cbf_filter=cbf_safety_filter,              # same
            is_rwd=is_rwd,
        )

    else:
        raise ValueError(f"Unknown ControlMode: {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# §4  Convenience entry points
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline() -> SimplePipeline | IntermediatePipeline | AdvancedPipeline:
    """Build pipeline from GP_CONTROL_MODE + GP_DRIVE_CONFIG env vars."""
    return build_powertrain(
        mode=ControlMode.from_env(),
        is_rwd=_is_rwd_from_env(default=True),  # default RWD = Ter26 (current car)
    )