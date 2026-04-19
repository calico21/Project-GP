# powertrain/modes/advanced/explicit_mpqp_allocator.py
# Project-GP — Batch 3: Explicit mpQP KKT Runtime Torque Allocator
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces the 12-iteration projected-gradient SOCP scan with a single KKT
# linear solve conditioned on the Batch 2 active-set classifier prediction.
#
# TVGeometry, AllocatorWeights, and yaw_moment_arms are inlined here to avoid
# a circular import (torque_vectoring → this module → torque_vectoring).
# Values are identical to torque_vectoring.py — single source of truth is
# maintained via the module docstring note; any change in torque_vectoring.py
# must be mirrored here.
#
# PIPELINE (~30–80 µs post-JIT):
#   1. Classifier → active-set A ∈ {0,1}¹²            (~5 µs)
#   2. Build KKT matrix K ∈ ℝ^{16×16} (static shape)  (~2 µs)
#   3. jnp.linalg.solve → [T*, λ*]                    (~10–20 µs)
#   4. Verify primal/dual feasibility                   (~3 µs)
#   5. 3-step polish if infeasible (~3% of calls)       (~20 µs max)
#
# CONSTRAINT ORDERING (must match active_set_classifier.py):
#   [0:4]   lower box  T_i = T_min_i
#   [4:8]   upper box  T_i = T_max_i
#   [8:12]  friction   |T_i| = T_fric_i  (sign from T_prev)
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from powertrain.modes.advanced.active_set_classifier import (
    ClassifierBundle, predict_active_set, normalise_theta,
    THETA_DIM, N_CONSTRAINTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Inlined geometry / weights (mirrors torque_vectoring.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class TVGeometry(NamedTuple):
    lf: float = 0.8525
    lr: float = 0.6975
    track_f: float = 1.200
    track_r: float = 1.180
    r_w: float = 0.2032
    h_cg: float = 0.330
    I_z: float = 150.0
    mass: float = 300.0
    kingpin_offset: float = 0.020
    mechanical_trail: float = 0.015
    steer_ratio: float = 4.2
    anti_squat: float = 0.30
    anti_dive_f: float = 0.40


class AllocatorWeights(NamedTuple):
    w_mz:   float = 8.0
    w_fx:   float = 1.0
    w_rate: float = 0.05
    w_loss: float = 1e-4


@jax.jit
def yaw_moment_arms(delta: jax.Array,
                    geo: TVGeometry = TVGeometry()) -> jax.Array:
    """Yaw moment arm per wheel [m] — identical to torque_vectoring.yaw_moment_arms."""
    cos_d = jnp.cos(delta)
    sin_d = jnp.sin(delta)
    arm_fl = -geo.track_f / 2.0 * cos_d + geo.lf * sin_d
    arm_fr = +geo.track_f / 2.0 * cos_d - geo.lf * sin_d
    arm_rl = -geo.track_r / 2.0
    arm_rr = +geo.track_r / 2.0
    return jnp.array([arm_fl, arm_fr, arm_rl, arm_rr]) / geo.r_w


# ─────────────────────────────────────────────────────────────────────────────
# §2  QP parameter struct
# ─────────────────────────────────────────────────────────────────────────────

class QPParams(NamedTuple):
    mz_ref:  jax.Array   # scalar  [Nm]
    fx_d:    jax.Array   # scalar  [N]
    t_min:   jax.Array   # (4,)    [Nm]
    t_max:   jax.Array   # (4,)    [Nm]
    t_fric:  jax.Array   # (4,)    [Nm]
    delta:   jax.Array   # scalar  [rad]
    t_prev:  jax.Array   # (4,)    [Nm]
    omega:   jax.Array   # (4,)    [rad/s]


def pack_theta_raw(p: QPParams) -> jax.Array:
    return jnp.concatenate([
        p.mz_ref[None], p.fx_d[None],
        p.t_min, p.t_max, p.t_fric,
        p.delta[None],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# §3  QP cost matrices
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("geo", "weights"))
def build_qp_matrices(
    p:       QPParams,
    geo:     TVGeometry      = TVGeometry(),
    weights: AllocatorWeights = AllocatorWeights(),
) -> tuple[jax.Array, jax.Array]:
    arms    = yaw_moment_arms(p.delta, geo)
    ones_rw = jnp.ones(4) / geo.r_w

    Q = (2.0 * weights.w_mz   * jnp.outer(arms, arms)
       + 2.0 * weights.w_fx   * jnp.outer(ones_rw, ones_rw)
       + 2.0 * weights.w_rate  * jnp.eye(4)
       + 2.0 * weights.w_loss  * jnp.diag(jnp.abs(p.omega) + 1.0))

    c = (-2.0 * weights.w_mz   * p.mz_ref * arms
       - 2.0 * weights.w_fx   * p.fx_d   * ones_rw
       - 2.0 * weights.w_rate  * p.t_prev)

    return Q, c


# ─────────────────────────────────────────────────────────────────────────────
# §4  KKT system assembly
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def build_kkt_system(
    Q:          jax.Array,
    c:          jax.Array,
    active_set: jax.Array,
    p:          QPParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Build static-shape (16×16) KKT system. Inactive constraint rows are zeroed.

    [ Q      A_actᵀ ] [ T ]   [ -c    ]
    [ A_act  0      ] [ λ ] = [ b_act ]
    """
    N, M = 4, N_CONSTRAINTS

    I4 = jnp.eye(4)

    # Each active constraint becomes: e_i^T T = bound_i
    b_lower = p.t_min
    b_upper = p.t_max
    fric_sign = jnp.sign(p.t_prev + 1e-6)
    b_fric  = fric_sign * p.t_fric

    A_full = jnp.concatenate([I4, I4, I4], axis=0)              # (12, 4)
    b_full = jnp.concatenate([b_lower, b_upper, b_fric])        # (12,)

    # Zero out inactive rows
    mask  = active_set[:, None]
    A_act = A_full * mask
    b_act = b_full * active_set

    top    = jnp.concatenate([Q,     A_act.T],              axis=1)  # (4, 16)
    bottom = jnp.concatenate([A_act, jnp.zeros((M, M))],   axis=1)  # (12, 16)
    K      = jnp.concatenate([top, bottom], axis=0)                  # (16, 16)
    rhs    = jnp.concatenate([-c, b_act])                            # (16,)

    return K, rhs


# ─────────────────────────────────────────────────────────────────────────────
# §5  KKT solve + feasibility check
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def solve_kkt(K: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
    sol = jnp.linalg.solve(K + 1e-8 * jnp.eye(K.shape[0]), rhs)
    return sol[:4], sol[4:]


@jax.jit
def check_kkt_feasibility(
    T:          jax.Array,
    lambda_:    jax.Array,
    active_set: jax.Array,
    p:          QPParams,
    tol:        float = 3.0,
) -> jax.Array:
    primal_ok = (jnp.all(T >= p.t_min - tol)
                 & jnp.all(T <= p.t_max + tol)
                 & jnp.all(jnp.abs(T) <= p.t_fric + tol))
    dual_ok   = jnp.all((lambda_ >= -1.0) | (active_set < 0.5))
    return primal_ok & dual_ok


# ─────────────────────────────────────────────────────────────────────────────
# §6  Polish fallback
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def polish_step(
    T_init:  jax.Array,
    Q:       jax.Array,
    c:       jax.Array,
    p:       QPParams,
    n_steps: int   = 3,
    lr:      float = 0.30,
) -> jax.Array:
    def one_step(T, _):
        T_new = T - lr * (Q @ T + c)
        T_new = jnp.clip(T_new, p.t_min, p.t_max)
        T_new = jnp.clip(T_new, -p.t_fric, p.t_fric)
        return T_new, None
    T_opt, _ = jax.lax.scan(one_step, T_init, None, length=n_steps)
    return T_opt


# ─────────────────────────────────────────────────────────────────────────────
# §7  Full allocator step
# ─────────────────────────────────────────────────────────────────────────────

def make_explicit_allocator_step(clf_bundle: ClassifierBundle):
    from powertrain.modes.advanced.active_set_classifier import make_predict_fns
    _predict, _ = make_predict_fns(clf_bundle)   # compiled once, closed over bundle

    @jax.jit
    def _step(
        p:       QPParams,
        geo:     TVGeometry      = TVGeometry(),
        weights: AllocatorWeights = AllocatorWeights(),
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        theta_norm = normalise_theta(pack_theta_raw(p))
        active_set = _predict(theta_norm)           # no bundle arg — closed over
        Q, c       = build_qp_matrices(p, geo, weights)
        K, rhs     = build_kkt_system(Q, c, active_set, p)
        T_kkt, lam = solve_kkt(K, rhs)
        feasible   = check_kkt_feasibility(T_kkt, lam, active_set, p)
        T_opt = jax.lax.cond(
            feasible,
            lambda t: t,
            lambda t: polish_step(t, Q, c, p),
            T_kkt,
        )
        return T_opt, active_set, ~feasible

    return _step


def explicit_allocator_step(
    clf_bundle: ClassifierBundle,
    p:          QPParams,
    geo:        TVGeometry       = TVGeometry(),
    weights:    AllocatorWeights  = AllocatorWeights(),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Thin wrapper — creates a compiled step fn and calls it immediately."""
    return make_explicit_allocator_step(clf_bundle)(p, geo, weights)

# ─────────────────────────────────────────────────────────────────────────────
# §8  Integration shim
# ─────────────────────────────────────────────────────────────────────────────

class ExplicitAllocatorState(NamedTuple):
    T_prev:       jax.Array
    polish_count: jax.Array


def init_explicit_allocator_state() -> ExplicitAllocatorState:
    return ExplicitAllocatorState(
        T_prev=jnp.zeros(4),
        polish_count=jnp.array(0, dtype=jnp.int32),
    )


def make_explicit_allocator(clf_bundle: ClassifierBundle,
                            geo: TVGeometry = TVGeometry(),
                            weights: AllocatorWeights = AllocatorWeights()):
    @jax.jit
    def alloc_fn(state, mz_ref, fx_d, t_min, t_max, t_fric, delta, omega):
        p = QPParams(
            mz_ref=mz_ref, fx_d=fx_d,
            t_min=t_min, t_max=t_max, t_fric=t_fric,
            delta=delta, t_prev=state.T_prev, omega=omega,
        )
        T_opt, _, polished = explicit_allocator_step(clf_bundle, p, geo, weights)
        new_state = ExplicitAllocatorState(
            T_prev=T_opt,
            polish_count=state.polish_count + polished.astype(jnp.int32),
        )
        return T_opt, new_state
    return alloc_fn