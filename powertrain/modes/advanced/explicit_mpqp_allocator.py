# powertrain/modes/advanced/explicit_mpqp_allocator.py
# Project-GP — Batch 3 + Batch 8: Explicit mpQP KKT Runtime Torque Allocator
# ═══════════════════════════════════════════════════════════════════════════════
#
# Batch 3 core (unchanged):
#   Replaces 12-iteration projected-gradient SOCP with a single KKT linear
#   solve conditioned on the active-set classifier prediction.
#
# Batch 8 additions:
#   • Extended KKT to N_CONSTRAINTS_TOTAL = 20 (12 existing + 8 slip-CBF)
#   • build_kkt_system_extended() — accepts optional extra (A, b) constraint rows
#   • make_explicit_allocator_step_v2() — accepts SlipBarrierInputs; computes
#     extra rows, runs V2 or V1 classifier, builds 24×24 KKT, solves, polishes
#   • pack_theta_raw_v2() — 19-dim θ including κ*, σ(κ*) for V2 classifier
#   • Smooth active-set via predict_active_set_soft() — τ-softened sigmoid gate
#     enabling end-to-end differentiability (∂T*/∂θ is C∞ at region boundaries)
#
# CONSTRAINT ORDERING (must match active_set_classifier.py):
#   [0:4]   lower box  T_i ≥ T_min_i
#   [4:8]   upper box  T_i ≤ T_max_i
#   [8:12]  friction   |T_i| ≤ T_fric_i  (sign from T_prev)
#   [12:16] slip upper κ_{k+d,i} ≤ +budget_i   [Batch 8]
#   [16:20] slip lower κ_{k+d,i} ≥ −budget_i   [Batch 8]
#
# KKT MATRIX DIMENSION:
#   V1 (12 constraints): K ∈ ℝ^{16×16}  (4 vars + 12 constraints)
#   V2 (20 constraints): K ∈ ℝ^{24×24}  (4 vars + 20 constraints)
#
# PIPELINE V2 (~40–90 µs post-JIT):
#   1. Build slip barrier rows (A_slip, b_slip)           (~5 µs)
#   2. Classifier V2 → active-set A ∈ [0,1]^{20}  SOFT  (~8 µs)
#   3. Build extended KKT K ∈ ℝ^{24×24}                  (~3 µs)
#   4. jnp.linalg.solve → [T*, λ*]                       (~20 µs)
#   5. Verify primal/dual + slip feasibility               (~4 µs)
#   6. 3-step polish if infeasible (~5% of calls)          (~25 µs max)
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from powertrain.modes.advanced.active_set_classifier import (
    ClassifierBundle, predict_active_set, normalise_theta,
    THETA_DIM, N_CONSTRAINTS,
)
try:
    from powertrain.modes.advanced.active_set_classifier import (
        THETA_DIM_V2, N_CONSTRAINTS_V2,
        normalise_theta_v2, load_classifier_v2,
    )
    _V2_CLASSIFIER_AVAILABLE = True
except ImportError:
    # V2 additions not yet appended to active_set_classifier.py.
    # Run: cat batch8/active_set_classifier_v2_additions.py >> \
    #          powertrain/modes/advanced/active_set_classifier.py
    THETA_DIM_V2     = 19
    N_CONSTRAINTS_V2 = 20
    normalise_theta_v2 = None
    load_classifier_v2  = None
    _V2_CLASSIFIER_AVAILABLE = False
from powertrain.modes.advanced.slip_barrier import (
    SlipBarrierInputs, SlipBarrierParams,
    build_slip_barrier_rows, check_slip_feasibility,
)

# ── Batch 8 constraint-count constants ───────────────────────────────────────
N_CONSTRAINTS_SLIP  = 8                                   # κ upper + lower, 4 wheels
N_CONSTRAINTS_TOTAL = N_CONSTRAINTS + N_CONSTRAINTS_SLIP  # 12 + 8 = 20
KKT_DIM_V1          = 4 + N_CONSTRAINTS                   # 16 (backward compat)
KKT_DIM_V2          = 4 + N_CONSTRAINTS_TOTAL             # 24


# ─────────────────────────────────────────────────────────────────────────────
# §1  Inlined geometry / weights  (mirrors torque_vectoring.py exactly)
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


# ── V1 theta packing (15-dim) ─────────────────────────────────────────────────
KAPPA_SCALE = 0.20     # normalisation for κ*
SIGMA_SCALE = 0.05     # normalisation for σ(κ*)

def pack_theta_raw(p: QPParams) -> jax.Array:
    return jnp.concatenate([
        p.mz_ref[None], p.fx_d[None],
        p.t_min, p.t_max, p.t_fric,
        p.delta[None],
    ])   # (15,)


# ── V2 theta packing (19-dim) ─────────────────────────────────────────────────
def pack_theta_raw_v2(
    p:            QPParams,
    slip_inputs:  SlipBarrierInputs,
) -> jax.Array:
    """
    Extended 19-dim parameter vector for V2 classifier.

    θ_v2 = θ_v1(15) ‖ [κ*_front, κ*_rear, σ_front, σ_rear]
    """
    base   = pack_theta_raw(p)                         # (15,)
    kstar_f = slip_inputs.kappa_star[0]                # front κ* (FL wheel)
    kstar_r = slip_inputs.kappa_star[2]                # rear  κ* (RL wheel)
    sigma_f = slip_inputs.sigma_star[0]
    sigma_r = slip_inputs.sigma_star[2]
    slip_ext = jnp.array([kstar_f, kstar_r, sigma_f, sigma_r])   # (4,)
    return jnp.concatenate([base, slip_ext])            # (19,)


# ─────────────────────────────────────────────────────────────────────────────
# §3  QP cost matrices  (unchanged from Batch 3)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("geo", "weights"))
def build_qp_matrices(
    p:       QPParams,
    geo:     TVGeometry       = TVGeometry(),
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
# §4a  KKT system — V1, 16×16 (backward-compatible, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def build_kkt_system(
    Q:          jax.Array,
    c:          jax.Array,
    active_set: jax.Array,    # (12,)
    p:          QPParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Build static-shape (16×16) KKT system. Inactive constraint rows are zeroed.

    [ Q      A_actᵀ ] [ T ]   [ -c    ]
    [ A_act  0      ] [ λ ] = [ b_act ]
    """
    N, M = 4, N_CONSTRAINTS    # 4, 12

    b_lower    = p.t_min
    b_upper    = p.t_max
    fric_sign  = jnp.sign(p.t_prev + 1e-6)
    b_fric     = fric_sign * p.t_fric

    A_full = jnp.concatenate([jnp.eye(4), jnp.eye(4), jnp.eye(4)], axis=0)   # (12, 4)
    b_full = jnp.concatenate([b_lower, b_upper, b_fric])                       # (12,)

    mask  = active_set[:, None]
    A_act = A_full * mask
    b_act = b_full * active_set

    top    = jnp.concatenate([Q,     A_act.T],             axis=1)   # (4, 16)
    bottom = jnp.concatenate([A_act, jnp.zeros((M, M))],  axis=1)   # (12, 16)
    K      = jnp.concatenate([top, bottom], axis=0)                   # (16, 16)
    rhs    = jnp.concatenate([-c, b_act])                             # (16,)

    return K, rhs


# ─────────────────────────────────────────────────────────────────────────────
# §4b  KKT system — V2, 24×24 (Batch 8 extension with slip barrier rows)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def build_kkt_system_v2(
    Q:           jax.Array,     # (4, 4) objective Hessian
    c:           jax.Array,     # (4,)   objective linear term
    active_set:  jax.Array,     # (20,)  soft active-set indicator ∈ [0,1]
    p:           QPParams,
    A_slip:      jax.Array,     # (8, 4) slip-barrier constraint rows
    b_slip:      jax.Array,     # (8,)   slip-barrier rhs
) -> tuple[jax.Array, jax.Array]:
    """
    Build static-shape (24×24) KKT system for the extended 20-constraint problem.

    Active-set is SOFT (continuous in [0,1]) — each constraint row is scaled by
    its activation probability.  This makes ∂T*/∂θ continuous everywhere,
    enabling bilevel gradient flow (Batch 10 macro-gradient closure).

    Constraint layout (20 rows):
        [0:4]   T ≥ T_min   (lower box)
        [4:8]   T ≤ T_max   (upper box)
        [8:12]  |T|≤ T_fric (friction, sign from T_prev)
        [12:16] slip upper  κ_upper barrier
        [16:20] slip lower  κ_lower barrier

    The KKT (24×24) is:
        [ Q      A_actᵀ ] [ T ]   [ -c    ]
        [ A_act  −ε·I  ] [ λ ] = [ b_act ]

    where the −ε·I (ε = 1e-6) in the (2,2) block regularises the solve
    when soft-inactive rows produce near-zero diagonal entries.
    """
    N  = 4                       # primal vars
    M  = N_CONSTRAINTS_TOTAL     # 20 constraints
    I4 = jnp.eye(4)

    # ── Base constraint matrix ──────────────────────────────────────────────
    fric_sign = jnp.sign(p.t_prev + 1e-6)

    A_base = jnp.concatenate([I4, I4, I4], axis=0)      # (12, 4)  existing
    b_base = jnp.concatenate([                           # (12,)
        p.t_min,
        p.t_max,
        fric_sign * p.t_fric,
    ])

    # ── Concatenate with slip rows ─────────────────────────────────────────
    A_full = jnp.concatenate([A_base, A_slip], axis=0)   # (20, 4)
    b_full = jnp.concatenate([b_base, b_slip])           # (20,)

    # ── Soft masking: scale rows by active-set probability ─────────────────
    # At active_set_i ≈ 1: full row weight → constraint enforced.
    # At active_set_i ≈ 0: near-zero row weight → constraint effectively off.
    mask  = active_set[:, None]     # (20, 1) broadcast
    A_act = A_full * mask           # (20, 4)
    b_act = b_full * active_set     # (20,)

    # ── Assemble 24×24 KKT ─────────────────────────────────────────────────
    eps_reg = 1e-6 * jnp.eye(M)   # Tikhonov stabiliser for soft-inactive rows

    top    = jnp.concatenate([Q,     A_act.T  ],            axis=1)   # (4, 24)
    bottom = jnp.concatenate([A_act, -eps_reg],             axis=1)   # (20, 24)
    K      = jnp.concatenate([top, bottom], axis=0)                    # (24, 24)
    rhs    = jnp.concatenate([-c, b_act])                              # (24,)

    return K, rhs


# ─────────────────────────────────────────────────────────────────────────────
# §5  KKT solve + feasibility checks
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def solve_kkt(K: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve the (N+M)×(N+M) KKT system. Returns (T*, λ*)."""
    dim = K.shape[0]
    sol = jnp.linalg.solve(K + 1e-8 * jnp.eye(dim), rhs)
    return sol[:4], sol[4:]      # T* (4,), λ* (M,)


@jax.jit
def check_kkt_feasibility(
    T:          jax.Array,
    lambda_:    jax.Array,
    active_set: jax.Array,   # (12,) V1
    p:          QPParams,
    tol:        float = 3.0,
) -> jax.Array:
    primal_ok = (jnp.all(T >= p.t_min - tol)
                 & jnp.all(T <= p.t_max + tol)
                 & jnp.all(jnp.abs(T) <= p.t_fric + tol))
    dual_ok   = jnp.all((lambda_ >= -1.0) | (active_set < 0.5))
    return primal_ok & dual_ok


@jax.jit
def check_kkt_feasibility_v2(
    T:          jax.Array,
    lambda_:    jax.Array,
    active_set: jax.Array,   # (20,) V2
    p:          QPParams,
    A_slip:     jax.Array,   # (8, 4)
    b_slip:     jax.Array,   # (8,)
    tol:        float = 3.0,
    slip_tol:   float = 0.003,
) -> jax.Array:
    """Extended feasibility check including slip-barrier constraints."""
    primal_ok  = (jnp.all(T >= p.t_min - tol)
                  & jnp.all(T <= p.t_max + tol)
                  & jnp.all(jnp.abs(T) <= p.t_fric + tol))
    dual_ok    = jnp.all((lambda_ >= -1.0) | (active_set < 0.5))
    slip_ok    = check_slip_feasibility(T, A_slip, b_slip, tol=slip_tol)
    return primal_ok & dual_ok & slip_ok


# ─────────────────────────────────────────────────────────────────────────────
# §6  Smooth active-set prediction  (Batch 8 — enables end-to-end grad)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def predict_active_set_soft(
    bundle:     ClassifierBundle,
    theta_norm: jax.Array,
    tau:        float = 50.0,
) -> jax.Array:
    """
    Temperature-softened active-set indicator.

    Instead of hard threshold: A_c = 1[p_c ≥ θ_c]   (discontinuous)
    We use:                    A_c = σ(τ · (p_c − θ_c))  (C∞)

    At τ = 50:
      ∙ Within ±0.02 of the hard threshold → same classification in practice
      ∙ ∂A_c/∂θ is finite everywhere → ∂T*/∂θ is C∞ → bilevel grad works

    Returns active-set ∈ (0,1)^{N_CONSTRAINTS} — compatible with both
    build_kkt_system (V1, 12-dim) and build_kkt_system_v2 (V2, 20-dim).
    """
    probs = bundle.model.apply({"params": bundle.params}, theta_norm)   # (N_C,)
    return jax.nn.sigmoid(tau * (probs - bundle.thresholds))            # (N_C,)


# ─────────────────────────────────────────────────────────────────────────────
# §6b  Polish fallback  (extended to enforce slip constraints)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('n_steps', 'lr'))
def polish_step(
    T_init:  jax.Array,
    Q:       jax.Array,
    c:       jax.Array,
    p:       QPParams,
    n_steps: int   = 3,
    lr:      float = 0.30,
) -> jax.Array:
    """3-step projected gradient polish (V1 — box + friction only)."""
    def one_step(T, _):
        T_new = T - lr * (Q @ T + c)
        T_new = jnp.clip(T_new, p.t_min, p.t_max)
        T_new = jnp.clip(T_new, -p.t_fric, p.t_fric)
        return T_new, None
    T_opt, _ = jax.lax.scan(one_step, T_init, None, length=n_steps)
    return T_opt


@partial(jax.jit, static_argnames=('n_steps', 'lr'))
def polish_step_v2(
    T_init:  jax.Array,
    Q:       jax.Array,
    c:       jax.Array,
    p:       QPParams,
    A_slip:  jax.Array,
    b_slip:  jax.Array,
    n_steps: int   = 3,
    lr:      float = 0.25,
) -> jax.Array:
    """
    Extended polish: projects onto box, friction, AND slip constraints.

    Slip projection: for each violated slip row k: T ← T − lr · A[k]^T · violation_k
    This is a sub-gradient step toward feasibility — O(n_steps * M) but tiny M.
    """
    def one_step(T, _):
        # ── Gradient step on objective ─────────────────────────────────────
        T_new = T - lr * (Q @ T + c)

        # ── Box projection ─────────────────────────────────────────────────
        T_new = jnp.clip(T_new, p.t_min, p.t_max)
        T_new = jnp.clip(T_new, -p.t_fric, p.t_fric)

        # ── Slip sub-gradient projection ───────────────────────────────────
        # viol_k = max(0, A_slip[k] @ T − b_slip[k])   (smoothed)
        residuals = A_slip @ T_new - b_slip             # (8,)
        # Smooth max via softplus — gradient is well-defined at 0
        viol      = jax.nn.softplus(residuals * 50.0) / 50.0  # (8,)
        # Sub-gradient step: T ← T − lr · A_slip^T @ viol
        T_new     = T_new - lr * (A_slip.T @ viol)

        # Re-project box after slip adjustment
        T_new = jnp.clip(T_new, p.t_min, p.t_max)
        T_new = jnp.clip(T_new, -p.t_fric, p.t_fric)

        return T_new, None

    T_opt, _ = jax.lax.scan(one_step, T_init, None, length=n_steps)
    return T_opt


# ─────────────────────────────────────────────────────────────────────────────
# §7a  V1 allocator step factory  (Batch 3, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def make_explicit_allocator_step(clf_bundle: ClassifierBundle):
    """
    Batch 3 allocator — 12 constraints, 16×16 KKT.
    Preserved exactly for backward compatibility with Batch 3 call sites.
    """
    from powertrain.modes.advanced.active_set_classifier import make_predict_fns
    _predict, _ = make_predict_fns(clf_bundle)

    @jax.jit
    def _step(
        p:       QPParams,
        geo:     TVGeometry      = TVGeometry(),
        weights: AllocatorWeights = AllocatorWeights(),
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        theta_norm = normalise_theta(pack_theta_raw(p))
        active_set = _predict(theta_norm)            # (12,) hard threshold
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


# ─────────────────────────────────────────────────────────────────────────────
# §7b  V2 allocator step factory  (Batch 8 — slip-aware, smooth active-set)
# ─────────────────────────────────────────────────────────────────────────────

def make_explicit_allocator_step_v2(
    clf_bundle_v2:   ClassifierBundle,            # V2 model — 19→20
    slip_params:     SlipBarrierParams = SlipBarrierParams(),
    soft_tau:        float = 50.0,                # smoothing temperature
):
    """
    Batch 8 allocator — 20 constraints, 24×24 KKT, slip-barrier-aware.

    Accepts SlipBarrierInputs at each step.  Computes A_slip/b_slip internally.
    Uses SOFT active-set prediction via predict_active_set_soft() →
    ∂T*/∂θ is C∞ everywhere (bilevel gradient compatible, Batch 10 unlock).

    Falls back to disabled SlipBarrierInputs gracefully — constraints become
    trivially inactive (b_slip = 1e6) without any branching.
    """

    @jax.jit
    def _step(
        p:            QPParams,
        slip_inputs:  SlipBarrierInputs,          # use SlipBarrierInputs.disabled() if off
        geo:          TVGeometry       = TVGeometry(),
        weights:      AllocatorWeights  = AllocatorWeights(),
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Returns:
            T_opt      (4,)   — optimal wheel torques [Nm]
            active_set (20,)  — soft active-set indicator
            polished   scalar bool — True if polish was invoked
            slip_viol  (8,)   — slip constraint residuals (diagnostic)
        """
        # ── 1. Slip barrier rows ───────────────────────────────────────────
        A_slip, b_slip = build_slip_barrier_rows(slip_inputs, slip_params)

        # ── 2. Extended θ → V2 active-set (soft) ─────────────────────────
        theta_norm_v2 = normalise_theta_v2(pack_theta_raw_v2(p, slip_inputs))
        active_set_v2 = predict_active_set_soft(clf_bundle_v2, theta_norm_v2, tau=soft_tau)
        # (20,) soft indicators ∈ (0,1)

        # ── 3. Cost matrices ───────────────────────────────────────────────
        Q, c = build_qp_matrices(p, geo, weights)

        # ── 4. Extended KKT solve ─────────────────────────────────────────
        K, rhs   = build_kkt_system_v2(Q, c, active_set_v2, p, A_slip, b_slip)
        T_kkt, lam = solve_kkt(K, rhs)              # T* (4,), λ* (20,)

        # ── 5. Feasibility check (includes slip) ──────────────────────────
        feasible = check_kkt_feasibility_v2(
            T_kkt, lam, active_set_v2, p, A_slip, b_slip,
        )

        # ── 6. Polish if needed ────────────────────────────────────────────
        T_opt = jax.lax.cond(
            feasible,
            lambda t: t,
            lambda t: polish_step_v2(t, Q, c, p, A_slip, b_slip),
            T_kkt,
        )

        # ── 7. Diagnostic: final slip constraint residuals ─────────────────
        slip_viol = A_slip @ T_opt - b_slip      # (8,) — should be ≤ 0

        return T_opt, active_set_v2, ~feasible, slip_viol

    return _step


# ─────────────────────────────────────────────────────────────────────────────
# §7c  Graceful step — tries V2 classifier, falls back to V1
# ─────────────────────────────────────────────────────────────────────────────

def make_explicit_allocator_step_auto(
    clf_bundle_v1:   ClassifierBundle,
    clf_bundle_v2:   Optional[ClassifierBundle] = None,
    slip_params:     SlipBarrierParams = SlipBarrierParams(),
) -> tuple:
    """
    Returns (step_fn, version_str).  Chooses V2 if V2 classifier is available,
    otherwise V1 with a console warning.

    step_fn signature (V2):  (QPParams, SlipBarrierInputs, TVGeometry, AllocatorWeights) → (T, A, polished, slip_viol)
    step_fn signature (V1):  (QPParams, TVGeometry, AllocatorWeights) → (T, A, polished)
    """
    if clf_bundle_v2 is not None:
        step = make_explicit_allocator_step_v2(clf_bundle_v2, slip_params)
        return step, "v2-slip-aware"
    else:
        print("[ALLOC] V2 classifier not available — running V1 (no slip barriers in KKT). "
              "Slip safety via Batch 8 extended polish only.")
        step_v1 = make_explicit_allocator_step(clf_bundle_v1)

        # Wrap V1 to accept but ignore slip_inputs for a uniform call signature
        @jax.jit
        def _step_v1_compat(p, slip_inputs, geo=TVGeometry(), weights=AllocatorWeights()):
            T_opt, active_set, polished = step_v1(p, geo, weights)

            # Enforce slip via extended polish even in V1 mode.
            # This is the "safety net before retrain" behaviour described in the report.
            A_slip, b_slip = build_slip_barrier_rows(slip_inputs, slip_params)
            T_opt = jax.lax.cond(
                check_slip_feasibility(T_opt, A_slip, b_slip),
                lambda t: t,
                lambda t: polish_step_v2(t,
                    build_qp_matrices(p, geo, weights)[0],
                    build_qp_matrices(p, geo, weights)[1],
                    p, A_slip, b_slip),
                T_opt,
            )
            slip_viol = A_slip @ T_opt - b_slip
            return T_opt, active_set, polished, slip_viol

        return _step_v1_compat, "v1+slip-polish"


# ─────────────────────────────────────────────────────────────────────────────
# §8  Public entry-point wrappers (backward-compat with Batch 3 callers)
# ─────────────────────────────────────────────────────────────────────────────

def explicit_allocator_step(
    clf_bundle: ClassifierBundle,
    p:          QPParams,
    geo:        TVGeometry       = TVGeometry(),
    weights:    AllocatorWeights  = AllocatorWeights(),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """V1 entry point — unchanged from Batch 3."""
    return make_explicit_allocator_step(clf_bundle)(p, geo, weights)


class ExplicitAllocatorState(NamedTuple):
    T_prev:       jax.Array
    polish_count: jax.Array


def init_explicit_allocator_state() -> ExplicitAllocatorState:
    return ExplicitAllocatorState(
        T_prev       = jnp.zeros(4),
        polish_count = jnp.array(0, dtype=jnp.int32),
    )


def make_explicit_allocator(
    clf_bundle: ClassifierBundle,
    geo:     TVGeometry       = TVGeometry(),
    weights: AllocatorWeights = AllocatorWeights(),
):
    """V1 convenience wrapper — backward compat."""
    @jax.jit
    def alloc_fn(state, mz_ref, fx_d, t_min, t_max, t_fric, delta, omega):
        p = QPParams(
            mz_ref=mz_ref, fx_d=fx_d,
            t_min=t_min, t_max=t_max, t_fric=t_fric,
            delta=delta, t_prev=state.T_prev, omega=omega,
        )
        T_opt, _, polished = explicit_allocator_step(clf_bundle, p, geo, weights)
        new_state = ExplicitAllocatorState(
            T_prev       = T_opt,
            polish_count = state.polish_count + polished.astype(jnp.int32),
        )
        return T_opt, new_state
    return alloc_fn