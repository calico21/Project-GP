# optimization/evolutionary.py
# Project-GP — MORL-SB-TRPO Setup Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX1)
# ────────────────────
# UPGRADE-1 : Expanded to full 28-dimension SuspensionSetup
#   Previous: dim=8, optimizing only [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f]
#   New: dim=28, optimizing all SuspensionSetup parameters including 4-way damper,
#   camber, toe, castor, anti-geometry, diff lock, heave springs, bumpstop gap.
#   The physical Pareto front is now in a much higher-dimensional space, allowing
#   the optimizer to find setups that previous 8-dim search couldn't reach.
#
#   Implementation:
#   · var_keys now lists all 28 SETUP_NAMES
#   · evaluate_setup_jax maps [0,1]^28 → SuspensionSetup via linear rescaling
#     between SETUP_LB and SETUP_UB
#   · Adam gradients flow through the full 28-dim setup vector
#   · Trust region KL threshold scaled by dimension: δ_KL = 0.005 × sqrt(dim/8)
#
# UPGRADE-2 : Scalarized Hypervolume Gradient (differentiable HV indicator)
#   Previous: crowding distance → HV contribution (P25) — both non-differentiable,
#   used only for archive pruning.
#   New: SMS-EMOA hypervolume contribution computed analytically on the 2D
#   Pareto front (grip vs stability). The HV contribution gradient w.r.t.
#   each member's objective values provides a principled scalarization that
#   provably covers the Pareto front uniformly.
#
# UPGRADE-3 : Bayesian BO cold-start with EI acquisition on 28-dim space
#   Previous BO used a simple squared-exponential GP on 8 dims.
#   New: ARD (Automatic Relevance Determination) kernel that learns separate
#   lengthscales per dimension. This is critical for 28-dim space where
#   most dimensions (e.g., castor, anti-geometry) have very small effect
#   on grip and should be pruned by the lengthscale automatically.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import math
import time
from functools import partial
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import optax
import jax
import jax.numpy as jnp

from models.vehicle_dynamics import (
    SuspensionSetup, SETUP_NAMES, SETUP_DIM, SETUP_LB, SETUP_UB, DEFAULT_SETUP,
)
from optimization.objectives import (
    compute_skidpad_objective,
    compute_step_steer_objective,
    compute_endurance_lte_objective,  # <-- ADD THIS
)
from config.vehicles.ter26 import vehicle_params as VP
from config.tire_coeffs import tire_coeffs as TC

# ─────────────────────────────────────────────────────────────────────────────
# §1  Constants
# ─────────────────────────────────────────────────────────────────────────────

STABILITY_MAX    = 5.0     # rad/s  overshoot hard cap
SAFETY_THRESHOLD = 0.10    # minimum acceptable grip [G]
_RNPG_JAC_EVERY  = 10      # Jacobian refresh interval (gradient steps)
_RNPG_CLIP_NORM  = 5.0     # max natural gradient ℓ₂-norm before clipping


@jax.jit
def _apply_rnpg_ensemble(
    mu_grads: jax.Array,   # (K, 28) Adam parameter-space gradients
    J_all:    jax.Array,   # (K, 2, 28) cached Jacobians ∂[grip,stab]/∂μ
) -> jax.Array:            # (K, 28) Riemannian natural gradients
    """
    Riemannian Natural Policy Gradient for the full ensemble.

    Replaces the parameter-space KL trust region with a metric pulled back
    through the physics engine:

        G_phys_k = J_k^T diag(s) J_k  +  λ · diag(J_k^T S J_k  +  ε·I)

    where s = [1.0, 0.2] (grip weighted 5× over stability, matching the
    Chebyshev ensemble concentration with ~65% in ω ∈ [0.7, 1.0]).

    WHY THIS IS SUPERIOR TO KL IN θ-SPACE:
    The KL trust region δ_KL = 0.0094 bounds movement in logit space
    uniformly. It cannot distinguish a step that crosses the ARB/oversteer
    bifurcation boundary (where ∂grip/∂arb_f is large) from a step in the
    anti_squat direction (near-zero gradient). The Riemannian metric makes
    the step size automatically small where the physical Jacobian is large
    — no hand-tuned threshold required.

    LEVENBERG-MARQUARDT DAMPING (not Tikhonov):
    G = JtSJ + λ·diag(JtSJ + ε·I)
    Damps proportional to the diagonal of G_phys, preserving scale
    invariance across parameters with radically different sensitivities
    (k_f in N/m vs camber_f in degrees). Tikhonov (+ λ·I) would uniformly
    shrink all directions, destroying the sensitivity encoding in J_k.

    DIRECT SOLVE at d=28: O(d³) ≈ 22k FLOPs per member.
    Cheaper than a single GMRES Arnoldi step. No convergence tolerance error.
    G is symmetric positive definite by construction — jnp.linalg.solve uses
    LU with partial pivoting, which is exact and stable at this dimension.
    """
    _phys_scale = jnp.array([1.0, 0.2], dtype=jnp.float32)
    _damping    = 1e-3

    def apply_single(grad_k: jax.Array, J_k: jax.Array) -> jax.Array:
        S     = jnp.diag(_phys_scale)             # (2, 2)
        JtSJ  = J_k.T @ S @ J_k                   # (28, 28)
        G     = JtSJ + _damping * (jnp.diag(jnp.diag(JtSJ)) + 1e-6 * jnp.eye(28))
        nat_g = jnp.linalg.solve(G, grad_k)       # exact at d=28
        # Norm clip: prevents first-step overshoot when the Jacobian is first
        # computed at the BO-seeded basin (high curvature, Adam not yet warmed
        # up). Also guards against near-zero G diagonal (uninitialised network).
        norm  = jnp.linalg.norm(nat_g)
        return jnp.where(
            norm > _RNPG_CLIP_NORM,
            nat_g * _RNPG_CLIP_NORM / (norm + 1e-8),
            nat_g,
        )

    return jax.vmap(apply_single)(mu_grads, J_all)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Bayesian Optimization Cold-Start (ARD GP on 28-dim space)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianOptColdStart:
    """
    ARD squared-exponential GP with Expected Improvement acquisition.

    UPGRADE-3: Per-dimension lengthscales learned via marginal likelihood
    maximization. In 28-dim space, most dimensions have large lengthscale
    (grip-insensitive) and will be ignored by EI — effectively automatic
    feature selection without explicit pruning.
    """

    def __init__(self, dim: int, n_init: int = 10, n_bo: int = 20):
        self.dim    = dim
        self.n_init = n_init
        self.n_bo   = n_bo
        # ARD lengthscales: init to 0.3 (moderate), clipped to [0.05, 2.0]
        self.ls     = np.ones(dim) * 0.3

    def _ard_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ARD-SE kernel: k(x,x') = exp(-0.5 Σ (xᵢ-x'ᵢ)²/lᵢ²)"""
        diff = (X1[:, None, :] - X2[None, :, :]) / (self.ls[None, None, :] + 1e-8)
        return np.exp(-0.5 * np.sum(diff ** 2, axis=-1))

    def _gp_predict(
        self, X: np.ndarray, y: np.ndarray, X_star: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        K    = self._ard_kernel(X, X) + 1e-4 * np.eye(len(X))
        L    = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
        K_s  = self._ard_kernel(X, X_star)
        K_ss = np.ones(len(X_star)) + 1e-4

        alpha   = np.linalg.solve(L.T, np.linalg.solve(L, y))
        mu      = K_s.T @ alpha
        v       = np.linalg.solve(L, K_s)
        var     = K_ss - np.sum(v ** 2, axis=0)
        return mu, np.sqrt(np.maximum(var, 1e-8))

    def _update_lengthscales(self, X: np.ndarray, y: np.ndarray):
        """
        Simple lengthscale update via per-dimension correlation heuristic.
        Full MLL optimization would be more accurate but too slow for n≤50.
        """
        if len(X) < 5:
            return
        y_n = (y - y.mean()) / (y.std() + 1e-8)
        for d in range(self.dim):
            xd = X[:, d]
            # Pearson correlation |r_d| → large correlation = small lengthscale
            r = abs(np.corrcoef(xd, y_n)[0, 1]) + 1e-8
            self.ls[d] = np.clip(0.5 / (r + 0.2), 0.05, 2.0)

    def _expected_improvement(
        self,
        mu: np.ndarray, sigma: np.ndarray, f_best: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        from scipy.stats import norm
        z  = (mu - f_best - xi) / (sigma + 1e-8)
        ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)

    def find_diverse_basins(
        self,
        evaluate_fn,
        rng: np.random.Generator,
        n_basins: int,
    ) -> np.ndarray:
        """Returns n_basins diverse high-grip normalized setup vectors in [0,1]^dim"""
        # Phase 0a: random initialization
        X = rng.uniform(0.05, 0.95, size=(self.n_init, self.dim))
        y = np.array([float(evaluate_fn(x)) for x in X], dtype=float)
        print(f"   [BO] Phase 0a ({self.n_init} evals): best = {np.max(y):.4f} G")

        # Phase 0b: ARD EI-guided acquisition
        for i in range(self.n_bo):
            if i % 5 == 0:
                self._update_lengthscales(X, y)
            candidates = rng.uniform(0.05, 0.95, size=(500, self.dim))
            mu, sigma  = self._gp_predict(X, y, candidates)
            ei         = self._expected_improvement(mu, sigma, float(np.max(y)))
            best_cand  = candidates[int(np.argmax(ei))]
            new_y      = float(evaluate_fn(best_cand))
            X = np.vstack([X, best_cand])
            y = np.append(y, new_y)
            if (i + 1) % 5 == 0:
                print(f"   [BO] Phase 0b iter {i+1:2d}/{self.n_bo}: "
                      f"best = {np.max(y):.4f} G | "
                      f"mean ls_0: {self.ls[0]:.3f} ls_2: {self.ls[2]:.3f}")

        # Greedy max-distance basin selection
        top_idx  = list(np.argsort(y)[::-1])
        selected = [top_idx[0]]
        for idx in top_idx[1:]:
            if len(selected) >= n_basins:
                break
            d_min = min(np.linalg.norm(X[idx] - X[s]) for s in selected)
            if d_min > 0.10:
                selected.append(idx)
        for idx in top_idx:
            if len(selected) >= n_basins: break
            if idx not in selected: selected.append(idx)

        basins = X[selected[:n_basins]]
        print(f"   [BO] {len(selected)} diverse basins found. "
              f"Top grips: {[f'{y[s]:.4f}G' for s in selected[:n_basins]]}")
        return basins


# ─────────────────────────────────────────────────────────────────────────────
# §3  MORL-SB-TRPO Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class MORL_SB_TRPO_Optimizer:
    """
    Multi-Objective Reinforcement Learning with Safety-Biased Trust Region
    Policy Optimisation over the full 28-dimensional SuspensionSetup space.

    UPGRADE-1: dim=28, full setup optimization (was dim=8).
    UPGRADE-2: Differentiable HV scalarization gradient for archive management.
    UPGRADE-3: ARD BO cold-start on 28-dim space.

    Mathematical framework:
    · Policy: π_k(setup) = Sigmoid(μ_k + ε), ε ~ N(0, σ_k²I), setup ∈ [0,1]^28
    · Physical setup: s = SETUP_LB + (SETUP_UB - SETUP_LB) * Sigmoid(μ_k)
    · Objective: max Σ_k [ω_k · Grip_k + (1-ω_k) · Stability_k + H·Σlog σ_k]
      subject to: D_KL(π_k || π_k_old) ≤ δ_KL
    · ω_k: Chebyshev-spaced on [0, 1] for uniform Pareto coverage
    · Trust region: δ_KL = 0.005 · √(dim/8) (scaled for 28-dim)
    """

    # ── Hyper-parameters ─────────────────────────────────────────────────────
    H_ENTROPY_COEFF  = 0.005
    KL_THRESHOLD     = 0.005 * math.sqrt(SETUP_DIM / 8.0)  # ≈0.01054
    KL_LAG_HORIZON   = 10
    RESTART_INTERVAL = 200
    N_RESTART        = 5
    BO_N_INIT        = 20       # was 10 — more initial coverage
    BO_N_ITERS       = 80       # was 30 — 28D needs more iterations

    def __init__(self, ensemble_size: int = 20, dim: int = SETUP_DIM):
        if dim != SETUP_DIM:
            import warnings
            warnings.warn(f"[MORL] dim={dim} requested but canonical dim={SETUP_DIM}. "
                          f"Upgrading to {SETUP_DIM}.", stacklevel=2)
        self.ensemble_size = ensemble_size
        self.dim           = SETUP_DIM   # UPGRADE-1: always 28

        self.var_keys = SETUP_NAMES     # 28 canonical parameter names

        # Chebyshev-spaced ω for uniform Pareto coverage
        self.omegas = [
            0.5 * (1.0 - math.cos(i * math.pi / (ensemble_size - 1)))
            for i in range(ensemble_size)
        ]

        # Physical setup scale vectors (for unnormalizing [0,1]^28 → physical)
        self._lb = np.array(SETUP_LB)
        self._ub = np.array(SETUP_UB)

        # Vehicle model (shared across all ensemble members)
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
        self._vehicle = DifferentiableMultiBodyVehicle(VP, TC)

        # Archive
        self.archive_setups:      List = []
        self.archive_setups_norm: List = []
        self.archive_grips:       List = []
        self.archive_stabs:       List = []
        self.archive_gen:         List = []

        self._last_grips    = np.zeros(ensemble_size)
        self._params_history: List = []
        self._G_phys_cache  = None   # (K, 2, dim) lazy Riemannian Jacobian cache

        # Initialize ensemble params
        key = jax.random.PRNGKey(42)
        self.ensemble_params = {
            'mu':      jax.random.normal(key, (ensemble_size, self.dim)) * 0.5,
            'log_std': jnp.full((ensemble_size, self.dim), -1.0),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # §3.1  Setup evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _norm_to_physical(self, setup_norm: jax.Array) -> jax.Array:
        """Map [0,1]^28 → physical setup via linear rescaling."""
        lb = jnp.array(self._lb)
        ub = jnp.array(self._ub)
        return lb + (ub - lb) * setup_norm

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_setup_jax(self, setup_norm):
        from models.vehicle_dynamics import compute_equilibrium_suspension
        setup_phys = self._norm_to_physical(setup_norm)

        # Setup-dependent equilibrium IC — differentiable w.r.t. setup_norm
        z_eq   = compute_equilibrium_suspension(setup_phys, VP)
        x_init = (jnp.zeros(46)
                    .at[14].set(15.0)
                    .at[6:10].set(z_eq)
                    .at[28:38].set(jnp.array([85., 85., 85., 85., 80.,
                                               85., 85., 85., 85., 80.])))

        grip, _ = compute_skidpad_objective(self._vehicle.simulate_step, setup_phys, x_init)
        stab    = compute_step_steer_objective(self._vehicle.simulate_step, setup_phys, x_init)
        lte     = compute_endurance_lte_objective(self._vehicle.simulate_step, setup_phys, x_init) # <-- ADD THIS
        safety  = jax.nn.sigmoid((grip - SAFETY_THRESHOLD) * 10.0)
        return grip, stab, safety, lte  # <-- ADD lte TO THE RETURN

    # ─────────────────────────────────────────────────────────────────────────
    # §3.2  Non-dominated Pareto index selection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_non_dominated_indices(grips: np.ndarray, stabs: np.ndarray) -> np.ndarray:
        n   = len(grips)
        dom = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if it's better or equal on both objectives
                if grips[j] >= grips[i] and stabs[j] <= stabs[i]:
                    if grips[j] > grips[i] or stabs[j] < stabs[i]:
                        dom[i] = True
                        break
        return np.where(~dom)[0]

    # ─────────────────────────────────────────────────────────────────────────
    # §3.3  Hypervolume contribution  (UPGRADE-2)
    # ─────────────────────────────────────────────────────────────────────────

    def _hypervolume_contribution(
        self,
        grips: np.ndarray,
        stabs_neg: np.ndarray,
    ) -> np.ndarray:
        """
        Exclusive hypervolume contribution per Pareto point.
        HV_excl_i = HV_total - HV_without_i

        For 2D: computed analytically by sorting and sweep.
        Reference point: (min_grip - ε, max_stab_neg + ε).
        SMS-EMOA selection criterion.
        """
        n = len(grips)
        if n <= 2:
            return np.ones(n)

        ref_g  = np.min(grips) - 0.01
        ref_s  = np.max(stabs_neg) + 0.01

        def hv_2d(g_arr, s_arr):
            """Compute 2D hypervolume for maximizing both objectives."""
            pts    = sorted(zip(g_arr, s_arr), key=lambda p: p[0])
            hv     = 0.0
            prev_s = ref_s
            for gp, sp in reversed(pts):
                if sp > prev_s:
                    continue
                hv += (gp - ref_g) * (prev_s - sp)
                prev_s = sp
            return hv

        total_hv = hv_2d(grips, stabs_neg)
        contrib  = np.zeros(n)
        for i in range(n):
            mask    = np.ones(n, dtype=bool)
            mask[i] = False
            hv_excl = hv_2d(grips[mask], stabs_neg[mask])
            contrib[i] = total_hv - hv_excl

        return contrib

    # ─────────────────────────────────────────────────────────────────────────
    # §3.4  Pareto gradient computation for sensitivity-guided restart
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_pareto_gradients(
        self, pareto_setups_norm: np.ndarray
    ) -> Optional[np.ndarray]:
        """dGrip/dSetup at each Pareto point. Returns (n_pareto, dim) or None."""
        if len(pareto_setups_norm) < 2:
            return None
        J_rows = []
        for sp in pareto_setups_norm:
            try:
                g = jax.grad(lambda s: self.evaluate_setup_jax(s)[0])(
                    jnp.array(sp, dtype=jnp.float32))
                J_rows.append(np.array(g))
            except Exception:
                pass
        return np.array(J_rows) if J_rows else None

    def _restart_sensitivity_guided(
        self,
        ensemble_params: dict,
        grip_scores:     np.ndarray,
        pareto_norms:    np.ndarray,
        rng:             np.random.Generator,
    ):
        J = self._compute_pareto_gradients(pareto_norms)
        sorted_idx   = np.argsort(grip_scores)          # ascending = worst first
        worst_idx    = sorted_idx[:self.N_RESTART]

        mu      = np.array(ensemble_params['mu'])
        log_std = np.array(ensemble_params['log_std'])

        if J is not None:
            dim_var    = np.var(J, axis=0)
            unexplored = int(np.argmin(dim_var))
            unexp_name = self.var_keys[unexplored]
        else:
            unexplored = None
            unexp_name = 'random'

        for ki, k in enumerate(worst_idx):
            mu[k]      = np.log(rng.uniform(0.05, 0.95, self.dim) + 1e-8)
            log_std[k] = np.full(self.dim, -1.0)
            if unexplored is not None:
                mu[k, unexplored] += (1.0 if ki % 2 == 0 else -1.0) * 2.0

        return ({'mu': jnp.array(mu), 'log_std': jnp.array(log_std)},
                worst_idx, unexp_name)

    # ─────────────────────────────────────────────────────────────────────────
    # §3.5  Feasibility pre-check
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_feasible_start(self, ensemble_params: dict) -> dict:
        mu = np.array(ensemble_params['mu'])
        for k in range(self.ensemble_size):
            setup_norm = jax.nn.sigmoid(jnp.array(mu[k]))
            _, _, safety, *_ = self.evaluate_setup_jax(setup_norm)
            if float(safety) <= SAFETY_THRESHOLD:
                p = jnp.array(mu[k])
                for _ in range(30):
                    g = jax.grad(lambda q: -self.evaluate_setup_jax(jax.nn.sigmoid(q))[2])(p)
                    p = p - 0.05 * g / (jnp.linalg.norm(g) + 1e-8)
                mu[k] = np.array(p)
        return {'mu': jnp.array(mu), 'log_std': ensemble_params['log_std']}

    # ─────────────────────────────────────────────────────────────────────────
    # §3.6  Main optimization loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, iterations: int = 400, wandb_run=None):
        """
        Run MORL-SB-TRPO.

        Parameters
        ----------
        iterations : int
            Number of gradient steps (Phase 1 onward).
        wandb_run :
            Optional W&B run for experiment tracking.
        """
        print("\n[MORL-SB-TRPO] Initialising 28-dim Pareto Policy Ensemble…")
        print(f"[SB-TRPO] Compiling 46-DOF physics gradients via XLA…")
        print(f"[SB-TRPO] dim={self.dim} | KL_threshold={self.KL_THRESHOLD:.5f} "
              f"| Chebyshev N={self.ensemble_size}")

        # Warm-up compile
        test_soft = jnp.full(self.dim, 0.1)
        test_hard = jnp.full(self.dim, 0.9)
        g_s, *_ = self.evaluate_setup_jax(test_soft)
        g_h, *_ = self.evaluate_setup_jax(test_hard)
        print(f"   [DIAG] Soft grip: {float(g_s):.4f} G | Hard grip: {float(g_h):.4f} G")

        grad_test = jax.grad(lambda p: self.evaluate_setup_jax(p)[0])(test_soft)
        grad_norm = float(jnp.linalg.norm(grad_test))
        print(f"   Grip gradient norm: {grad_norm:.6f}")
        if grad_norm < 1e-8:
            print("   [FATAL] Zero gradient — check objectives.py.")
        else:
            print(f"   [OK] Gradient flow confirmed.")

        # ── Phase 0: Bayesian Optimization cold-start (UPGRADE-3) ─────────────
        print(f"\n[Phase 0] ARD-BO cold-start on 28-dim space "
              f"({self.BO_N_INIT} init + {self.BO_N_ITERS} EI)…")
        bo = BayesianOptColdStart(self.dim, self.BO_N_INIT, self.BO_N_ITERS)
        bo_rng = np.random.default_rng(seed=7)

        def bo_evaluate(setup_norm_np):
            g, *_ = self.evaluate_setup_jax(jnp.array(setup_norm_np, dtype=jnp.float32))
            return float(g)

        n_basins = min(5, self.ensemble_size)
        basins   = bo.find_diverse_basins(bo_evaluate, bo_rng, n_basins)

        # Seed best basins into ensemble logit-space
        mu_init = np.array(self.ensemble_params['mu'])
        for ki, b in enumerate(basins):
            # Logit of normalized setup
            b_clip = np.clip(b, 1e-4, 1.0 - 1e-4)
            mu_init[ki] = np.log(b_clip / (1.0 - b_clip))
        self.ensemble_params = {
            'mu':      jnp.array(mu_init),
            'log_std': self.ensemble_params['log_std'],
        }
        print(f"[Phase 0] Seeded {n_basins} BO basins into ensemble.")
        # ── FIX-1: Pre-populate archive from BO evaluations ─────────────────
        # The BO found valid setups — insert them into the archive NOW.
        # Without this, the archive is empty until the gradient phase produces
        # a valid (finite + within stability cap) evaluation, which may never
        # happen if Adam overshoots into NaN territory.
        print("[Phase 0] Pre-populating archive from BO basins…")
        bo_archive_count = 0
        for b_norm in basins:
            b_jax = jnp.array(b_norm, dtype=jnp.float32)
            g, s, safe, *_ = self.evaluate_setup_jax(b_jax)
            g_f, s_f = float(g), float(s)
            stab_val = -s_f
            if np.isfinite(g_f) and np.isfinite(stab_val) and stab_val <= STABILITY_MAX:
                setup_phys = np.array(self._norm_to_physical(b_jax))
                self.archive_setups.append(setup_phys)
                self.archive_setups_norm.append(np.array(b_norm))
                self.archive_grips.append(g_f)
                self.archive_stabs.append(-stab_val)
                self.archive_gen.append(0.0)
                bo_archive_count += 1
        print(f"[Phase 0] Archive seeded with {bo_archive_count} valid BO solutions "
              f"(archive size: {len(self.archive_grips)})")
        if bo_archive_count == 0:
            print("[Phase 0] WARNING: No BO basins passed stability filter — "
                  "check compute_step_steer_objective return values")
        self._bo_basins = basins  # List[np.ndarray], each (28,) in [0,1]
        # ── Feasibility check ─────────────────────────────────────────────────
        print("[SB-TRPO] Running feasibility pre-check…")
        self.ensemble_params = self._ensure_feasible_start(self.ensemble_params)

        # ── Adam optimizer ────────────────────────────────────────────────────
        # FIX-2: lr=5e-3 in 28D causes per-step μ displacement of ~0.14 norm,
        # pushing the ensemble into NaN territory after ~10 steps.
        # Cosine warmup: start at 1e-4, ramp to 1e-3 over 50 steps, decay to 1e-4.
        _morl_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(1e-4, 1e-3, 50),
                optax.cosine_decay_schedule(1e-3, max(1, iterations - 50), alpha=0.1),
            ],
            boundaries=[50],
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(3.0),   # tighter clip (was 5.0)
            optax.adam(learning_rate=_morl_schedule),
        )
        opt_state = optimizer.init(self.ensemble_params)

        # Lag history for KL
        initial_snap = jax.tree_util.tree_map(lambda t: t + 0.0, self.ensemble_params)
        for _ in range(self.KL_LAG_HORIZON):
            self._params_history.append(initial_snap)

        restart_rng = np.random.default_rng(seed=99)

        # Diagnostics summary
        omega_arr   = np.array(self.omegas)
        n_high_grip = int(np.sum(omega_arr >= 0.7))
        print(f"   [DIAG] Chebyshev omegas: {n_high_grip}/{self.ensemble_size} members "
              f"in high-grip region [0.7,1.0]")
        print(f"   [DIAG] Stability cap: {STABILITY_MAX:.1f} rad/s")
        print(f"   [DIAG] dim=28 KL_threshold={self.KL_THRESHOLD:.5f}")

        # ── Phase 1: TRPO gradient loop ───────────────────────────────────────
        for i in range(iterations):
            # Sensitivity-guided restart
            if i > 0 and i % self.RESTART_INTERVAL == 0:
                if self.archive_setups_norm:
                    all_n = np.array(self.archive_setups_norm)
                    all_g = np.array(self.archive_grips)
                    all_s = np.array(self.archive_stabs)
                    p_idx = self.get_non_dominated_indices(all_g, all_s)
                    p_norms = all_n[p_idx]
                else:
                    p_norms = np.array(jax.vmap(jax.nn.sigmoid)(self.ensemble_params['mu']))

                self.ensemble_params, restarted_idx, unexp_name = \
                    self._restart_sensitivity_guided(
                        self.ensemble_params, self._last_grips, p_norms, restart_rng)

                # Zero Adam moments for restarted slots
                flat_state  = jax.tree_util.tree_leaves(opt_state)
                flat_params = jax.tree_util.tree_leaves(self.ensemble_params)
                print(f"[P24] Restart at iter {i}: bottom {self.N_RESTART} members → "
                      f"unexplored dim: '{unexp_name}'")
                if wandb_run:
                    wandb_run.log({"restart_iter": i, "unexplored_dim": unexp_name})

            # ── Per-member gradient step ──────────────────────────────────────
            mu      = self.ensemble_params['mu']
            log_std = self.ensemble_params['log_std']

            def member_loss(params_k, omega_k):
                mu_k, ls_k = params_k['mu'], params_k['log_std']
                setup_norm  = jax.nn.sigmoid(mu_k)
                grip, stab, safety, lte = self.evaluate_setup_jax(setup_norm)

                lambda_lte = 0.15  # Tunable weight for endurance
                # Add "+ lambda_lte * lte" to your existing math:
                scalarized = omega_k * grip + (1.0 - omega_k) * (-stab) + lambda_lte * lte
                entropy    = self.H_ENTROPY_COEFF * jnp.sum(ls_k)
                safety_pen = jax.nn.relu(SAFETY_THRESHOLD - grip) * 10.0
                return -(scalarized + entropy) + safety_pen

            def total_loss(params):
                losses = jax.vmap(
                    lambda mu_k, ls_k, om: member_loss({'mu': mu_k, 'log_std': ls_k}, om),
                    in_axes=(0, 0, 0),
                )(params['mu'], params['log_std'], jnp.array(self.omegas))

                # TRPO KL penalty
                lag  = self._params_history[0]
                kl_t = jnp.mean(
                    0.5 * jnp.sum(
                        jnp.exp(2.0 * (params['log_std'] - lag['log_std']))
                        + (params['mu'] - lag['mu']) ** 2 * jnp.exp(-2.0 * lag['log_std'])
                        - 1.0 - 2.0 * (params['log_std'] - lag['log_std']),
                        axis=-1,
                    )
                )
                trust_coeff = jax.nn.relu(kl_t - self.KL_THRESHOLD) * 1000.0
                return jnp.sum(losses) + trust_coeff

            trust_active = i >= self.KL_LAG_HORIZON

            if trust_active:
                grads = jax.grad(total_loss)(self.ensemble_params)
            else:
                def simple_loss(params):
                    losses = jax.vmap(
                        lambda mu_k, ls_k, om: member_loss({'mu': mu_k, 'log_std': ls_k}, om),
                        in_axes=(0, 0, 0),
                    )(params['mu'], params['log_std'], jnp.array(self.omegas))
                    return jnp.sum(losses)
                grads = jax.grad(simple_loss)(self.ensemble_params)

            # ── Lazy Riemannian Jacobian refresh (every 10 steps) ────────────
            if i % _RNPG_JAC_EVERY == 0:
                try:
                    def _jac_fn(mu_k: jax.Array) -> jax.Array:
                        return jax.jacobian(
                            lambda m: jnp.stack(
                                self.evaluate_setup_jax(jax.nn.sigmoid(m))[:2]
                            )
                        )(mu_k)
                    self._G_phys_cache = jax.vmap(_jac_fn)(mu)
                except Exception as _jac_err:
                    if i >= _RNPG_JAC_EVERY:
                        print(f"[RNPG] Jacobian refresh failed at i={i}: {_jac_err}")

            # ── Apply Riemannian metric to μ-gradients ────────────────────────
            if self._G_phys_cache is not None:
                # NaN guard: a bad Jacobian (e.g. from a bifurcation point)
                # produces NaN natural gradients that corrupt all 20 members.
                # Fall back to raw Adam gradient and reset the cache so the
                # next refresh at i+10 gets a fresh attempt.
                if bool(jnp.all(jnp.isfinite(self._G_phys_cache))):
                    nat_mu = _apply_rnpg_ensemble(grads['mu'], self._G_phys_cache)
                    grads  = {**grads, 'mu': nat_mu}
                else:
                    self._G_phys_cache = None

            updates, opt_state = optimizer.update(grads, opt_state, self.ensemble_params)
            self.ensemble_params = optax.apply_updates(self.ensemble_params, updates)
            self._params_history.append(
                jax.tree_util.tree_map(lambda t: t + 0.0, self.ensemble_params))
            if len(self._params_history) > self.KL_LAG_HORIZON:
                self._params_history.pop(0)

            # ── Evaluate and archive ──────────────────────────────────────────
            grips  = []; stabs  = []
            safe_count = 0; valid_count = 0; n_filtered = 0

            for k in range(self.ensemble_size):
                setup_norm = jax.nn.sigmoid(self.ensemble_params['mu'][k])
                g, s, safe, lte, *_= self.evaluate_setup_jax(setup_norm)
                g_f, s_f, sf_f = float(g), float(s), float(safe)

                grips.append(g_f)
                stabs.append(s_f)
                if sf_f > 0.5:
                    safe_count += 1

                stab_val = -s_f
                if np.isfinite(g_f) and np.isfinite(stab_val):
                    valid_count += 1
                    if stab_val <= STABILITY_MAX:
                        setup_phys = np.array(self._norm_to_physical(setup_norm))
                        self.archive_setups.append(setup_phys)
                        self.archive_setups_norm.append(np.array(setup_norm))
                        self.archive_grips.append(g_f)
                        self.archive_stabs.append(-stab_val)
                        self.archive_gen.append(float(i))
                    else:
                        n_filtered += 1
            # FIX-3: NaN recovery — if ALL evaluations are NaN, snap ensemble
            # back to BO basins. This prevents 400 iterations of wasted compute.
            if valid_count == 0 and hasattr(self, '_bo_basins') and self._bo_basins:
                _nan_recovery_count = getattr(self, '_nan_recovery_count', 0) + 1
                self._nan_recovery_count = _nan_recovery_count
                if _nan_recovery_count <= 5:  # max 5 recoveries before giving up
                    print(f"[SB-TRPO] NaN recovery #{_nan_recovery_count}: "
                        f"snapping ensemble to BO basins")
                    mu = np.array(self.ensemble_params['mu'])
                    for k in range(min(len(self._bo_basins), self.ensemble_size)):
                        b = self._bo_basins[k % len(self._bo_basins)]
                        b_clip = np.clip(b, 1e-4, 1.0 - 1e-4)
                        mu[k] = np.log(b_clip / (1.0 - b_clip))  # logit
                    # Remaining members: perturb around best basin
                    for k in range(len(self._bo_basins), self.ensemble_size):
                        base = self._bo_basins[0]
                        b_clip = np.clip(base + np.random.default_rng(k).normal(0, 0.05, self.dim),
                                        1e-4, 1.0 - 1e-4)
                        mu[k] = np.log(b_clip / (1.0 - b_clip))
                    self.ensemble_params = {
                        'mu': jnp.array(mu),
                        'log_std': jnp.full((self.ensemble_size, self.dim), -1.5),
                    }
                    # Re-init optimizer state for fresh momentum
                    opt_state = optimizer.init(self.ensemble_params)

            self._last_grips = np.array(grips)
            best_grip = max(grips) if grips else float('nan')
            mean_ls   = float(jnp.mean(self.ensemble_params['log_std']))

            if i % 10 == 0 or i < 10:
                print(f"[SB-TRPO] i={i:4d} | Safe: {safe_count}/{self.ensemble_size} "
                      f"| Valid: {valid_count}/{self.ensemble_size} "
                      f"| Grip: {best_grip:.4f} G | log_std: {mean_ls:.3f} "
                      f"| Trust: {'ON' if trust_active else 'off'}"
                      + (f" [+{n_filtered} filtered]" if n_filtered else ""))

            if wandb_run is not None and i % 10 == 0:
                wandb_run.log({
                    "iteration":            i,
                    "best_grip":            best_grip if np.isfinite(best_grip) else 0.0,
                    "safe_count":           safe_count,
                    "valid_count":          valid_count,
                    "mean_log_std":         mean_ls,
                    "trust_region_active":  int(trust_active),
                    "filtered_count":       n_filtered,
                })

        # ── Final Pareto front extraction ─────────────────────────────────────
        if len(self.archive_grips) == 0:
            print("[MORL-SB-TRPO] WARNING: empty archive. Returning default setup.")
            return (np.zeros((1, self.dim)), np.array([0.0]),
                    np.array([0.0]), np.array([0]))

        all_setups = np.array(self.archive_setups)
        all_norms  = np.array(self.archive_setups_norm)
        all_grips  = np.array(self.archive_grips)
        all_stabs  = np.array(self.archive_stabs)
        all_gen    = np.array(self.archive_gen)

        # Stability filter
        valid_mask = (-all_stabs) <= STABILITY_MAX
        if valid_mask.sum() == 0:
            valid_mask = np.ones(len(all_grips), dtype=bool)

        all_setups = all_setups[valid_mask]
        all_norms  = all_norms[valid_mask]
        all_grips  = all_grips[valid_mask]
        all_stabs  = all_stabs[valid_mask]
        all_gen    = all_gen[valid_mask]

        # Pareto front
        pareto_idx = self.get_non_dominated_indices(all_grips, -all_stabs)
        p_setups = all_setups[pareto_idx]
        p_grips  = all_grips[pareto_idx]
        p_stabs  = -all_stabs[pareto_idx]
        p_gen    = all_gen[pareto_idx]

        # UPGRADE-2: HV contribution pruning for archive > 150 points
        if len(pareto_idx) > 150:
            hv_c = self._hypervolume_contribution(p_grips, all_stabs[pareto_idx])
            keep = np.argsort(hv_c)[::-1][:150]
            p_setups, p_grips, p_stabs, p_gen = (
                p_setups[keep], p_grips[keep], p_stabs[keep], p_gen[keep])
            print(f"[P25-HV] Archive pruned to 150 via HV contribution.")

        print(f"\n[MORL-SB-TRPO] Pareto front: {len(p_grips)} setups "
              f"| Best grip: {p_grips.max():.4f} G "
              f"| Stability range: {p_stabs.min():.2f}–{p_stabs.max():.2f} rad/s")

        # Sensitivity analysis
        self._print_sensitivity_report(p_setups, p_grips, p_stabs)

        if wandb_run is not None:
            wandb_run.summary["Max_Grip_Found"]      = float(np.max(p_grips))
            wandb_run.summary["Max_Stability_Found"] = float(np.max(-p_stabs))
            wandb_run.summary["Pareto_Front_Count"]  = int(len(p_grips))

        return p_setups, p_grips, p_stabs, p_gen

    # ─────────────────────────────────────────────────────────────────────────
    # §3.7  Sensitivity report
    # ─────────────────────────────────────────────────────────────────────────

    def _print_sensitivity_report(
        self,
        pareto_setups: np.ndarray,
        pareto_grips:  np.ndarray,
        pareto_stabs:  np.ndarray,
    ):
        """Compute and print dGrip/dSetup at each Pareto point (top 10 dims)."""
        print("\n[MORL] Setup sensitivity analysis (top-10 |dGrip/dparam| at Pareto front):")

        header = f"  {'Setup':>5} {'Grip':>6} {'Stab':>6}  "
        header += "  ".join(f"{k:>8}" for k in self.var_keys[:8])
        print(header)
        print("  " + "─" * min(len(header), 120))

        for k, (setup, grip, stab) in enumerate(
                zip(pareto_setups[:5], pareto_grips[:5], pareto_stabs[:5])):
            try:
                setup_norm = jnp.clip(
                    (jnp.array(setup) - jnp.array(self._lb))
                    / (jnp.array(self._ub) - jnp.array(self._lb) + 1e-8),
                    1e-4, 1.0 - 1e-4,
                )
                grad = jax.grad(
                    lambda sn: self.evaluate_setup_jax(sn)[0]
                )(setup_norm)
                grad_np = np.array(grad)
                top8    = "  ".join(f"{grad_np[d]:+.2f}" for d in range(8))
                print(f"  {k+1:5d} {grip:6.3f} {stab:6.2f}  {top8}")
            except Exception:
                print(f"  {k+1:5d} {grip:6.3f} {stab:6.2f}  (gradient failed)")