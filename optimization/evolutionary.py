import os
import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
from functools import partial
from collections import deque
from scipy.stats import norm as _sci_norm

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT
from optimization.objectives import (compute_skidpad_objective,
                                      compute_frequency_response_objective)

SAFETY_THRESHOLD  = 0.0
GRIP_MIN_PHYSICAL = 0.5
GRIP_MAX_PHYSICAL = 3.0

FIXED_HV_REF_POINT = [GRIP_MIN_PHYSICAL, -2.0]

STABILITY_MAX = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-member gradient clipping (retained from previous version)
# ─────────────────────────────────────────────────────────────────────────────
def _clip_grads_per_member(grads: dict, max_norm: float = 1.0) -> dict:
    def clip_single_member(g_flat):
        norm  = jnp.linalg.norm(g_flat)
        scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
        return g_flat * scale
    return jax.tree_util.tree_map(
        lambda g: jax.vmap(clip_single_member)(g),
        grads,
    )


# =============================================================================
# P12 — Bayesian Optimisation Cold-Start
# =============================================================================
class BayesianOptColdStart:
    """
    P12: BO phase-0 cold-start (iterations 0-30).

    Runs (n_init + n_bo) evaluations of the grip objective using a GP
    surrogate + Expected Improvement acquisition before the MORL loop
    begins.  Identifies 3-5 diverse high-grip basins that are used to
    seed the ensemble instead of the previous hardcoded logit_init[0/1].

    Algorithm
    ─────────
    Phase 0a (random):  n_init uniform random evaluations to build an
                        initial GP prior.
    Phase 0b (EI-guided): n_bo iterations of argmax EI acquisition over
                        300 random candidates per step.
    Basin selection:    Top-20 by grip, greedy max-distance pruned to
                        n_basins diverse setups (min L2 = 0.15 in [0,1]^d).

    Expected: safe count at i=0 jumps from 13/20 → 17/20.
              HV reaches 1.70 by iteration 100 (vs. iteration 400 currently).
    """

    def __init__(self, dim: int, n_init: int = 5, n_bo: int = 25,
                 length_scale: float = 0.3, noise: float = 1e-3):
        self.dim          = dim
        self.n_init       = n_init
        self.n_bo         = n_bo
        self.length_scale = length_scale
        self.noise        = noise

    # ── GP internals ──────────────────────────────────────────────────────────

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared-exponential (RBF) kernel."""
        diff  = X1[:, None, :] - X2[None, :, :]
        dist2 = np.sum(diff ** 2 / self.length_scale ** 2, axis=-1)
        return np.exp(-0.5 * dist2)

    def _gp_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_te: np.ndarray):
        """GP posterior mean and std at test points."""
        K   = self._kernel(X_tr, X_tr) + (self.noise + 1e-6) * np.eye(len(X_tr))
        Ks  = self._kernel(X_tr, X_te)
        L   = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
        mu    = Ks.T @ alpha
        v     = np.linalg.solve(L, Ks)
        var   = np.maximum(1.0 - np.sum(v ** 2, axis=0), 1e-10)
        return mu, np.sqrt(var)

    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray,
                               y_best: float, xi: float = 0.01) -> np.ndarray:
        imp = mu - y_best - xi
        z   = imp / (sigma + 1e-10)
        ei  = imp * _sci_norm.cdf(z) + sigma * _sci_norm.pdf(z)
        return np.maximum(ei, 0.0)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, evaluate_fn, rng: np.random.Generator,
            n_basins: int = 5) -> np.ndarray:
        """
        Parameters
        ----------
        evaluate_fn : callable(setup_norm np.ndarray (dim,)) -> float
            Returns the grip score for a normalised setup.
        rng         : seeded np.random.Generator
        n_basins    : number of diverse basins to return

        Returns
        -------
        basins : np.ndarray (n_basins, dim)  normalised setups in [0,1]^d
        """
        # ── Phase 0a: random initialisation ──────────────────────────────────
        X = rng.uniform(0.05, 0.95, size=(self.n_init, self.dim))
        y = np.array([float(evaluate_fn(x)) for x in X], dtype=float)
        print(f"   [P12-BO] Phase 0a ({self.n_init} random evals): "
              f"best grip = {np.max(y):.4f} G")

        # ── Phase 0b: EI-guided acquisition ──────────────────────────────────
        for i in range(self.n_bo):
            candidates     = rng.uniform(0.05, 0.95, size=(300, self.dim))
            mu, sigma      = self._gp_predict(X, y, candidates)
            ei             = self._expected_improvement(mu, sigma, float(np.max(y)))
            best_cand      = candidates[int(np.argmax(ei))]
            new_y          = float(evaluate_fn(best_cand))
            X              = np.vstack([X, best_cand])
            y              = np.append(y, new_y)
            if (i + 1) % 10 == 0:
                print(f"   [P12-BO] Phase 0b iter {i+1:2d}/{self.n_bo}: "
                      f"best grip = {np.max(y):.4f} G")

        # ── Basin selection: greedy max-distance ──────────────────────────────
        top_idx  = list(np.argsort(y)[::-1])
        selected = [top_idx[0]]
        for idx in top_idx[1:]:
            if len(selected) >= n_basins:
                break
            min_dist = min(np.linalg.norm(X[idx] - X[s]) for s in selected)
            if min_dist > 0.15:
                selected.append(idx)

        # Fill to n_basins if diversity threshold was too strict
        for idx in top_idx:
            if len(selected) >= n_basins:
                break
            if idx not in selected:
                selected.append(idx)

        basins = X[selected[:n_basins]]
        print(f"   [P12-BO] {len(selected)} diverse basins. "
              f"Grips: {[f'{y[s]:.4f}G' for s in selected[:n_basins]]}")
        return basins


# =============================================================================
# MORL-SB-TRPO Optimizer
# =============================================================================
class MORL_SB_TRPO_Optimizer:
    """
    Multi-Objective Reinforcement Learning with Safety-Biased Trust Region
    Policy Optimisation.

    CHANGE LOG vs previous version
    ─────────────────────────────────────────────────────────────────────────
    P12 — Bayesian Optimisation Cold-Start (Phase 0)

        Problem: first 50 MORL iterations are essentially random exploration.
        Fix: run 30 BO evaluations (BayesianOptColdStart) before iteration 0.
        The top-5 diverse basins seed the ensemble mu params, replacing the
        hardcoded logit_init[0] = soft / logit_init[1] = hard setup.
        Expected: safe count at i=0: 13/20 → 17/20.

    P24 — Sensitivity-Guided Restart

        Problem: FIX G used random logit-space reinitialisation for the bottom
        N_RESTART members. Wide restarts often land in already-explored regions.
        Fix: compute the Jacobian dGrip/dSetup at each current Pareto point.
        The dimension with MINIMUM gradient variance across Pareto members is
        the least-explored axis.  Restarted members are initialised randomly
        but with a ±2.0 logit-space nudge along this unexplored dimension.
        Alternating sign ensures restarts spread to BOTH ends of that axis.

    P25 — Hypervolume Gradient Archive Management

        Problem: NSGA-II crowding distance prunes by geometric spacing, which
        is independent of the hypervolume objective being maximised.
        Fix: replace crowding distance with the exclusive hypervolume
        contribution (HV_total − HV_without_point_i) for each Pareto member.
        Points with higher HV contribution are kept; low-contribution points
        (which lie in already-densely-sampled regions of the Pareto front)
        are pruned.  This is the SMS-EMOA selection criterion.

    P27 — W&B Full Experiment Tracking

        Problem: only Max_Grip_Found and Max_Stability_Found logged.
        Fix: run() accepts wandb_run=None.  When supplied:
          – Per-iteration: HV, best_grip, safe_count, valid_count logged every iter.
          – Per-20-iters: per-member grip/stab/KL/log_std time-series.
          – Restart events: iteration, unexplored dim, grip before/after.
          – End of run: Pareto front Table, sensitivity Table, final HV.

    RETAINED FROM PREVIOUS VERSION
    ─────────────────────────────────────────────────────────────────────────
    FIX B — STABILITY_MAX = 5.0 cap
    FIX G — Population restart every 200 iterations (now sensitivity-guided)
    FIX H — Chebyshev omega spacing
    10-iter KL reference policy lag (deque)
    Per-member gradient clipping
    Adam lr=5e-3, log_std floor = -1.2
    Stability boundary cost + stability floor cost
    """

    KL_LAG_HORIZON  = 10
    H_ENTROPY_COEFF = 0.005
    RESTART_INTERVAL = 200
    N_RESTART        = 5

    def __init__(self, ensemble_size=20, dim=8, rng_seed=42):
        self.dim           = dim
        self.ensemble_size = ensemble_size
        self.var_keys      = ['k_f', 'k_r', 'arb_f', 'arb_r',
                               'c_f', 'c_r', 'h_cg', 'brake_bias_f']

        self.raw_bounds = jnp.array([
            [15000., 15000.,  100.,  100., 1500., 1500., 0.25, 0.45],
            [60000., 60000., 2000., 2000., 6000., 6000., 0.35, 0.75],
        ])

        self.vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        self.key     = jax.random.PRNGKey(rng_seed)

        i_arr       = np.arange(self.ensemble_size)
        self.omegas = jnp.array(
            0.5 * (1.0 - np.cos(i_arr * np.pi / max(self.ensemble_size - 1, 1)))
        )

        k1, _ = jax.random.split(self.key)
        self.ensemble_params = {
            'mu':      jax.random.uniform(k1, (self.ensemble_size, self.dim),
                                           minval=-0.5, maxval=0.5),
            'log_std': jnp.full((self.ensemble_size, self.dim), -1.0),
        }

        self.archive_setups       = []
        self.archive_setups_norm  = []   # P24: store normalised for gradient computation
        self.archive_grips        = []
        self.archive_stabs        = []
        self.archive_gen          = []

        self._params_history: deque = deque(maxlen=self.KL_LAG_HORIZON)
        self._last_grips = np.full(self.ensemble_size, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Core jitted ops (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_setup(self, x_norm):
        return self.raw_bounds[0] + x_norm * (self.raw_bounds[1] - self.raw_bounds[0])

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_setup_jax(self, setup_norm):
        params = self.unnormalize_setup(setup_norm)

        x_init_skidpad = jnp.zeros(46).at[14].set(15.0)
        obj_grip, min_safety = compute_skidpad_objective(
            self.vehicle.simulate_step, params, x_init_skidpad
        )
        obj_grip = jnp.clip(obj_grip, GRIP_MIN_PHYSICAL, GRIP_MAX_PHYSICAL)

        x_init_freq = jnp.zeros(46).at[14].set(15.0)
        resonance   = compute_frequency_response_objective(
            self.vehicle.simulate_step, params, x_init_freq
        )
        obj_stability = -resonance
        return obj_grip, obj_stability, min_safety

    @partial(jax.jit, static_argnums=(0,))
    def sb_trpo_policy_loss(self, params, old_params, omega, key):
        mu,     log_std     = params['mu'],     params['log_std']
        old_mu, old_log_std = old_params['mu'], old_params['log_std']

        eps        = jax.random.normal(key, mu.shape)
        setup_norm = jax.nn.sigmoid(mu + jnp.exp(log_std) * eps)

        grip, stability, safety = self.evaluate_setup_jax(setup_norm)

        reward           = omega * grip + (1.0 - omega) * stability
        safety_violation = jnp.clip(safety, -5.0, 0.0)
        safety_cost      = -1000.0 * safety_violation ** 2

        var,     old_var = jnp.exp(2 * log_std), jnp.exp(2 * old_log_std)
        kl = jnp.sum(
            old_log_std - log_std
            + (var + (mu - old_mu) ** 2) / (2 * old_var)
            - 0.5
        )
        kl_penalty = 10.0 * jnp.maximum(0.0, kl - 0.005)

        stab_overshoot     = -stability
        stab_margin        = STABILITY_MAX - stab_overshoot
        stab_boundary_cost = 5.0 * jax.nn.relu(0.5 - stab_margin) ** 2
        stability_floor_cost = 100.0 * jax.nn.relu(-stability - 2.0) ** 2

        entropy_bonus = self.H_ENTROPY_COEFF * jnp.sum(log_std)

        loss = (-reward - entropy_bonus
                + safety_cost + kl_penalty
                + stab_boundary_cost + stability_floor_cost)
        return loss, (grip, stability, safety, kl)

    @partial(jax.jit, static_argnums=(0,))
    def update_ensemble(self, ensemble_params, old_ensemble_params,
                        omegas, opt_state, keys):
        vmap_loss_grad = vmap(
            value_and_grad(self.sb_trpo_policy_loss, has_aux=True),
            in_axes=(0, 0, 0, 0),
        )
        (losses, aux), grads = vmap_loss_grad(
            ensemble_params, old_ensemble_params, omegas, keys
        )
        grip, stability, safety, kl = aux
        return grads, grip, stability, safety, kl

    # ─────────────────────────────────────────────────────────────────────────
    # Pareto helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_non_dominated_indices(self, grip_scores, stability_scores):
        objs         = np.stack([grip_scores, stability_scores], axis=1)
        is_efficient = np.ones(objs.shape[0], dtype=bool)
        for i, c in enumerate(objs):
            if is_efficient[i]:
                dominates_c = np.logical_and(
                    np.all(objs >= c, axis=1),
                    np.any(objs >  c, axis=1),
                )
                if np.any(dominates_c):
                    is_efficient[i] = False
        return np.where(is_efficient)[0]

    def hypervolume_indicator(self, grip_scores, stab_scores, ref_point=None):
        """2-D hypervolume. Fixed reference point for comparable convergence curves."""
        if len(grip_scores) == 0:
            return 0.0
        if ref_point is None:
            ref_point = FIXED_HV_REF_POINT

        pareto_idx  = self.get_non_dominated_indices(grip_scores, stab_scores)
        pareto_objs = np.stack([grip_scores[pareto_idx],
                                 stab_scores[pareto_idx]], axis=1)

        sorted_idx = np.argsort(pareto_objs[:, 0])
        pts        = pareto_objs[sorted_idx]

        hv, prev_y = 0.0, ref_point[1]
        for pt in reversed(pts):
            if pt[0] > ref_point[0] and pt[1] > prev_y:
                hv    += (pt[0] - ref_point[0]) * (pt[1] - prev_y)
                prev_y = pt[1]
        return hv

    # ─────────────────────────────────────────────────────────────────────────
    # P25 — Hypervolume Gradient Archive Management
    # ─────────────────────────────────────────────────────────────────────────

    def _hypervolume_contribution(self, grip_scores: np.ndarray,
                                   stab_scores: np.ndarray,
                                   ref_point=None) -> np.ndarray:
        """
        P25: Exclusive hypervolume contribution for each point.

        contrib[i] = HV_total - HV_without_point_i

        This is the SMS-EMOA criterion: keeping points with high exclusive
        contribution maximises the hypervolume indicator directly, whereas
        NSGA-II crowding distance only approximates Pareto density.

        Complexity: O(n²) — acceptable for n ≤ 150.
        """
        if ref_point is None:
            ref_point = FIXED_HV_REF_POINT
        n = len(grip_scores)
        if n <= 2:
            return np.full(n, np.inf)

        total_hv      = self.hypervolume_indicator(grip_scores, stab_scores, ref_point)
        contributions = np.zeros(n)
        for i in range(n):
            mask              = np.ones(n, dtype=bool)
            mask[i]           = False
            hv_without        = self.hypervolume_indicator(
                grip_scores[mask], stab_scores[mask], ref_point)
            contributions[i]  = total_hv - hv_without

        return contributions

    def compute_crowding_distance(self, objs):
        """NSGA-II crowding distance — kept for reference / fallback."""
        num_points = objs.shape[0]
        distances  = np.zeros(num_points)
        if num_points <= 2:
            return np.full(num_points, np.inf)
        for m in range(objs.shape[1]):
            sorted_indices                = np.argsort(objs[:, m])
            distances[sorted_indices[0]]  = np.inf
            distances[sorted_indices[-1]] = np.inf
            rng   = objs[sorted_indices[-1], m] - objs[sorted_indices[0], m]
            scale = rng if rng != 0 else 1.0
            for i in range(1, num_points - 1):
                distances[sorted_indices[i]] += (
                    (objs[sorted_indices[i + 1], m] -
                     objs[sorted_indices[i - 1], m]) / scale
                )
        return distances

    # ─────────────────────────────────────────────────────────────────────────
    # P24 — Sensitivity-Guided Restart helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_pareto_gradients(self,
                                   pareto_setups_norm: np.ndarray) -> np.ndarray | None:
        """
        P24 helper: compute dGrip/dSetup Jacobian at each Pareto point.

        Uses the same jitted physics engine as the optimiser — no extra model.
        Returns shape (n_valid, dim) or None if fewer than 2 points succeed.
        """
        jacobians = []
        for sn in pareto_setups_norm[:min(15, len(pareto_setups_norm))]:
            try:
                dg = np.array(
                    jax.grad(lambda s: self.evaluate_setup_jax(s)[0])(
                        jnp.array(sn, dtype=jnp.float32)
                    )
                )
                if np.all(np.isfinite(dg)):
                    jacobians.append(dg)
            except Exception:
                pass
        return np.stack(jacobians) if len(jacobians) >= 2 else None

    def _restart_sensitivity_guided(self, ensemble_params: dict,
                                     grip_scores: np.ndarray,
                                     pareto_setups_norm: np.ndarray,
                                     rng: np.random.Generator):
        """
        P24: Sensitivity-guided restart.

        Algorithm
        ─────────
        1. Compute Jacobian J = dGrip/dSetup at each current Pareto point.
        2. Compute per-dimension variance of J across Pareto members.
           Low variance → all Pareto members agree → dimension well-explored.
           Low variance → dimension likely saturated → poor restart target.
           Minimum variance dimension = LEAST explored direction.
        3. Restart the bottom N_RESTART members with random logit-space mu
           PLUS a ±2.0 offset along the unexplored dimension (alternating
           sign to spread to both extremes of the axis).

        Falls back to pure random restart if fewer than 2 Pareto gradients
        are available (early iterations before the archive is populated).

        Returns (updated_ensemble_params, worst_idx, unexplored_dim_name)
        """
        J = self._compute_pareto_gradients(pareto_setups_norm)

        sorted_by_grip = np.argsort(grip_scores)          # ascending
        worst_idx      = sorted_by_grip[:self.N_RESTART]

        mu      = np.array(ensemble_params['mu'])
        log_std = np.array(ensemble_params['log_std'])

        if J is not None:
            dim_variance    = np.var(J, axis=0)            # (dim,)
            unexplored_dim  = int(np.argmin(dim_variance))
            unexplored_name = self.var_keys[unexplored_dim]
        else:
            unexplored_dim  = None
            unexplored_name = 'random (too few Pareto points)'

        for ki, k in enumerate(worst_idx):
            setup_rand = rng.uniform(0.05, 0.95, size=self.dim)
            mu[k]      = np.log(setup_rand / (1.0 - setup_rand + 1e-8))
            log_std[k] = np.full(self.dim, -1.0)
            # Sensitivity-guided perturbation along unexplored dimension
            if unexplored_dim is not None:
                direction          = 1.0 if ki % 2 == 0 else -1.0
                mu[k, unexplored_dim] += direction * 2.0   # large logit offset

        return ({'mu': jnp.array(mu), 'log_std': jnp.array(log_std)},
                worst_idx,
                unexplored_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Feasibility pre-check (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_feasible_start(self, ensemble_params):
        mu = np.array(ensemble_params['mu'])
        for k in range(self.ensemble_size):
            setup_norm = jax.nn.sigmoid(jnp.array(mu[k]))
            _, _, safety = self.evaluate_setup_jax(setup_norm)
            if float(safety) <= SAFETY_THRESHOLD:
                p = jnp.array(mu[k])
                for _ in range(30):
                    def safety_loss(params_k):
                        return -self.evaluate_setup_jax(jax.nn.sigmoid(params_k))[2]
                    g = jax.grad(safety_loss)(p)
                    p = p - 0.05 * g / (jnp.linalg.norm(g) + 1e-8)
                mu[k] = np.array(p)
        return {'mu': jnp.array(mu), 'log_std': ensemble_params['log_std']}

    # ─────────────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, iterations: int = 400, wandb_run=None):
        """
        Parameters
        ----------
        iterations : int
            Number of MORL gradient steps (Phase 1 onward).
        wandb_run  : wandb.Run or None
            P27: if supplied, logs per-iteration diagnostics, member
            time-series, restart events, and final Pareto front/sensitivity.
        """
        print("\n[MORL-SB-TRPO] Initialising Pareto Policy Ensemble…")
        print("[SB-TRPO] Compiling 46-DOF physics gradients via XLA…")

        test_soft = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        test_hard = jnp.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9])
        g_soft, s_soft, _ = self.evaluate_setup_jax(test_soft)
        g_hard, s_hard, _ = self.evaluate_setup_jax(test_hard)
        print(f"   [DIAG] Soft grip: {float(g_soft):.4f} G | Hard grip: {float(g_hard):.4f} G")
        if abs(float(g_soft) - float(g_hard)) < 1e-4:
            print("   [DIAG] WARNING: setup_params may not be reaching the physics engine.")

        print("[SB-TRPO] Testing gradient flow through objective…")
        try:
            grad_test = jax.grad(lambda p: self.evaluate_setup_jax(p)[0])(test_soft)
            grad_norm = float(jnp.linalg.norm(grad_test))
            print(f"   Grip gradient norm : {grad_norm:.6f}")
            if grad_norm < 1e-8:
                print("   [FATAL] Zero gradient — check objectives.py.")
            else:
                print(f"   [OK] Gradient flow confirmed — TRPO is active.")
        except Exception as e:
            print(f"   [WARN] Gradient test raised: {e}")

        sigma = float(jnp.exp(jnp.array(-1.0)))
        print(f"   [DIAG] Entropy bonus: coeff={self.H_ENTROPY_COEFF:.4f} | "
              f"At log_std=-1.0: Σ contribution = "
              f"{self.H_ENTROPY_COEFF * self.dim * (-1.0):.4f} G-equiv per member")

        lr = 5e-3
        kl_per_step = (lr ** 2) / (2.0 * sigma ** 2) * self.dim
        kl_lagged   = kl_per_step * self.KL_LAG_HORIZON
        print(f"   [DIAG] KL: per-step≈{kl_per_step:.6f}, "
              f"{self.KL_LAG_HORIZON}-step lag≈{kl_lagged:.6f} (threshold=0.005)")

        omega_arr   = np.array(self.omegas)
        n_high_grip = int(np.sum(omega_arr >= 0.7))
        print(f"   [Fix H] Chebyshev omegas: {n_high_grip}/{self.ensemble_size} members "
              f"in high-grip region [0.7, 1.0].")
        print(f"   [Fix B] Stability_Overshoot cap: {STABILITY_MAX:.1f}.")
        print(f"   [P24]   Sensitivity-guided restart every {self.RESTART_INTERVAL} iters.")
        print(f"   [P25]   HV contribution replaces crowding distance for archive pruning.")

        # ── P12: Bayesian Optimisation Cold-Start ──────────────────────────────
        print("\n[P12-BO] Running Bayesian Optimisation cold-start (30 evals)…")
        bo_rng = np.random.default_rng(seed=7)
        bo     = BayesianOptColdStart(dim=self.dim, n_init=5, n_bo=25)
        bo_eval = lambda s: float(self.evaluate_setup_jax(jnp.array(s, dtype=jnp.float32))[0])
        bo_basins = bo.run(bo_eval, bo_rng, n_basins=5)   # (≤5, dim) in [0,1]^d

        # ── Initialise ensemble: BO basins seed first members, rest random ──────
        rng          = np.random.default_rng(seed=42)
        uniform_init = rng.uniform(0.05, 0.95, size=(self.ensemble_size, self.dim))
        logit_init   = np.log(uniform_init / (1.0 - uniform_init))

        # Replace first N_BASINS members with BO-discovered setups
        n_bo_basins = min(len(bo_basins), self.ensemble_size)
        for j, basin in enumerate(bo_basins[:n_bo_basins]):
            basin_clipped  = np.clip(basin, 0.02, 0.98)
            logit_init[j]  = np.log(basin_clipped / (1.0 - basin_clipped + 1e-8))

        self.ensemble_params['mu'] = jnp.array(logit_init)

        if wandb_run is not None:
            wandb_run.log({"bo_coldstart/best_grip": float(bo_eval(bo_basins[0])),
                           "bo_coldstart/n_basins":   n_bo_basins})

        print(f"[P12-BO] {n_bo_basins} ensemble members seeded from BO basins.")

        print("[SB-TRPO] Running feasibility pre-check…")
        self.ensemble_params = self._ensure_feasible_start(self.ensemble_params)

        optimizer = optax.adam(learning_rate=5e-3)
        opt_state = optimizer.init(self.ensemble_params)

        initial_snapshot = jax.tree_util.tree_map(
            lambda t: t + 0.0, self.ensemble_params
        )
        for _ in range(self.KL_LAG_HORIZON):
            self._params_history.append(initial_snapshot)

        restart_rng = np.random.default_rng(seed=99)

        for i in range(iterations):
            # ── P24: Sensitivity-Guided Restart ───────────────────────────────
            if i > 0 and i % self.RESTART_INTERVAL == 0:
                # Build normalised Pareto setups for gradient computation
                if len(self.archive_setups_norm) > 0:
                    all_norms  = np.array(self.archive_setups_norm)
                    all_grips  = np.array(self.archive_grips)
                    all_stabs  = np.array(self.archive_stabs)
                    pareto_idx_arch = self.get_non_dominated_indices(
                        all_grips, all_stabs
                    )
                    pareto_norms = all_norms[pareto_idx_arch]
                else:
                    pareto_norms = np.array(
                        jax.vmap(jax.nn.sigmoid)(self.ensemble_params['mu'])
                    )

                self.ensemble_params, restarted_idx, unexplored_name = \
                    self._restart_sensitivity_guided(
                        self.ensemble_params, self._last_grips,
                        pareto_norms, restart_rng
                    )

                # Zero Adam moments for restarted slots (prevent KL spike)
                opt_state_leaves = jax.tree_util.tree_leaves(opt_state)
                new_leaves = []
                for leaf in opt_state_leaves:
                    if (hasattr(leaf, 'shape') and len(leaf.shape) >= 1
                            and leaf.shape[0] == self.ensemble_size):
                        leaf = leaf.at[restarted_idx].set(
                            jnp.zeros_like(leaf[restarted_idx])
                        )
                    new_leaves.append(leaf)
                opt_state = jax.tree_util.tree_unflatten(
                    jax.tree_util.tree_structure(opt_state), new_leaves)

                # Flush KL history for restarted members
                new_mu_snap = jnp.array(self.ensemble_params['mu'])
                new_ls_snap = jnp.array(self.ensemble_params['log_std'])
                for snap in self._params_history:
                    snap['mu']      = snap['mu'].at[restarted_idx].set(
                        new_mu_snap[restarted_idx])
                    snap['log_std'] = snap['log_std'].at[restarted_idx].set(
                        new_ls_snap[restarted_idx])

                print(f"   [P24] i={i}: restarted bottom {self.N_RESTART} members "
                      f"along '{unexplored_name}' "
                      f"(grips before: {sorted(self._last_grips)[:self.N_RESTART]})")

                # P27: log restart event
                if wandb_run is not None:
                    wandb_run.log({
                        "restart/iteration":       i,
                        "restart/unexplored_dim":  unexplored_name,
                        "restart/min_grip_before": float(
                            min(sorted(self._last_grips)[:self.N_RESTART])),
                    })

            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, self.ensemble_size)

            old_params = self._params_history[0]

            grads, grip_arr, stab_arr, safety_arr, kl_arr = self.update_ensemble(
                self.ensemble_params, old_params, self.omegas, opt_state, keys
            )

            grads_clipped = _clip_grads_per_member(grads, max_norm=1.0)
            updates, opt_state = optimizer.update(
                grads_clipped, opt_state, self.ensemble_params
            )
            self.ensemble_params = optax.apply_updates(self.ensemble_params, updates)
            self.ensemble_params = {
                'mu':      self.ensemble_params['mu'],
                'log_std': jnp.maximum(self.ensemble_params['log_std'], -1.2),
            }

            self._params_history.append(
                jax.tree_util.tree_map(lambda t: t + 0.0, self.ensemble_params)
            )

            grips    = np.array(grip_arr)
            stabs    = np.array(stab_arr)
            safeties = np.array(safety_arr)
            self._last_grips = grips.copy()

            valid_mask = (
                (safeties > SAFETY_THRESHOLD) &
                np.isfinite(grips)            &
                (grips > GRIP_MIN_PHYSICAL)   &
                np.isfinite(stabs)            &
                (stabs >= -STABILITY_MAX)
            )

            if np.any(valid_mask):
                samples = np.array(
                    jax.vmap(lambda m: jax.nn.sigmoid(m))(self.ensemble_params['mu'])
                )
                phys = np.array(jax.vmap(self.unnormalize_setup)(jnp.array(samples)))
                self.archive_setups.extend(phys[valid_mask])
                self.archive_setups_norm.extend(samples[valid_mask])  # P24: store norm
                self.archive_grips.extend(grips[valid_mask])
                self.archive_stabs.extend(stabs[valid_mask])
                self.archive_gen.extend([i] * int(np.sum(valid_mask)))

            hv = (self.hypervolume_indicator(
                      np.array(self.archive_grips),
                      np.array(self.archive_stabs))
                  if len(self.archive_grips) >= 2 else 0.0)

            # P27: per-iteration lightweight logging (every iteration)
            if wandb_run is not None:
                log_dict = {"iteration": i, "hypervolume": hv}
                if np.any(valid_mask):
                    log_dict["best_grip_iter"] = float(np.max(grips[valid_mask]))
                wandb_run.log(log_dict)

            if i % 20 == 0:
                safe_count   = int(np.sum(safeties > SAFETY_THRESHOLD))
                valid_count  = int(np.sum(valid_mask))
                best_grip    = (float(np.max(grips[valid_mask]))
                                if valid_count > 0 else float('nan'))
                mean_kl      = float(jnp.mean(kl_arr))
                mean_log_std = float(jnp.mean(self.ensemble_params['log_std']))
                n_filtered   = int(np.sum(stabs < -STABILITY_MAX))
                trust_active = mean_kl > 0.0005
                lag_active   = i >= self.KL_LAG_HORIZON

                print(
                    f"   [SB-TRPO] i={i:>4d} | "
                    f"Safe: {safe_count}/{self.ensemble_size} | "
                    f"Valid: {valid_count}/{self.ensemble_size} | "
                    f"Grip: {best_grip:.4f} G | "
                    f"Stab: {float(np.max(stabs)):.4f} | "
                    f"HV: {hv:.6f} | "
                    f"KL: {mean_kl:.6f} | "
                    f"log_std: {mean_log_std:.3f}"
                    + (f" [+{n_filtered} artefacts filtered]" if n_filtered > 0 else "")
                    + (" [TR active]" if trust_active else "")
                    + (" [10-step lag]" if lag_active else " [burn-in]")
                )

                # P27: full per-iteration diagnostic logging every 20 steps
                if wandb_run is not None:
                    member_dict = {}
                    for j in range(self.ensemble_size):
                        member_dict[f"member/grip_{j:02d}"]    = float(grips[j])
                        member_dict[f"member/stab_{j:02d}"]    = float(-stabs[j])
                        member_dict[f"member/kl_{j:02d}"]      = float(kl_arr[j])
                        member_dict[f"member/log_std_{j:02d}"] = float(
                            np.mean(self.ensemble_params['log_std'][j]))
                    wandb_run.log({
                        "iteration":                   i,
                        "diagnostics/safe_count":      safe_count,
                        "diagnostics/valid_count":     valid_count,
                        "diagnostics/best_grip":       best_grip if not np.isnan(best_grip) else 0.0,
                        "diagnostics/max_stability":   float(np.max(-stabs)),
                        "diagnostics/mean_kl":         mean_kl,
                        "diagnostics/mean_log_std":    mean_log_std,
                        "diagnostics/artefacts_filt":  n_filtered,
                        "diagnostics/trust_region_on": int(trust_active),
                        **member_dict,
                    })

        # ── Final Pareto front ─────────────────────────────────────────────────
        if len(self.archive_grips) > 0:
            all_setups      = np.array(self.archive_setups)
            all_setups_norm = np.array(self.archive_setups_norm)
            all_grips       = np.array(self.archive_grips)
            all_stabs       = np.array(self.archive_stabs)
            all_gen         = np.array(self.archive_gen)

            df = pd.DataFrame(all_setups, columns=self.var_keys)
            df['grip']       = all_grips
            df['stab']       = -all_stabs
            df['Generation'] = all_gen
            df.rename(columns={'stab': 'Stability_Overshoot'}, inplace=True)

            try:
                _hist_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'morl_full_history.csv')
                df.to_csv(_hist_path, index=False)
            except Exception:
                pass

            df = df[df['Stability_Overshoot'] <= STABILITY_MAX].copy()

            df_unique  = df.drop_duplicates().sort_values('grip', ascending=False)
            if len(df_unique) == 0:
                print(f"[MORL-SB-TRPO] WARNING: No setups survived stability filter.")
                return (np.zeros((1, self.dim)),
                        np.array([0.0]), np.array([0.0]), np.array([0]))

            pareto_idx = self.get_non_dominated_indices(
                df_unique['grip'].values,
                df_unique['Stability_Overshoot'].values,
            )
            df_pareto = df_unique.iloc[pareto_idx].copy()

            # ── P25: HV Contribution Archive Pruning ──────────────────────────
            if len(df_pareto) > 150:
                hv_contrib = self._hypervolume_contribution(
                    df_pareto['grip'].values,
                    -df_pareto['Stability_Overshoot'].values,  # convert to raw stab
                )
                keep      = np.argsort(hv_contrib)[::-1][:150]
                df_pareto = df_pareto.iloc[keep]
                print(f"[P25] Archive pruned to 150 via HV contribution "
                      f"(was {len(df_pareto) + len(pareto_idx) - 150}).")

            # ── Sensitivity analysis via JAX gradients ────────────────────────
            print("\n[MORL] Setup sensitivity analysis (dGrip/d(param) at each Pareto point):")
            print(f"  {'Setup':<6} {'Grip':>6} {'Stab':>6}  " +
                  "  ".join(f"{k[:7]:>7}" for k in self.var_keys))
            print("  " + "─" * (6 + 8 + 8 + 9 * len(self.var_keys)))

            bounds_range     = self.raw_bounds[1] - self.raw_bounds[0]
            sensitivity_rows = []
            all_grips_arr    = df_pareto['grip'].values

            for idx in range(len(df_pareto)):
                phys_setup = jnp.array(
                    df_pareto[self.var_keys].values[idx], dtype=jnp.float32)
                setup_norm = jnp.clip(
                    (phys_setup - self.raw_bounds[0]) / (bounds_range + 1e-8),
                    0.02, 0.98)
                grip_val = df_pareto['grip'].values[idx]
                stab_val = df_pareto['Stability_Overshoot'].values[idx]
                grip_min = float(np.min(all_grips_arr))
                grip_max = float(np.max(all_grips_arr))
                omega_est = float(np.clip(
                    0.3 + 0.6 * (grip_val - grip_min) / max(grip_max - grip_min, 1e-6),
                    0.3, 0.9))
                try:
                    def total_reward(s):
                        g, st, _ = self.evaluate_setup_jax(s)
                        return omega_est * g + (1.0 - omega_est) * st
                    dReward = jax.grad(total_reward)(setup_norm)
                    sens    = np.array(dReward) * np.array(bounds_range) * 0.10
                    sensitivity_rows.append(sens)
                    sens_str = "  ".join(f"{s:+7.4f}" for s in sens)
                    print(f"  {idx+1:<6} {grip_val:>6.3f} {stab_val:>6.2f}  {sens_str}")
                except Exception:
                    sensitivity_rows.append(np.zeros(self.dim))

            # ── Human-readable setup cards for top 3 ─────────────────────────
            print("\n[MORL] Top-3 Setup Cards:")
            unit_map  = {'k_f': 'N/m', 'k_r': 'N/m', 'arb_f': 'N·m/rad',
                          'arb_r': 'N·m/rad', 'c_f': 'N·s/m', 'c_r': 'N·s/m',
                          'h_cg': 'm', 'brake_bias_f': '%'}
            scale_map = {'brake_bias_f': 100.0}
            for idx in range(min(3, len(df_pareto))):
                row  = df_pareto.iloc[idx]
                grip = row['grip']
                stab = row['Stability_Overshoot']
                gen  = row['Generation']
                print(f"\n  ┌─ Setup {idx+1}  "
                      f"[Grip: {grip:.3f} G | Stability_Overshoot: {stab:.2f} | Gen: {gen}]")
                if len(sensitivity_rows) > idx:
                    top_param = self.var_keys[int(np.argmax(np.abs(sensitivity_rows[idx])))]
                    print(f"  │  Most sensitive parameter: {top_param}")
                for k in self.var_keys:
                    val  = row[k] * scale_map.get(k, 1.0)
                    unit = unit_map.get(k, '')
                    print(f"  │  {k:<14} = {val:>10.1f}  {unit}")
                print(f"  └{'─'*52}")

            # Save sensitivity CSV
            try:
                sens_df = df_pareto[self.var_keys + ['grip', 'Stability_Overshoot']].copy()
                for ki, key in enumerate(self.var_keys):
                    sens_df[f'd_grip_d_{key}'] = [
                        (sensitivity_rows[idx][ki] if idx < len(sensitivity_rows) else 0.0)
                        for idx in range(len(df_pareto))
                    ]
                _sens_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'morl_sensitivity.csv')
                sens_df.to_csv(_sens_path, index=False)
                print(f"\n[MORL] Sensitivity report saved to {_sens_path}")
            except Exception as e:
                print(f"[MORL] Sensitivity save failed: {e}")

            # ── P27: Final W&B logging ────────────────────────────────────────
            if wandb_run is not None:
                try:
                    import wandb
                    # Pareto front table
                    pareto_table = wandb.Table(
                        columns=self.var_keys + ["Grip_G", "Stability_Overshoot", "Generation"]
                    )
                    for _, row in df_pareto.iterrows():
                        pareto_table.add_data(
                            *[row[k] for k in self.var_keys],
                            row['grip'], row['Stability_Overshoot'], row['Generation']
                        )
                    wandb_run.log({"final/pareto_front_table": pareto_table})

                    # Sensitivity table
                    if len(sensitivity_rows) > 0:
                        sens_table = wandb.Table(
                            columns=["setup_id", "grip", "stability"] + self.var_keys
                        )
                        for idx, srow in enumerate(sensitivity_rows):
                            sens_table.add_data(
                                idx + 1,
                                float(df_pareto.iloc[idx]['grip']),
                                float(df_pareto.iloc[idx]['Stability_Overshoot']),
                                *srow.tolist(),
                            )
                        wandb_run.log({"final/sensitivity_table": sens_table})

                    # Scalar summaries
                    wandb_run.log({
                        "final/hypervolume":       hv,
                        "final/pareto_count":      len(df_pareto),
                        "final/max_grip":          float(np.max(df_pareto['grip'].values)),
                        "final/min_stability_os":  float(np.min(df_pareto['Stability_Overshoot'].values)),
                    })
                    print("[P27] Final W&B diagnostics logged.")
                except Exception as e:
                    print(f"[P27] W&B final logging failed: {e}")

            return (df_pareto[self.var_keys].values,
                    df_pareto['grip'].values,
                    df_pareto['Stability_Overshoot'].values,
                    df_pareto['Generation'].values)
        else:
            print("[MORL-SB-TRPO] WARNING: No feasible setups found in archive.")
            return (np.zeros((1, self.dim)),
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([0]))