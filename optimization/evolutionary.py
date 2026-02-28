import sys
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
from functools import partial
from collections import deque

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT
from optimization.objectives import (compute_skidpad_objective,
                                      compute_frequency_response_objective)

SAFETY_THRESHOLD  = 0.0
GRIP_MIN_PHYSICAL = 0.5
GRIP_MAX_PHYSICAL = 3.0

# Fixed hypervolume reference point — enables comparable convergence curves.
FIXED_HV_REF_POINT = [GRIP_MIN_PHYSICAL, -2.0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-member gradient clipping (Bug 3 fix — retained from previous version)
# ─────────────────────────────────────────────────────────────────────────────
def _clip_grads_per_member(grads: dict, max_norm: float = 1.0) -> dict:
    """
    Clips gradient norms independently for each ensemble member.
    Replaces optax.clip_by_global_norm which throttled all 20 members
    whenever a single outlier had a large gradient.
    """
    def clip_single_member(g_flat):
        norm  = jnp.linalg.norm(g_flat)
        scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
        return g_flat * scale

    return jax.tree_util.tree_map(
        lambda g: jax.vmap(clip_single_member)(g),
        grads,
    )


class MORL_SB_TRPO_Optimizer:
    """
    Multi-Objective Reinforcement Learning with Safety-Biased Trust Region
    Policy Optimisation.

    CHANGE LOG vs previous version
    ─────────────────────────────────────────────────────────────────────────
    NEW — Maximum-entropy bonus (entropy regularisation)

        Symptom: HV plateau after ~iteration 100, only 21 Pareto points in
        3 tight clusters, +9.3% HV improvement over 400 iterations.

        Root cause: `log_std` is initialised at -1.0 (σ=0.368) and evolves
        freely, but the KL trust-region penalty effectively suppresses its
        divergence. Members that find high-reward regions collapse their
        distributions (log_std → -∞) to exploit rather than explore.
        Members stuck in flat regions don't widen because their reward
        gradient is too small to overcome the KL penalty on log_std changes.
        After ~100 iterations all members have narrow, specialised
        distributions — the ensemble stops covering new regions of parameter
        space and the Pareto front stops growing.

        Fix: add a maximum-entropy bonus to the TRPO loss:
            entropy_bonus = H_ENTROPY_COEFF × Σ log_std_i
        This is the standard maximum-entropy RL augmentation (used in SAC,
        MaxEnt IRL, and entropy-regularised TRPO). The constant
        0.5 × log(2πe) terms in the Gaussian entropy are dropped because
        they don't depend on trainable parameters.

        The coefficient H_ENTROPY_COEFF = 0.005 is sized so that:
          • At log_std = -1.0 (σ=0.368, initial value):
            entropy contribution = 0.005 × 8 × (-1.0) = -0.040 — negligible
            relative to a typical reward of ~1.4 G
          • At log_std = -3.0 (σ=0.050, collapsed distribution):
            entropy contribution = 0.005 × 8 × (-3.0) = -0.120 — meaningful
            penalty that pushes log_std back toward -1.0 to -2.0
          • At log_std = 0.0 (σ=1.0, over-wide distribution):
            entropy contribution = 0 — no penalty for exploring widely

        Physical interpretation: each ensemble member is incentivised to
        maintain enough distributional width to explore its neighbourhood of
        setup space rather than collapsing onto a single point.  This is
        correct — the ensemble is meant to cover the Pareto front, not
        converge to individual point estimates.

        Expected effect: HV improvement increases from +9.3% to +25%+ over
        400 iterations, with Pareto front growing from 21 to 40+ points.

    RETAINED from previous version
    ────────────────────────────────
    Bug 1/4 FIX — 10-iteration KL reference policy lag (deque)
    Bug 3 FIX   — per-member gradient clipping (_clip_grads_per_member)
    Fix A       — lr 5e-3, no global clip in optimizer chain
    Fix B       — gradient flow test
    Fix C       — physical sanity clipping in evaluate_setup_jax
    Fix D       — physical sanity filter in archive valid_mask
    Fix E       — valid count in iteration log
    Fix F       — KL threshold 0.0005
    Fix G       — fixed hypervolume reference point
    """

    # 10-iteration reference policy lag (Bug 4 fix)
    KL_LAG_HORIZON = 10

    # Maximum-entropy regularisation coefficient (NEW)
    # Sized so entropy term is ~3-8% of typical reward magnitude.
    # At log_std=-1: contribution ≈ 0.040 G  (negligible, allows exploitation)
    # At log_std=-3: contribution ≈ 0.120 G  (meaningful, prevents collapse)
    H_ENTROPY_COEFF = 0.005

    def __init__(self, ensemble_size=20, dim=8, rng_seed=42):
        self.dim           = dim
        self.ensemble_size = ensemble_size
        self.var_keys      = ['k_f', 'k_r', 'arb_f', 'arb_r',
                               'c_f', 'c_r', 'h_cg', 'brake_bias_f']

        self.raw_bounds = jnp.array([
            [15000., 15000.,    0.,    0., 1000., 1000., 0.25, 0.45],
            [60000., 60000., 2000., 2000., 6000., 6000., 0.35, 0.75],
        ])

        self.vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        self.key     = jax.random.PRNGKey(rng_seed)
        self.omegas  = jnp.linspace(0.0, 1.0, self.ensemble_size)

        k1, _ = jax.random.split(self.key)
        self.ensemble_params = {
            'mu':      jax.random.uniform(k1, (self.ensemble_size, self.dim),
                                           minval=-0.5, maxval=0.5),
            'log_std': jnp.full((self.ensemble_size, self.dim), -1.0),
        }

        self.archive_setups = []
        self.archive_grips  = []
        self.archive_stabs  = []
        self.archive_gen    = []

        # 10-iteration KL lag (Bug 4 fix)
        self._params_history: deque = deque(maxlen=self.KL_LAG_HORIZON)

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
        # Fix C: physical sanity clip before TRPO loss
        obj_grip = jnp.clip(obj_grip, GRIP_MIN_PHYSICAL, GRIP_MAX_PHYSICAL)

        x_init_freq = jnp.zeros(46).at[14].set(15.0)
        resonance   = compute_frequency_response_objective(
            self.vehicle.simulate_step, params, x_init_freq
        )
        obj_stability = -resonance
        return obj_grip, obj_stability, min_safety

    # ─────────────────────────────────────────────────────────────────────────
    @partial(jax.jit, static_argnums=(0,))
    def sb_trpo_policy_loss(self, params, old_params, omega, key):
        """
        Safety-Biased TRPO loss for a single ensemble member.

        Loss = −reward − entropy_bonus + safety_cost + kl_penalty

        The entropy bonus (NEW) prevents log_std from collapsing to −∞ by
        rewarding distributional width.  It is equivalent to adding a Gaussian
        entropy term H(π) = Σ log_std_i (up to constants) to the objective.
        """
        mu,     log_std     = params['mu'],     params['log_std']
        old_mu, old_log_std = old_params['mu'], old_params['log_std']

        eps        = jax.random.normal(key, mu.shape)
        setup_norm = jax.nn.sigmoid(mu + jnp.exp(log_std) * eps)

        grip, stability, safety = self.evaluate_setup_jax(setup_norm)

        reward           = omega * grip + (1.0 - omega) * stability
        safety_violation = jnp.clip(safety, -5.0, 0.0)
        safety_cost      = -1000.0 * safety_violation ** 2

        # KL divergence trust-region penalty (Fix F: threshold 0.0005)
        var,     old_var = jnp.exp(2 * log_std), jnp.exp(2 * old_log_std)
        kl = jnp.sum(
            old_log_std - log_std
            + (var + (mu - old_mu) ** 2) / (2 * old_var)
            - 0.5
        )
        kl_penalty = 50.0 * jnp.maximum(0.0, kl - 0.0005)

        # ── NEW: maximum-entropy bonus ────────────────────────────────────
        # H(π) for diagonal Gaussian = Σ [log_std_i + 0.5*log(2πe)]
        # The constant 0.5*log(2πe) terms are parameter-independent and
        # omitted. Only Σ log_std_i matters for the gradient.
        #
        # Negative sign because:
        #   • We minimise `loss` in the optimiser
        #   • We want to MAXIMISE entropy  →  MINIMISE (−entropy)
        #   • −(H_ENTROPY_COEFF × Σ log_std) subtracted = added to reward side
        #
        # Effect on log_std gradient:
        #   ∂loss/∂log_std_i += −H_ENTROPY_COEFF  (independent of i)
        # This is a constant downward push on the loss in the log_std direction,
        # which translates to an upward push on log_std itself — preventing
        # the distribution from collapsing.
        entropy_bonus = self.H_ENTROPY_COEFF * jnp.sum(log_std)

        loss = -reward - entropy_bonus + safety_cost + kl_penalty
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
        """2-D hypervolume. Fix G: fixed reference point."""
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

    def compute_crowding_distance(self, objs):
        """NSGA-II crowding distance for Pareto diversity."""
        num_points = objs.shape[0]
        distances  = np.zeros(num_points)
        if num_points <= 2:
            return np.full(num_points, np.inf)
        for m in range(objs.shape[1]):
            sorted_indices                = np.argsort(objs[:, m])
            distances[sorted_indices[0]]  = np.inf
            distances[sorted_indices[-1]] = np.inf
            rng = objs[sorted_indices[-1], m] - objs[sorted_indices[0], m]
            scale = rng if rng != 0 else 1.0
            for i in range(1, num_points - 1):
                distances[sorted_indices[i]] += (
                    (objs[sorted_indices[i + 1], m] -
                     objs[sorted_indices[i - 1], m]) / scale
                )
        return distances

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

    def run(self, iterations=400):
        print("\n[MORL-SB-TRPO] Initialising Pareto Policy Ensemble…")
        print("[SB-TRPO] Compiling 46-DOF physics gradients via XLA…")

        test_soft = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        test_hard = jnp.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9])
        g_soft, s_soft, _ = self.evaluate_setup_jax(test_soft)
        g_hard, s_hard, _ = self.evaluate_setup_jax(test_hard)
        print(f"   [DIAG] Soft grip: {float(g_soft):.4f} G | Hard grip: {float(g_hard):.4f} G")
        if abs(float(g_soft) - float(g_hard)) < 1e-4:
            print("   [DIAG] WARNING: setup_params may not be reaching the physics engine.")

        # Fix B: gradient flow test
        print("[SB-TRPO] Testing gradient flow through objective…")
        try:
            grad_test = jax.grad(lambda p: self.evaluate_setup_jax(p)[0])(test_soft)
            grad_norm = float(jnp.linalg.norm(grad_test))
            print(f"   Grip gradient norm : {grad_norm:.6f}")
            if grad_norm < 1e-8:
                print("   [FATAL] Zero gradient — check objectives.py for non-differentiable ops.")
            else:
                print(f"   [OK] Gradient flow confirmed — TRPO is active.")
        except Exception as e:
            print(f"   [WARN] Gradient test raised: {e}")

        # Entropy coefficient diagnostic
        sigma = float(jnp.exp(jnp.array(-1.0)))
        print(f"   [DIAG] Entropy bonus: coeff={self.H_ENTROPY_COEFF:.4f} | "
              f"At log_std=-1.0: Σ contribution = "
              f"{self.H_ENTROPY_COEFF * self.dim * (-1.0):.4f} G-equiv per member")
        print(f"   [DIAG] At log_std=-3.0 (collapsed): "
              f"{self.H_ENTROPY_COEFF * self.dim * (-3.0):.4f} G-equiv — "
              f"resists collapse.")

        # KL lag diagnostic
        lr = 5e-3
        kl_per_step = (lr ** 2) / (2.0 * sigma ** 2) * self.dim
        kl_lagged   = kl_per_step * self.KL_LAG_HORIZON
        print(f"   [DIAG] KL: per-step≈{kl_per_step:.6f}, "
              f"{self.KL_LAG_HORIZON}-step lag≈{kl_lagged:.6f} (threshold=0.0005)")
        if kl_lagged > 0.0005:
            print(f"   [OK] Trust region activates after {self.KL_LAG_HORIZON}-iter burn-in.")

        # Structured initial population
        rng          = np.random.default_rng(seed=42)
        uniform_init = rng.uniform(0.05, 0.95, size=(self.ensemble_size, self.dim))
        logit_init   = np.log(uniform_init / (1.0 - uniform_init))
        logit_init[0] = np.log(0.4 / 0.6) * np.ones(self.dim)
        logit_init[1] = np.log(0.7 / 0.3) * np.ones(self.dim)
        self.ensemble_params['mu'] = jnp.array(logit_init)

        print("[SB-TRPO] Running feasibility pre-check…")
        self.ensemble_params = self._ensure_feasible_start(self.ensemble_params)

        # Bug 3 fix: Adam only (no global clip) — per-member clipping applied manually
        optimizer = optax.adam(learning_rate=5e-3)
        opt_state = optimizer.init(self.ensemble_params)

        # Seed KL history with initial policy (correct burn-in behaviour)
        initial_snapshot = jax.tree_util.tree_map(
            lambda t: t + 0.0, self.ensemble_params
        )
        for _ in range(self.KL_LAG_HORIZON):
            self._params_history.append(initial_snapshot)

        for i in range(iterations):
            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, self.ensemble_size)

            # Bug 4 fix: 10-step lagged reference policy
            old_params = self._params_history[0]

            grads, grip_arr, stab_arr, safety_arr, kl_arr = self.update_ensemble(
                self.ensemble_params, old_params, self.omegas, opt_state, keys
            )

            # Bug 3 fix: per-member gradient clipping
            grads_clipped = _clip_grads_per_member(grads, max_norm=1.0)

            updates, opt_state = optimizer.update(
                grads_clipped, opt_state, self.ensemble_params
            )
            self.ensemble_params = optax.apply_updates(self.ensemble_params, updates)

            # Append post-update snapshot — deque auto-evicts oldest
            self._params_history.append(
                jax.tree_util.tree_map(lambda t: t + 0.0, self.ensemble_params)
            )

            grips    = np.array(grip_arr)
            stabs    = np.array(stab_arr)
            safeties = np.array(safety_arr)

            # Fix D: physical sanity filter
            valid_mask = (
                (safeties > SAFETY_THRESHOLD) &
                np.isfinite(grips)            &
                (grips > GRIP_MIN_PHYSICAL)   &
                np.isfinite(stabs)
            )

            if np.any(valid_mask):
                samples = np.array(
                    jax.vmap(lambda m: jax.nn.sigmoid(m))(self.ensemble_params['mu'])
                )
                phys = np.array(jax.vmap(self.unnormalize_setup)(jnp.array(samples)))
                self.archive_setups.extend(phys[valid_mask])
                self.archive_grips.extend(grips[valid_mask])
                self.archive_stabs.extend(stabs[valid_mask])
                self.archive_gen.extend([i] * int(np.sum(valid_mask)))

            hv = (self.hypervolume_indicator(
                      np.array(self.archive_grips),
                      np.array(self.archive_stabs))
                  if len(self.archive_grips) >= 2 else 0.0)

            if i % 20 == 0:
                safe_count  = int(np.sum(safeties > SAFETY_THRESHOLD))
                valid_count = int(np.sum(valid_mask))
                best_grip   = (float(np.max(grips[valid_mask]))
                               if valid_count > 0 else float('nan'))
                mean_kl     = float(jnp.mean(kl_arr))
                # log_std mean across ensemble and dims — tracks exploration health
                mean_log_std = float(jnp.mean(self.ensemble_params['log_std']))
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
                    f"log_std: {mean_log_std:.3f}"  # NEW: tracks entropy health
                    + (" [TR active]" if trust_active else "")
                    + (" [10-step lag]" if lag_active else " [burn-in]")
                )

        # ── Final Pareto front ────────────────────────────────────────────
        if len(self.archive_grips) > 0:
            all_setups = np.array(self.archive_setups)
            all_grips  = np.array(self.archive_grips)
            all_stabs  = np.array(self.archive_stabs)
            all_gen    = np.array(self.archive_gen)

            df = pd.DataFrame(all_setups, columns=self.var_keys)
            df['grip']       = all_grips
            df['stab']       = -all_stabs
            df['Generation'] = all_gen
            df.rename(columns={'stab': 'Understeer_Margin'}, inplace=True)

            df_unique  = df.drop_duplicates().sort_values('grip', ascending=False)
            pareto_idx = self.get_non_dominated_indices(
                df_unique['grip'].values,
                df_unique['Understeer_Margin'].values,
            )
            df_pareto = df_unique.iloc[pareto_idx].copy()

            if len(df_pareto) > 150:
                objs = np.stack([df_pareto['grip'].values,
                                  df_pareto['Understeer_Margin'].values], axis=1)
                cd   = self.compute_crowding_distance(objs)
                keep = np.argsort(cd)[::-1][:150]
                df_pareto = df_pareto.iloc[keep]

            return (df_pareto[self.var_keys].values,
                    df_pareto['grip'].values,
                    df_pareto['Understeer_Margin'].values,
                    df_pareto['Generation'].values)
        else:
            print("[MORL-SB-TRPO] WARNING: No feasible setups found in archive.")
            return (np.zeros((1, self.dim)),
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([0]))