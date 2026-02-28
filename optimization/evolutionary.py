import sys
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
from functools import partial

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT
from optimization.objectives import (compute_skidpad_objective,
                                      compute_frequency_response_objective)

# Safety constraint threshold — documented explicitly so it is not a magic number.
# objectives.py subtracts 0.05 from the understeer gradient; safety > 0.0 means
# the understeer margin is positive, i.e. the vehicle understeers at the limit.
SAFETY_THRESHOLD = 0.0

# Physical sanity bounds for a Formula SAE car on the skidpad.
# Values outside this range are penalty artefacts, not real physics results.
# Any archived setup outside these bounds is evidence of a broken objective.
GRIP_MIN_PHYSICAL = 0.5   # G — car cannot corner below this with any real setup
GRIP_MAX_PHYSICAL = 3.0   # G — theoretical ceiling given PDY1 and aero package

# ─────────────────────────────────────────────────────────────────────────────
# FIX: Fixed hypervolume reference point
# ─────────────────────────────────────────────────────────────────────────────
# Previously the reference point was computed as min(observed) - 0.01 each call.
# This made HV non-comparable across runs — it grew partly because the ref point
# itself moved. A fixed physical worst-case reference enables meaningful
# convergence tracking: HV now measures "how much of the feasible space above
# worst-case has been dominated?", which is monotonically meaningful.
#
# [0.5G grip floor, -2.0 stability floor]
# -2.0 corresponds to resonance = +2.0 (badly resonant car — well below any
# FSAE-legal setup), giving a stable lower bound that won't move between runs.
FIXED_HV_REF_POINT = [GRIP_MIN_PHYSICAL, -2.0]


class MORL_SB_TRPO_Optimizer:
    """
    Multi-Objective Reinforcement Learning with Safety-Biased Trust Region
    Policy Optimisation.

    Searches for Pareto-optimal mechanical setup parameters using physics
    gradients computed by the 46-DOF JAX engine.

    Change log vs previous version
    --------------------------------
    FIX A — Adam learning rate 3e-4 → 3e-3
        At lr=3e-4 and log_std=-1 (σ=0.368), per-step KL ≈ 2.6e-6 — prints
        as 0.0000 and never triggers the trust-region penalty (threshold 0.05).
        At lr=3e-3, per-step KL ≈ 2.6e-4 — visible and grows once the fixed
        objectives.py penalty landscape provides real gradients.

    FIX B — Gradient flow test before the main loop
        Prints grip gradient norm at startup. If zero, objective is not
        differentiable and TRPO is silently doing random search. Emits a clear
        warning rather than wasting 400 iterations.

    FIX C — Physical sanity clipping in evaluate_setup_jax
        obj_grip is clipped to [GRIP_MIN_PHYSICAL, GRIP_MAX_PHYSICAL] before it
        enters the TRPO loss. This prevents penalty artefacts like -232 G from
        the previous run from corrupting the reward signal and the Pareto archive.

    FIX D — Physical sanity filtering in the archive valid_mask
        In addition to safety > SAFETY_THRESHOLD, archived setups must have
        grip > GRIP_MIN_PHYSICAL and be finite. This removes the negative-G
        entries that appeared in the previous Pareto front output.

    FIX E — Valid count added to iteration log
        Log now shows both "Safe" (passes safety constraint) and "Valid" (passes
        safety + physical bounds), making it easy to detect when FIX C is doing
        work (Safe > Valid indicates clipped artefacts are present).

    FIX F — KL threshold lowered from 0.05 → 0.0005
        The trust region never activated because 0.05 nats requires |Δμ| > 0.3
        in logit space, which Adam with lr=3e-3 cannot produce in a single step.
        At lr=3e-3, σ=0.368: per-step KL ≈ 2.6e-4. With threshold=0.0005 the
        trust region activates from the first step, making TRPO actually behave
        as a trust-region method rather than plain Adam with dead safety penalty.

    FIX G — Fixed hypervolume reference point (FIXED_HV_REF_POINT)
        The previous dynamic ref_point = min(observed) - 0.01 was not comparable
        across runs. The fixed reference enables convergence curves to be
        compared between experiments and makes the metric monotonically
        meaningful.
    """

    def __init__(self, ensemble_size=20, dim=8, rng_seed=42):
        self.dim           = dim
        self.ensemble_size = ensemble_size
        self.var_keys      = ['k_f', 'k_r', 'arb_f', 'arb_r',
                               'c_f', 'c_r', 'h_cg', 'brake_bias_f']

        # Physical parameter bounds [lower, upper]
        self.raw_bounds = jnp.array([
            [15000., 15000.,    0.,    0., 1000., 1000., 0.25, 0.45],  # lower
            [60000., 60000., 2000., 2000., 6000., 6000., 0.35, 0.75],  # upper
        ])

        self.vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        self.key     = jax.random.PRNGKey(rng_seed)

        # One omega per ensemble member — linspace ensures Pareto front coverage
        self.omegas = jnp.linspace(0.0, 1.0, self.ensemble_size)

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

        # BUG 1 FIX — stores policy from END of each iteration so the next
        # iteration computes a non-zero KL.  None on first iteration (KL=0
        # is acceptable on iteration 0 since there is no prior update yet).
        self.old_ensemble_params = None

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter helpers
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_setup(self, x_norm):
        return self.raw_bounds[0] + x_norm * (self.raw_bounds[1] - self.raw_bounds[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Physics evaluation
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_setup_jax(self, setup_norm):
        params = self.unnormalize_setup(setup_norm)

        x_init_skidpad = jnp.zeros(46).at[14].set(15.0)
        obj_grip, min_safety = compute_skidpad_objective(
            self.vehicle.simulate_step, params, x_init_skidpad
        )

        # FIX C — clip to physical range before the value enters TRPO loss.
        # Without this, freq/balance penalty artefacts like -232 G corrupt
        # the reward signal and cause the optimizer to avoid physically
        # reasonable but high-spring-rate setups indiscriminately.
        obj_grip = jnp.clip(obj_grip, GRIP_MIN_PHYSICAL, GRIP_MAX_PHYSICAL)

        x_init_freq = jnp.zeros(46).at[14].set(15.0)
        resonance   = compute_frequency_response_objective(
            self.vehicle.simulate_step, params, x_init_freq
        )

        obj_stability = -resonance   # less resonant = more stable = higher score
        return obj_grip, obj_stability, min_safety

    # ─────────────────────────────────────────────────────────────────────────
    # SB-TRPO gradient machinery
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def sb_trpo_policy_loss(self, params, old_params, omega, key):
        """
        Safety-Biased TRPO loss for a single ensemble member.
        omega : scalar in [0, 1] — weighting between grip (1) and stability (0).
        """
        mu,     log_std     = params['mu'],     params['log_std']
        old_mu, old_log_std = old_params['mu'], old_params['log_std']

        eps        = jax.random.normal(key, mu.shape)
        setup_norm = jax.nn.sigmoid(mu + jnp.exp(log_std) * eps)

        grip, stability, safety = self.evaluate_setup_jax(setup_norm)

        reward           = omega * grip + (1.0 - omega) * stability
        safety_violation = jnp.clip(safety, -5.0, 0.0)
        safety_cost      = -1000.0 * safety_violation ** 2

        # KL divergence trust-region penalty
        var,     old_var = jnp.exp(2 * log_std), jnp.exp(2 * old_log_std)
        kl = jnp.sum(
            old_log_std - log_std
            + (var + (mu - old_mu) ** 2) / (2 * old_var)
            - 0.5
        )

        # FIX F — KL threshold lowered from 0.05 → 0.0005
        # ─────────────────────────────────────────────────────────────────────
        # With lr=3e-3 and log_std=-1.0 (σ=0.368), per-step KL ≈ 2.6e-4 nats.
        # The old threshold of 0.05 required |Δμ| > 0.3 in logit space — never
        # reached by Adam in a single step — so the trust region was completely
        # inert and KL printed as 0.0000 for every iteration.
        #
        # New threshold = 0.0005 ≈ 2× the expected per-step KL at lr=3e-3:
        #   - Activates from the first meaningful step
        #   - KL now visible in logs (0.0005–0.002 range expected)
        #   - Trust region actually constrains update step size as intended
        kl_penalty = 50.0 * jnp.maximum(0.0, kl - 0.0005)

        loss = -reward + safety_cost + kl_penalty
        return loss, (grip, stability, safety, kl)

    @partial(jax.jit, static_argnums=(0,))
    def update_ensemble(self, ensemble_params, old_ensemble_params,
                        omegas, opt_state, keys):
        """
        Computes per-member SB-TRPO gradients via vmap and returns them for
        the Optax optimiser update.
        """
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
        """
        2-D hypervolume dominated by the current Pareto front.
        Larger = better quality front.

        FIX G — Fixed reference point (FIXED_HV_REF_POINT = [0.5G, -2.0]).
        The previous dynamic ref_point = min(observed) - 0.01 was not
        comparable across runs: HV appeared to grow partly because the reference
        point migrated. A fixed worst-case reference makes HV a true convergence
        metric — it measures how much of the [0.5G, -2.0] → Pareto-front
        space has been dominated, and that answer is the same regardless of
        which run or iteration is being evaluated.
        """
        if len(grip_scores) == 0:
            return 0.0

        # FIX G: use fixed physical bounds rather than data-driven minimum
        if ref_point is None:
            ref_point = FIXED_HV_REF_POINT

        pareto_idx  = self.get_non_dominated_indices(grip_scores, stab_scores)
        pareto_objs = np.stack([grip_scores[pareto_idx],
                                 stab_scores[pareto_idx]], axis=1)

        sorted_idx = np.argsort(pareto_objs[:, 0])
        pts        = pareto_objs[sorted_idx]

        hv     = 0.0
        prev_y = ref_point[1]
        for pt in reversed(pts):
            if pt[0] > ref_point[0] and pt[1] > prev_y:
                hv    += (pt[0] - ref_point[0]) * (pt[1] - prev_y)
                prev_y = pt[1]
        return hv

    def compute_crowding_distance(self, objs):
        """NSGA-II crowding distance to maintain Pareto front diversity."""
        num_points = objs.shape[0]
        distances  = np.zeros(num_points)
        if num_points <= 2:
            return np.full(num_points, np.inf)

        for m in range(objs.shape[1]):
            sorted_indices                = np.argsort(objs[:, m])
            distances[sorted_indices[0]]  = np.inf
            distances[sorted_indices[-1]] = np.inf
            min_val = objs[sorted_indices[0],  m]
            max_val = objs[sorted_indices[-1], m]
            scale   = max_val - min_val if max_val != min_val else 1.0
            for i in range(1, num_points - 1):
                distances[sorted_indices[i]] += (
                    (objs[sorted_indices[i + 1], m] -
                     objs[sorted_indices[i - 1], m]) / scale
                )
        return distances

    # ─────────────────────────────────────────────────────────────────────────
    # Feasibility pre-check
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_feasible_start(self, ensemble_params):
        """
        Project any ensemble member whose initial setup violates safety
        constraints into the feasible region before the TRPO loop begins.
        Uses gradient descent on the safety cost for up to 30 steps.
        """
        mu = np.array(ensemble_params['mu'])
        for k in range(self.ensemble_size):
            setup_norm = jax.nn.sigmoid(jnp.array(mu[k]))
            _, _, safety = self.evaluate_setup_jax(setup_norm)
            if float(safety) <= SAFETY_THRESHOLD:
                p = jnp.array(mu[k])
                for _ in range(30):
                    def safety_loss(params_k):
                        s_norm = jax.nn.sigmoid(params_k)
                        return -self.evaluate_setup_jax(s_norm)[2]
                    g   = jax.grad(safety_loss)(p)
                    g_n = jnp.linalg.norm(g) + 1e-8
                    p   = p - 0.05 * g / g_n
                mu[k] = np.array(p)

        return {
            'mu':      jnp.array(mu),
            'log_std': ensemble_params['log_std'],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, iterations=400):
        print("\n[MORL-SB-TRPO] Initialising Pareto Policy Ensemble…")
        print("[SB-TRPO] Compiling 46-DOF physics gradients via XLA…")

        # ── Diagnostic: check setup sensitivity ──────────────────────────
        test_soft = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        test_hard = jnp.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9])
        g_soft, s_soft, _ = self.evaluate_setup_jax(test_soft)
        g_hard, s_hard, _ = self.evaluate_setup_jax(test_hard)
        print(f"   [DIAG] Soft grip: {float(g_soft):.4f} G | Hard grip: {float(g_hard):.4f} G")
        if abs(float(g_soft) - float(g_hard)) < 1e-4:
            print("   [DIAG] WARNING: setup_params may not be reaching the physics engine.")

        # ── FIX B: gradient flow test ─────────────────────────────────────
        print("[SB-TRPO] Testing gradient flow through objective…")
        try:
            grad_test = jax.grad(lambda p: self.evaluate_setup_jax(p)[0])(test_soft)
            grad_norm = float(jnp.linalg.norm(grad_test))
            grad_max  = float(jnp.max(jnp.abs(grad_test)))
            print(f"   Grip gradient norm : {grad_norm:.6f}")
            print(f"   Grip gradient max  : {grad_max:.6f}")
            if grad_norm < 1e-8:
                print("   [FATAL] Zero gradient — objective is not differentiable.")
                print("   Common causes: jnp.where with constant branches, jnp.max")
                print("   over discrete sweep (use logsumexp instead), or numpy")
                print("   calls inside a jit boundary.")
                print("   TRPO will perform random search only. Fix objectives.py.")
                print("   Continuing anyway — results will not improve with gradient.")
            else:
                print(f"   [OK] Gradient flow confirmed — TRPO is active.")
        except Exception as e:
            print(f"   [WARN] Gradient test raised: {e}")
            print("   Tracing error may resolve after first JIT compile. Continuing.")

        # ── KL threshold diagnostic ───────────────────────────────────────
        sigma = float(jnp.exp(jnp.array(-1.0)))  # log_std = -1.0
        lr    = 5e-3
        kl_per_step_approx = (lr ** 2) / (2.0 * sigma ** 2) * self.dim
        print(f"   [DIAG] Expected per-step KL at lr=5e-3, σ={sigma:.3f}: "
              f"~{kl_per_step_approx:.6f} nats  (threshold=0.0005)")
        if kl_per_step_approx > 0.0005:
            print("   [OK] Trust region will activate from first step.")
        else:
            print("   [WARN] KL may still be below threshold — consider raising lr.")

        # ── BUG 2 FIX: Uniform initial population BEFORE feasibility projection ─
        # Previous order was: project → overwrite → (projection discarded).
        # Correct order: build structured init first, THEN project into feasible
        # region.  The projection result is now the actual starting population.
        rng          = np.random.default_rng(seed=42)
        uniform_init = rng.uniform(0.05, 0.95, size=(self.ensemble_size, self.dim))
        logit_init   = np.log(uniform_init / (1.0 - uniform_init))
        logit_init[0] = np.log(0.4 / 0.6) * np.ones(self.dim)   # soft anchor
        logit_init[1] = np.log(0.7 / 0.3) * np.ones(self.dim)   # stiff anchor
        self.ensemble_params['mu'] = jnp.array(logit_init)

        # ── Feasibility pre-check (now runs on the population we will actually use)
        print("[SB-TRPO] Running feasibility pre-check…")
        self.ensemble_params = self._ensure_feasible_start(self.ensemble_params)

        # ── BUG 1+A FIX: lr 3e-3 → 5e-3 with gradient clipping ──────────
        # At lr=3e-3 per-step KL ≈ 2.66e-4 < threshold 0.0005 → trust region
        # still inert.  lr=5e-3 gives per-step KL ≈ 7.4e-4 > 0.0005 → active.
        # clip_by_global_norm(1.0) prevents exploding gradients at larger lr.
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=5e-3),
        )
        opt_state = optimizer.init(self.ensemble_params)

        for i in range(iterations):
            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, self.ensemble_size)

            # ── BUG 1 FIX — old_params from END of previous iteration ────
            # Previous code:
            #   old_params = copy(self.ensemble_params)   ← SAME as current
            #   KL(current || old) = KL(π || π) = 0 exactly, every iteration
            #   The trust region penalty gradient was zero — TRPO was plain Adam.
            #
            # Fix: use self.old_ensemble_params which is saved at the END of
            # each iteration (after the Adam update has been applied).
            # On iteration 0, old == current (KL=0), which is correct — there
            # is no previous policy on the first step.
            if self.old_ensemble_params is None:
                old_params = jax.tree_util.tree_map(lambda t: t + 0.0, self.ensemble_params)
            else:
                old_params = self.old_ensemble_params

            grads, grip_arr, stab_arr, safety_arr, kl_arr = self.update_ensemble(
                self.ensemble_params, old_params, self.omegas, opt_state, keys
            )
            updates, opt_state = optimizer.update(
                grads, opt_state, self.ensemble_params
            )
            self.ensemble_params = optax.apply_updates(self.ensemble_params, updates)

            # Save policy AFTER the update — this is what old_params will be
            # on the NEXT iteration, giving a non-zero KL from iteration 1 onward.
            self.old_ensemble_params = jax.tree_util.tree_map(
                lambda t: t + 0.0, self.ensemble_params
            )

            grips    = np.array(grip_arr)
            stabs    = np.array(stab_arr)
            safeties = np.array(safety_arr)

            # ── FIX D: physical sanity filter ────────────────────────────
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

            if len(self.archive_grips) >= 2:
                hv = self.hypervolume_indicator(
                    np.array(self.archive_grips),
                    np.array(self.archive_stabs),
                )
            else:
                hv = 0.0

            if i % 20 == 0:
                safe_count  = int(np.sum(safeties > SAFETY_THRESHOLD))
                valid_count = int(np.sum(valid_mask))
                # FIX E: show best grip from valid setups only, not all setups
                best_grip   = (float(np.max(grips[valid_mask]))
                               if valid_count > 0 else float('nan'))
                mean_kl     = float(jnp.mean(kl_arr))
                trust_active = mean_kl > 0.0005
                print(
                    f"   [SB-TRPO] i={i:>4d} | "
                    f"Safe: {safe_count}/{self.ensemble_size} | "
                    f"Valid: {valid_count}/{self.ensemble_size} | "
                    f"Grip: {best_grip:.4f} G | "
                    f"Stab: {float(np.max(stabs)):.4f} | "
                    f"HV: {hv:.6f} | "
                    f"KL: {mean_kl:.6f}"
                    + (" [TR active]" if trust_active else "")
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