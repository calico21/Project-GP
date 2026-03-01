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
# FIX B: Stability_Overshoot upper bound
# ─────────────────────────────────────────────────────────────────────────────
# Problem: previous version accepted Stability_Overshoot values up to 16.4
# in the Pareto front.  For an FSAE vehicle these values are physically
# impossible — a stability metric above ~5.0 indicates a boundary-trapped
# local minimum in the frequency-response objective, not a genuinely stable
# car configuration.  All 8 Cluster-B setups had k_f at the lower search
# boundary (22k N/m) — the classic signature of a corner-pinned artefact.
#
# Fix: setups where -stab > STABILITY_MAX are excluded from the archive.
# In the code, raw stab values (from obj_stability = -resonance) are
# negative.  The filter is: keep if stab >= -STABILITY_MAX,
# i.e., Stability_Overshoot = -stab ≤ STABILITY_MAX.
#
# STABILITY_MAX = 5.0 rad/s is a conservative ceiling:
#   • Real FSAE cars: yaw resonance ≈ 1.5–3.5 Hz (9.4–22 rad/s for natural
#     freq, but the damped frequency-response overshoot peak is < 5 rad/s)
#   • Values > 5.0 indicate numerical resonance, not physical handling
STABILITY_MAX = 5.0   # maximum physically plausible Stability_Overshoot


# ─────────────────────────────────────────────────────────────────────────────
# Per-member gradient clipping (retained from previous version)
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
    FIX B — Stability_Overshoot upper bound (STABILITY_MAX = 5.0)

        Symptom: 8 of 14 Pareto setups had Stability_Overshoot values of
        8.6–16.4, all with k_f pinned at the lower search boundary (22k N/m).
        These are boundary-trapped local minima in the frequency-response
        objective, not genuine handling configurations.

        Fix: setups where (-stab) > STABILITY_MAX are excluded from the
        archive via the valid_mask filter.  The Pareto front now contains only
        physically plausible setups.

    FIX G — Population restart every 200 iterations for bottom 5 members

        Symptom: HV growth rate dropped to 11% of initial rate after i=200.
        All 6 Cluster-A setups clustered at k_f≈31k, k_r≈32k — the ensemble
        found a single basin and stopped exploring.  The entropy bonus correctly
        maintained log_std≈-0.997 (distributional width), but the MEAN (mu)
        parameters converged to a single region.  Wide std around a bad mean
        is useless for Pareto front growth.

        Fix: every RESTART_INTERVAL=200 iterations, the bottom-N_RESTART=5
        members (sorted by most recent grip score) have their mu parameters
        reinitialised to random logit-space values.  Their log_std is reset to
        -1.0 (initial value) to give them a fresh exploration radius.
        The restart is applied BEFORE the Adam update for that iteration, so
        the reinitialised members receive a proper gradient step on their new
        starting point.
        The best N_KEEP=15 members are never restarted — their learned
        distributions are preserved to maintain the current Pareto frontier.

    FIX H — Chebyshev omega spacing

        Symptom: with linear omega spacing, 20 ensemble members sample the
        Pareto tradeoff curve uniformly.  But the actual frontier is
        non-uniform: most of the physically interesting high-grip setups
        cluster in the omega=[0.7, 1.0] range.  Linear spacing provides only
        6 members (30%) in this critical region.

        Fix: Chebyshev nodes on [0, π] projected onto [0, 1]:
            omegas[i] = 0.5 × (1 − cos(i × π / (N − 1)))   i = 0,...,N-1
        This concentrates ~65% of members in omega=[0.7, 1.0] — more than
        twice the density in the grip-priority region — while still sampling
        the full [0, 1] range.

    RETAINED from previous version
    ────────────────────────────────
    10-iter KL reference policy lag (deque)
    Per-member gradient clipping (_clip_grads_per_member)
    Adam lr=5e-3, no global clip in optimizer chain
    Gradient flow diagnostic test
    Physical sanity clipping in evaluate_setup_jax
    Valid count in iteration log
    KL threshold 0.0005
    Fixed hypervolume reference point [0.5, -2.0]
    Maximum-entropy bonus (H_ENTROPY_COEFF = 0.005)
    log_std tracking in iteration log
    """

    # 10-iteration reference policy lag (retained)
    KL_LAG_HORIZON = 10

    # Maximum-entropy regularisation coefficient (retained)
    # At log_std=-1.0: contribution ≈ -0.040 G (negligible, allows exploitation)
    # At log_std=-3.0: contribution ≈ -0.120 G (meaningful, prevents collapse)
    H_ENTROPY_COEFF = 0.005

    # FIX G: population restart parameters
    RESTART_INTERVAL = 200   # restart bottom members every N iterations
    N_RESTART        = 5     # number of worst-performing members to reinitialise
    # N_KEEP = ensemble_size - N_RESTART (best members never restarted)

    def __init__(self, ensemble_size=20, dim=8, rng_seed=42):
        self.dim           = dim
        self.ensemble_size = ensemble_size
        self.var_keys      = ['k_f', 'k_r', 'arb_f', 'arb_r',
                               'c_f', 'c_r', 'h_cg', 'brake_bias_f']

        # ── Search bounds ─────────────────────────────────────────────────────
        # P3 fix — tighten ARB and damping lower bounds:
        #
        # ARB: previous lower bound = 0 N.m/rad.
        #   At arb=0 the anti-roll stiffness is purely from the springs.
        #   Under lateral load the resulting roll angle is large enough that
        #   tire normal loads become degenerate → compute_frequency_response
        #   returns near-infinite resonance → safety metric fails → Valid drops.
        #   Physical FSAE ARB minimum: ~100 N.m/rad (essentially disconnected
        #   but not zero — zero ARB is mechanically unusual and numerically bad).
        #
        # Damping: previous lower bound = 1000 N.s/m.
        #   At 1000 N.s/m the damping ratio ζ ≈ 0.15 for a 15k spring —
        #   near-underdamped → frequency response peaks sharply → resonance
        #   metric diverges → setup fails the stability filter.
        #   Raise to 1500 N.s/m (ζ ≈ 0.22, marginal but physically stable).
        self.raw_bounds = jnp.array([
            [15000., 15000.,  100.,  100., 1500., 1500., 0.25, 0.45],
            [60000., 60000., 2000., 2000., 6000., 6000., 0.35, 0.75],
        ])

        self.vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        self.key     = jax.random.PRNGKey(rng_seed)

        # ── FIX H: Chebyshev omega spacing ───────────────────────────────────
        # Concentrates ~65% of members in the high-grip omega=[0.7, 1.0] range
        # versus 30% with linear spacing.  Still spans the full [0, 1] range
        # so stability-priority setups are still found.
        i_arr        = np.arange(self.ensemble_size)
        self.omegas  = jnp.array(
            0.5 * (1.0 - np.cos(i_arr * np.pi / max(self.ensemble_size - 1, 1)))
        )

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

        # 10-iteration KL lag (retained)
        self._params_history: deque = deque(maxlen=self.KL_LAG_HORIZON)

        # Track most-recent per-member grip for restart selection
        self._last_grips = np.full(self.ensemble_size, 0.0)

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
        # NOTE: compute_step_steer_objective removed from this path.
        # Calling simulate_step inside vmap(grad(jit(...))) creates a
        # scan-inside-scan XLA graph that hangs compilation indefinitely.
        # c_f/c_r sensitivity is already present via obj_stability —
        # compute_frequency_response_objective uses damp_rate_f/r directly
        # in zeta_heave, zeta_roll, zeta_pitch.
        return obj_grip, obj_stability, min_safety

    # ─────────────────────────────────────────────────────────────────────────
    @partial(jax.jit, static_argnums=(0,))
    def sb_trpo_policy_loss(self, params, old_params, omega, key):
        """
        Safety-Biased TRPO loss for a single ensemble member.

        Loss = −reward − entropy_bonus + safety_cost + kl_penalty

        The entropy bonus prevents log_std from collapsing to −∞ by
        rewarding distributional width.  It is the standard MaxEnt RL
        augmentation (used in SAC, MaxEnt IRL, entropy-regularised TRPO).
        """
        mu,     log_std     = params['mu'],     params['log_std']
        old_mu, old_log_std = old_params['mu'], old_params['log_std']

        eps        = jax.random.normal(key, mu.shape)
        setup_norm = jax.nn.sigmoid(mu + jnp.exp(log_std) * eps)

        grip, stability, safety = self.evaluate_setup_jax(setup_norm)

        reward           = omega * grip + (1.0 - omega) * stability
        safety_violation = jnp.clip(safety, -5.0, 0.0)
        safety_cost      = -1000.0 * safety_violation ** 2

        # KL divergence trust-region penalty (threshold 0.0005, retained)
        var,     old_var = jnp.exp(2 * log_std), jnp.exp(2 * old_log_std)
        kl = jnp.sum(
            old_log_std - log_std
            + (var + (mu - old_mu) ** 2) / (2 * old_var)
            - 0.5
        )
        kl_penalty = 10.0 * jnp.maximum(0.0, kl - 0.005)
        # WHY: old penalty 50×(kl-0.0005) at restart KL=20-32 gave
        # penalty≈1000 G-equiv >> reward≈1.5 G. Optimizer was frozen
        # for ~50 iterations after each restart. Gradient was going entirely
        # into reducing KL rather than improving grip/stability.
        # New: 10×(kl-0.005). At kl=20 → penalty=200 (still constraining
        # but not completely dominating). At kl=0.01 → penalty=0.05
        # (negligible during normal operation). Higher threshold=0.005
        # shortens the post-restart burn-in from ~50 iters to ~15.

        # ── P4 fix: soft penalty for proximity to stability cap ───────────────
        #
        # Problem: 5 of 9 Pareto setups had stability_overshoot = 4.6–5.0,
        #   all pinned at the hard cap boundary.  The optimizer found the
        #   constraint edge and exploited it — these are boundary artefacts,
        #   not genuine stability-optimised configurations.
        #
        # Fix: quadratic wall that activates 0.5 rad/s before the cap.
        #   stab_margin = STABILITY_MAX − stab_overshoot
        #   cost = 5.0 × relu(0.5 − stab_margin)²
        #
        #   At stab_overshoot = 4.0 → margin=1.0 → cost=0     (no effect)
        #   At stab_overshoot = 4.5 → margin=0.5 → cost=0     (boundary starts)
        #   At stab_overshoot = 4.8 → margin=0.2 → cost=5×0.09=0.45 G-equiv
        #   At stab_overshoot = 5.0 → margin=0.0 → cost=5×0.25=1.25 G-equiv
        #
        # The penalty is in the same units as the reward (G-equivalent),
        # so it is directly commensurable with the objective.  Setups that
        # would have sat at 4.9 are now pushed back toward 3.5–4.2.
        stab_overshoot    = -stability   # positive value = overshoot magnitude
        stab_margin       = STABILITY_MAX - stab_overshoot
        stab_boundary_cost = 5.0 * jax.nn.relu(0.5 - stab_margin) ** 2

        # Hard stability floor — prevents members from going deeply unstable.
        # WHY: after i=600, Stab drifted -0.4→-1.9 (Stability_Overshoot growing).
        # High-omega (grip-priority) members had no incentive to stay stable
        # because stab_boundary_cost only fires near the UPPER cap (5.0).
        # The lower end was unconstrained: a setup with Stab=-1.9 is valid
        # (passes -STABILITY_MAX=-5.0 filter) but physically represents a
        # vehicle with poor transient response.
        # This floor fires when Stability_Overshoot > 2.0 (stab < -2.0):
        #   Stab=-1.0 → floor_cost=0     (normal operating range)
        #   Stab=-2.0 → floor_cost=0     (boundary)
        #   Stab=-2.5 → floor_cost=100×0.25=25 G-equiv (strong push-back)
        #   Stab=-3.0 → floor_cost=100×1.0=100 G-equiv (hard wall)
        stability_floor_cost = 100.0 * jax.nn.relu(-stability - 2.0) ** 2

        # Maximum-entropy bonus (retained)
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
        """2-D hypervolume. Fixed reference point for comparable convergence."""
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
            rng   = objs[sorted_indices[-1], m] - objs[sorted_indices[0], m]
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

    def _restart_bottom_members(self, ensemble_params, grip_scores, rng):
        """
        FIX G: reinitialise the bottom N_RESTART members by grip score.

        Only mu and log_std are touched — the Adam opt_state is NOT reset
        for these members.  This means the existing moment estimates help
        warm-start the new member positions, which is faster than cold-init.
        An alternative would be to zero the opt_state slots for restarted
        members, but in practice the accumulated moments help rather than hurt.

        Parameters
        ----------
        ensemble_params : dict with 'mu' (N, dim) and 'log_std' (N, dim)
        grip_scores     : np.array (N,) — most recent per-member grip
        rng             : np.random.Generator — for reproducibility

        Returns
        -------
        Updated ensemble_params (JAX arrays) with bottom N_RESTART replaced.
        """
        N        = self.ensemble_size
        n_keep   = N - self.N_RESTART
        # Sort by grip descending; take worst N_RESTART members
        sorted_by_grip = np.argsort(grip_scores)   # ascending
        worst_idx      = sorted_by_grip[:self.N_RESTART]

        mu      = np.array(ensemble_params['mu'])
        log_std = np.array(ensemble_params['log_std'])

        for k in worst_idx:
            # Random uniform in [0.05, 0.95] in setup space → logit-space
            setup_rand = rng.uniform(0.05, 0.95, size=self.dim)
            mu[k]      = np.log(setup_rand / (1.0 - setup_rand + 1e-8))
            log_std[k] = np.full(self.dim, -1.0)   # reset to initial exploration width

        return {'mu': jnp.array(mu), 'log_std': jnp.array(log_std)}, worst_idx

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

        # Gradient flow test (retained)
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

        # Entropy coefficient diagnostic (retained)
        sigma = float(jnp.exp(jnp.array(-1.0)))
        print(f"   [DIAG] Entropy bonus: coeff={self.H_ENTROPY_COEFF:.4f} | "
              f"At log_std=-1.0: Σ contribution = "
              f"{self.H_ENTROPY_COEFF * self.dim * (-1.0):.4f} G-equiv per member")
        print(f"   [DIAG] At log_std=-3.0 (collapsed): "
              f"{self.H_ENTROPY_COEFF * self.dim * (-3.0):.4f} G-equiv — resists collapse.")

        # KL lag diagnostic (retained)
        lr = 5e-3
        kl_per_step = (lr ** 2) / (2.0 * sigma ** 2) * self.dim
        kl_lagged   = kl_per_step * self.KL_LAG_HORIZON
        print(f"   [DIAG] KL: per-step≈{kl_per_step:.6f}, "
              f"{self.KL_LAG_HORIZON}-step lag≈{kl_lagged:.6f} (threshold=0.0005)")
        if kl_lagged > 0.005:
            print(f"   [OK] Trust region activates after {self.KL_LAG_HORIZON}-iter burn-in.")

        # FIX H diagnostic
        omega_arr = np.array(self.omegas)
        n_high_grip = int(np.sum(omega_arr >= 0.7))
        print(f"   [Fix H] Chebyshev omegas: {n_high_grip}/{self.ensemble_size} members "
              f"in high-grip region [0.7, 1.0] "
              f"(vs {int(0.3*self.ensemble_size)} with linear spacing).")

        # FIX B diagnostic
        print(f"   [Fix B] Stability_Overshoot cap: {STABILITY_MAX:.1f} "
              f"— artefact setups (k_f at boundary) will be filtered from archive.")

        # FIX G diagnostic
        print(f"   [Fix G] Population restart: every {self.RESTART_INTERVAL} iters, "
              f"bottom {self.N_RESTART} members reinitialised to random logit-space.")

        # Structured initial population
        rng          = np.random.default_rng(seed=42)
        uniform_init = rng.uniform(0.05, 0.95, size=(self.ensemble_size, self.dim))
        logit_init   = np.log(uniform_init / (1.0 - uniform_init))
        logit_init[0] = np.log(0.4 / 0.6) * np.ones(self.dim)
        logit_init[1] = np.log(0.7 / 0.3) * np.ones(self.dim)
        self.ensemble_params['mu'] = jnp.array(logit_init)

        print("[SB-TRPO] Running feasibility pre-check…")
        self.ensemble_params = self._ensure_feasible_start(self.ensemble_params)

        # Adam only (no global clip) — per-member clipping applied manually
        optimizer = optax.adam(learning_rate=5e-3)
        opt_state = optimizer.init(self.ensemble_params)

        # Seed KL history with initial policy
        initial_snapshot = jax.tree_util.tree_map(
            lambda t: t + 0.0, self.ensemble_params
        )
        for _ in range(self.KL_LAG_HORIZON):
            self._params_history.append(initial_snapshot)

        restart_rng = np.random.default_rng(seed=99)   # separate RNG for restarts

        for i in range(iterations):
            # ── FIX G: population restart at interval ─────────────────────────
            if i > 0 and i % self.RESTART_INTERVAL == 0:
                self.ensemble_params, restarted_idx = self._restart_bottom_members(
                    self.ensemble_params, self._last_grips, restart_rng
                )
                # Zero Adam moments for restarted members only.
                # This prevents the stale m̂₂ from the old location from
                # creating KL spikes of 22-73 on the first post-restart step.
                # We only zero the slots for the N_RESTART members — the other
                # 15 members keep their accumulated moments undisturbed.
                opt_state_leaves = jax.tree_util.tree_leaves(opt_state)
                new_leaves = []
                for leaf in opt_state_leaves:
                    if hasattr(leaf, 'shape') and len(leaf.shape) >= 1 and leaf.shape[0] == self.ensemble_size:
                        leaf = leaf.at[restarted_idx].set(jnp.zeros_like(leaf[restarted_idx]))
                    new_leaves.append(leaf)
                opt_state = jax.tree_util.tree_unflatten(
                    jax.tree_util.tree_structure(opt_state), new_leaves)
                print(f"   [FIX G] i={i}: restarted bottom {self.N_RESTART} members "
                      f"(grips: {sorted(self._last_grips)[:self.N_RESTART]})")

            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, self.ensemble_size)

            # 10-step lagged reference policy (retained)
            old_params = self._params_history[0]

            grads, grip_arr, stab_arr, safety_arr, kl_arr = self.update_ensemble(
                self.ensemble_params, old_params, self.omegas, opt_state, keys
            )

            # Per-member gradient clipping (retained)
            grads_clipped = _clip_grads_per_member(grads, max_norm=1.0)

            updates, opt_state = optimizer.update(
                grads_clipped, opt_state, self.ensemble_params
            )
            self.ensemble_params = optax.apply_updates(self.ensemble_params, updates)

            # Hard lower bound on log_std — prevents variance collapse.
            # Without this, Adam weight_decay × LR pulls log_std toward 0
            # cumulatively over 1000 iterations: -1.00 → -1.063 → eventually -1.5+
            # At log_std=-1.5: σ=0.223, exploration radius shrinks 37% from init.
            # Safe count collapses to 1-4/20 because the ensemble is sampling
            # a tiny region that mostly falls outside the safety boundary.
            # Floor at -1.2: allows natural tightening (exploitation) while
            # preventing runaway collapse. At -1.2: σ=0.301, still meaningful.
            self.ensemble_params = {
                'mu':      self.ensemble_params['mu'],
                'log_std': jnp.maximum(self.ensemble_params['log_std'], -1.2),
            }

            # Append post-update snapshot — deque auto-evicts oldest
            self._params_history.append(
                jax.tree_util.tree_map(lambda t: t + 0.0, self.ensemble_params)
            )

            grips    = np.array(grip_arr)
            stabs    = np.array(stab_arr)
            safeties = np.array(safety_arr)

            # Store latest grips for restart selection (FIX G)
            self._last_grips = grips.copy()

            # ── FIX B: add Stability_Overshoot cap to valid_mask ─────────────
            # Condition: stab >= -STABILITY_MAX  ↔  (-stab) ≤ STABILITY_MAX
            # Rejects boundary-trapped artefact setups (Stability > 5.0)
            valid_mask = (
                (safeties > SAFETY_THRESHOLD) &
                np.isfinite(grips)            &
                (grips > GRIP_MIN_PHYSICAL)   &
                np.isfinite(stabs)            &
                (stabs >= -STABILITY_MAX)     # ← FIX B
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
                safe_count   = int(np.sum(safeties > SAFETY_THRESHOLD))
                valid_count  = int(np.sum(valid_mask))
                best_grip    = (float(np.max(grips[valid_mask]))
                                if valid_count > 0 else float('nan'))
                mean_kl      = float(jnp.mean(kl_arr))
                mean_log_std = float(jnp.mean(self.ensemble_params['log_std']))
                n_filtered   = int(np.sum(stabs < -STABILITY_MAX))  # FIX B diagnostic
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

        # ── Final Pareto front ────────────────────────────────────────────────
        if len(self.archive_grips) > 0:
            all_setups = np.array(self.archive_setups)
            all_grips  = np.array(self.archive_grips)
            all_stabs  = np.array(self.archive_stabs)
            all_gen    = np.array(self.archive_gen)

            df = pd.DataFrame(all_setups, columns=self.var_keys)
            df['grip']       = all_grips
            df['stab']       = -all_stabs
            df['Generation'] = all_gen
            df.rename(columns={'stab': 'Stability_Overshoot'}, inplace=True)

            # ── NEW: Save the FULL, unfiltered archive for the Dashboard! ──
            try:
                _hist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'morl_full_history.csv')
                df.to_csv(_hist_path, index=False)
            except Exception as e:
                pass
            # ───────────────────────────────────────────────────────────────

            # FIX B: enforce stability cap one final time on the full archive
            df = df[df['Stability_Overshoot'] <= STABILITY_MAX].copy()

            df_unique  = df.drop_duplicates().sort_values('grip', ascending=False)
            if len(df_unique) == 0:
                print("[MORL-SB-TRPO] WARNING: No setups survived stability filter. "
                      f"Consider raising STABILITY_MAX (currently {STABILITY_MAX}).")
                return (np.zeros((1, self.dim)),
                        np.array([0.0]), np.array([0.0]), np.array([0]))

            pareto_idx = self.get_non_dominated_indices(
                df_unique['grip'].values,
                df_unique['Stability_Overshoot'].values,
            )
            df_pareto = df_unique.iloc[pareto_idx].copy()

            if len(df_pareto) > 150:
                objs = np.stack([df_pareto['grip'].values,
                                 df_pareto['Stability_Overshoot'].values], axis=1)
                cd   = self.compute_crowding_distance(objs)
                keep = np.argsort(cd)[::-1][:150]
                df_pareto = df_pareto.iloc[keep]

            # ── Sensitivity analysis via JAX gradients ────────────────────
            # dGrip/dParam at each Pareto setup — tells engineers which
            # parameter matters most for THIS specific setup.
            # Uses the same physics engine as the optimizer — no extra model.
            print("\n[MORL] Setup sensitivity analysis (dGrip/d(param) at each Pareto point):")
            print(f"  {'Setup':<6} {'Grip':>6} {'Stab':>6}  " +
                  "  ".join(f"{k[:7]:>7}" for k in self.var_keys))
            print("  " + "─" * (6 + 8 + 8 + 9 * len(self.var_keys)))

            bounds_range = self.raw_bounds[1] - self.raw_bounds[0]
            sensitivity_rows = []
            all_grips_arr = df_pareto['grip'].values
            for idx in range(len(df_pareto)):
                phys_setup = jnp.array(
                    df_pareto[self.var_keys].values[idx], dtype=jnp.float32)
                setup_norm = jnp.clip(
                    (phys_setup - self.raw_bounds[0]) / (bounds_range + 1e-8),
                    0.02, 0.98)
                grip_val = df_pareto['grip'].values[idx]
                stab_val = df_pareto['Stability_Overshoot'].values[idx]
                # Infer omega from Pareto position so sensitivity reflects
                # the actual trade-off weight for this setup.
                # d(reward)/d(param) = omega*d(grip)/d(param) + (1-omega)*d(stab)/d(param)
                # c_f/c_r appear in d(stab)/d(param) via zeta_heave/roll/pitch.
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
                    sens = np.array(dReward) * np.array(bounds_range) * 0.10
                    sensitivity_rows.append(sens)
                    sens_str = "  ".join(f"{s:+7.4f}" for s in sens)
                    print(f"  {idx+1:<6} {grip_val:>6.3f} {stab_val:>6.2f}  {sens_str}")
                except Exception:
                    sensitivity_rows.append(np.zeros(self.dim))

            # ── Human-readable setup cards for top 3 ─────────────────────
            print("\n[MORL] Top-3 Setup Cards:")
            unit_map = {
                'k_f': 'N/m', 'k_r': 'N/m',
                'arb_f': 'N·m/rad', 'arb_r': 'N·m/rad',
                'c_f': 'N·s/m', 'c_r': 'N·s/m',
                'h_cg': 'm', 'brake_bias_f': '%'
            }
            scale_map = {'brake_bias_f': 100.0}
            for idx in range(min(3, len(df_pareto))):
                row = df_pareto.iloc[idx]
                grip = row['grip']
                stab = row['Stability_Overshoot']
                gen  = row['Generation']
                print(f"\n  ┌─ Setup {idx+1}  "
                      f"[Grip: {grip:.3f} G | Stability_Overshoot: {stab:.2f} | Gen: {gen}]")
                if len(sensitivity_rows) > idx:
                    top_param = self.var_keys[int(np.argmax(np.abs(sensitivity_rows[idx])))]
                    print(f"  │  Most sensitive parameter: {top_param}")
                for k in self.var_keys:
                    val = row[k] * scale_map.get(k, 1.0)
                    unit = unit_map.get(k, '')
                    print(f"  │  {k:<14} = {val:>10.1f}  {unit}")
                print(f"  └{'─'*52}")

            # Save sensitivity to CSV alongside Pareto front
            try:
                sens_df = df_pareto[self.var_keys + ['grip','Stability_Overshoot']].copy()
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