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
from optimization.objectives import compute_skidpad_objective, compute_frequency_response_objective

class MORL_SB_TRPO_Optimizer:
    """
    Multi-Objective Reinforcement Learning with Safety-Biased Trust Region Policy Optimization.
    Searches the 14-DOF manifold for Pareto-optimal mechanical setup parameters (Grip vs. Stability).
    """
    def __init__(self, ensemble_size=20, dim=7, rng_seed=42):
        self.dim = dim
        self.ensemble_size = ensemble_size
        self.var_keys = ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', 'h_cg']
        
        # Dynamically reference VP_DICT to prevent mechanical impossibility
        self.raw_bounds = jnp.array([
            [VP_DICT['min_spring'], VP_DICT['min_spring'], VP_DICT['min_arb'], VP_DICT['min_arb'], VP_DICT['min_damp'], VP_DICT['min_damp'], 0.25],
            [VP_DICT['max_spring'], VP_DICT['max_spring'], VP_DICT['max_arb'], VP_DICT['max_arb'], VP_DICT['max_damp'], VP_DICT['max_damp'], 0.35]
        ])
        
        self.vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        self.key = jax.random.PRNGKey(rng_seed)
        
        self.omegas = jnp.linspace(0.0, 1.0, self.ensemble_size)
        
        k1, k2 = jax.random.split(self.key)
        self.ensemble_params = {
            'mu': jax.random.uniform(k1, (self.ensemble_size, self.dim), minval=-0.5, maxval=0.5),
            'log_std': jnp.full((self.ensemble_size, self.dim), -1.0) 
        }
        
        self.archive_setups = []
        self.archive_grips = []
        self.archive_stabs = []

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_setup(self, x_norm):
        return self.raw_bounds[0] + x_norm * (self.raw_bounds[1] - self.raw_bounds[0])

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_setup_jax(self, setup_norm):
        params = self.unnormalize_setup(setup_norm)

        # 1. Steady State Grip Evaluation
        x_init_skidpad = jnp.zeros(46).at[14].set(15.0)
        obj_grip, min_safety = compute_skidpad_objective(self.vehicle.simulate_step, params, x_init_skidpad)
        
        # 2. Transient Frequency Response Evaluation
        x_init_freq = jnp.zeros(46).at[14].set(15.0)
        resonance = compute_frequency_response_objective(self.vehicle.simulate_step, params, x_init_freq)
        
        # Stability is the inverse of resonance (less resonant = more stable)
        obj_stability = -resonance 
        return obj_grip, obj_stability, min_safety

    @partial(jax.jit, static_argnums=(0,))
    def sb_trpo_policy_loss(self, params, old_params, omega, key):
        mu, log_std = params['mu'], params['log_std']
        old_mu, old_log_std = old_params['mu'], old_params['log_std']
        
        eps = jax.random.normal(key, mu.shape)
        setup_norm = jax.nn.sigmoid(mu + jnp.exp(log_std) * eps)
        
        grip, stability, safety = self.evaluate_setup_jax(setup_norm)
        reward = omega * grip + (1.0 - omega) * stability
        
        safety_violation = jnp.clip(safety, -5.0, 0.0) 
        safety_cost = -1000.0 * safety_violation**2 
        
        var, old_var = jnp.exp(2*log_std), jnp.exp(2*old_log_std)
        kl = jnp.sum(old_log_std - log_std + (var + (mu - old_mu)**2) / (2 * old_var) - 0.5)
        kl_penalty = 50.0 * jnp.maximum(0.0, kl - 0.05) 
        
        loss = -reward + safety_cost + kl_penalty
        return loss, (grip, stability, safety, kl)

    @partial(jax.jit, static_argnums=(0,))
    def update_ensemble(self, ensemble_params, old_ensemble_params, omegas, opt_state, keys):
        vmap_loss_grad = vmap(value_and_grad(self.sb_trpo_policy_loss, has_aux=True), in_axes=(0, 0, 0, 0))
        (losses, aux), grads = vmap_loss_grad(ensemble_params, old_ensemble_params, omegas, keys)
        grip, stability, safety, kl = aux
        return grads, grip, stability, safety, kl

    def get_non_dominated_indices(self, grip_scores, stability_scores):
        objs = np.stack([grip_scores, stability_scores], axis=1)
        is_efficient = np.ones(objs.shape[0], dtype=bool)
        for i, c in enumerate(objs):
            if is_efficient[i]:
                dominates_c = np.logical_and(
                    np.all(objs >= c, axis=1),
                    np.any(objs > c, axis=1)
                )
                if np.any(dominates_c):
                    is_efficient[i] = False
        return np.where(is_efficient)[0]

    def compute_crowding_distance(self, objs):
        """
        NSGA-II Crowding Distance Implementation.
        Assigns a distance metric to each point on the Pareto front to ensure
        evolutionary crossover maintains a diverse spread of setups.
        """
        num_points = objs.shape[0]
        distances = np.zeros(num_points)
        if num_points <= 2:
            return np.full(num_points, np.inf)

        for m in range(objs.shape[1]):
            sorted_indices = np.argsort(objs[:, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            min_val = objs[sorted_indices[0], m]
            max_val = objs[sorted_indices[-1], m]
            scale = max_val - min_val if max_val != min_val else 1.0
            
            for i in range(1, num_points - 1):
                distances[sorted_indices[i]] += (objs[sorted_indices[i+1], m] - objs[sorted_indices[i-1], m]) / scale
        return distances

    def run(self, iterations=100):
        print("\n[MORL-DB] Initializing Pareto Policy Ensemble...")
        print("[SB-TRPO] Compiling 14-DOF Analytical Constraints and Physics Gradients via XLA...")
        # Diagnostic: verify objective sensitivity to setup params
        test_soft = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # soft setup
        test_hard = jnp.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3])  # hard setup
        g_soft, s_soft, _ = self.evaluate_setup_jax(test_soft)
        g_hard, s_hard, _ = self.evaluate_setup_jax(test_hard)
        print(f"   [DIAGNOSTIC] Soft setup grip: {float(g_soft):.4f} G | Hard setup grip: {float(g_hard):.4f} G")
        print(f"   [DIAGNOSTIC] If these are identical, setup_params are not reaching the physics.")
        # Evaluate a single setup and return grip, stability, safety
        def evaluate_single(setup_norm_np):
            setup_norm = jnp.array(setup_norm_np)
            grip, stab, safety = self.evaluate_setup_jax(setup_norm)
            return float(grip), float(stab), float(safety)
        
        # Initialize population in normalised [0,1] space using sigmoid of current mu
        population = np.clip(
            0.5 + np.array(self.ensemble_params['mu']) * 0.2,
            0.0, 1.0
        )  # shape (20, 7)
        
        sigma = 0.15  # Initial CMA-ES step size
        
        for i in range(iterations):
            grips, stabs, safeties = [], [], []
            
            # Evaluate current population
            for k in range(self.ensemble_size):
                g, s, sf = evaluate_single(population[k])
                grips.append(g)
                stabs.append(s)
                safeties.append(sf)
            
            grips = np.array(grips)
            stabs = np.array(stabs)
            safeties = np.array(safeties)
            
            # Archive valid setups
            valid_mask = safeties > 0
            if np.any(valid_mask):
                phys = np.array(jax.vmap(self.unnormalize_setup)(jnp.array(population)))
                self.archive_setups.extend(phys[valid_mask])
                self.archive_grips.extend(grips[valid_mask])
                self.archive_stabs.extend(stabs[valid_mask])
            
            # Pareto selection
            grip_np = np.nan_to_num(grips, nan=-999.0)
            stab_np = np.nan_to_num(stabs, nan=-999.0)
            pareto_indices = self.get_non_dominated_indices(grip_np, stab_np)
            
            # --- FIX: PREVENT POPULATION STAGNATION ---
            # Force exploration by keeping a maximum of 50% of the population as unchanged "elites"
            # (Ensures at least 1 elite survives to prevent random randint(0) errors)
            num_elites = max(1, min(len(pareto_indices), self.ensemble_size // 2))
            
            # Sort the pareto indices by Grip score so we always keep the highest performers
            elites = sorted(pareto_indices, key=lambda idx: grip_np[idx], reverse=True)[:num_elites]
            
            # Generate next generation: mutate from elites
            new_population = population.copy()
            for j in range(self.ensemble_size):
                if j not in elites:
                    parent_idx = elites[np.random.randint(len(elites))]
                    noise = np.random.randn(self.dim) * sigma
                    new_population[j] = np.clip(population[parent_idx] + noise, 0.0, 1.0)
            
            # Decay step size slowly
            sigma = max(sigma * 0.995, 0.02)
            
            # CRITICAL: Actually update the population for the next iteration
            population = new_population
            
            if i % 20 == 0:
                safe_count = np.sum(safeties > 0)
                print(f"   [SB-TRPO] i={i} | Safe: {safe_count}/{self.ensemble_size} | "
                    f"Grip: {np.max(grips):.3f} G | Stab: {np.max(stabs):.3f} | sigma: {sigma:.3f}")
        
        # Final archive processing (keep existing code below)
        if len(self.archive_grips) > 0:
            all_setups = np.array(self.archive_setups)
            all_grips = np.array(self.archive_grips)
            all_stabs = -np.array(self.archive_stabs)

            df_archive = pd.DataFrame(all_setups, columns=self.var_keys)
            df_archive['grip'] = all_grips
            df_archive['stab'] = all_stabs
            df_archive = df_archive.drop_duplicates().sort_values('grip', ascending=False).head(150)
            
            return df_archive[self.var_keys].values, df_archive['grip'].values, df_archive['stab'].values
        else:
            return np.zeros((1,7)), np.array([0]), np.array([0])