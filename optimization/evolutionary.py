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

class MORL_SB_TRPO_Optimizer:
    def __init__(self, ensemble_size=20, dim=7, rng_seed=42):
        self.dim = dim
        self.ensemble_size = ensemble_size
        self.var_keys = ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', 'h_cg']
        
        self.raw_bounds = jnp.array([
            [15000., 15000., 0.,    0.,    1000., 1000., 0.25],
            [60000., 60000., 2000., 1500., 5000., 5000., 0.35]
        ])
        
        self.vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        self.key = jax.random.PRNGKey(rng_seed)
        
        self.omegas = jnp.linspace(0.0, 1.0, self.ensemble_size)
        
        k1, k2 = jax.random.split(self.key)
        self.ensemble_params = {
            'mu': jax.random.uniform(k1, (self.ensemble_size, self.dim), minval=-0.5, maxval=0.5),
            'log_std': jnp.full((self.ensemble_size, self.dim), -1.0) 
        }
        
        # --- NEW: ARCHIVE TO STORE HISTORY FOR THE DASHBOARD ---
        self.archive_setups = []
        self.archive_grips = []
        self.archive_stabs = []

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_setup(self, x_norm):
        return self.raw_bounds[0] + x_norm * (self.raw_bounds[1] - self.raw_bounds[0])

    @partial(jax.jit, static_argnums=(0,))
    def simulate_slalom_jax(self, setup_norm):
        params = self.unnormalize_setup(setup_norm)
        dt, T_max = 0.005, 3.0
        steps = int(T_max / dt)
        x_init = jnp.zeros(17).at[3].set(20.0)
        
        def step_fn(x, t):
            steer = jnp.where(t > 0.2, 0.15 * jnp.sin(2.0 * jnp.pi * 1.0 * t), 0.0)
            u = jnp.array([steer, 0.0])
            x_next = self.vehicle.simulate_step(x, u, params, dt)
            lat_g = jnp.abs(x_next[3] * x_next[5] / 9.81)
            yaw_rate = jnp.abs(x_next[5])
            safety_margin = 5.0 - yaw_rate 
            return x_next, (lat_g, yaw_rate, safety_margin)

        t_array = jnp.linspace(0, T_max, steps)
        _, (lat_gs, yaw_rates, safety_margins) = jax.lax.scan(step_fn, x_init, t_array)
        
        obj_grip = jnp.mean(lat_gs)
        obj_stability = -jnp.max(yaw_rates) 
        min_safety = jnp.min(safety_margins) 
        return obj_grip, obj_stability, min_safety

    @partial(jax.jit, static_argnums=(0,))
    def sb_trpo_policy_loss(self, params, old_params, omega, key):
        mu, log_std = params['mu'], params['log_std']
        old_mu, old_log_std = old_params['mu'], old_params['log_std']
        
        eps = jax.random.normal(key, mu.shape)
        setup_norm = jax.nn.sigmoid(mu + jnp.exp(log_std) * eps)
        
        grip, stability, safety = self.simulate_slalom_jax(setup_norm)
        reward = omega * grip + (1.0 - omega) * stability
        
        safety_violation = jnp.minimum(0.0, safety)
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

    def run(self, iterations=100):
        print("\n[MORL-DB] Initializing Pareto Policy Ensemble...")
        print("[SB-TRPO] Compiling Exact Analytical Constraints and Physics Gradients...")
        
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(self.ensemble_params)
        old_params = self.ensemble_params
        
        for i in range(iterations):
            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, self.ensemble_size)
            
            grads, grips, stabs, safeties, kls = self.update_ensemble(
                self.ensemble_params, old_params, self.omegas, opt_state, keys
            )
            
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(self.ensemble_params, updates)
            
            # --- NEW: RECORD VALID SETUPS TO ARCHIVE ---
            if i > 10 and i % 5 == 0:
                current_setups_norm = jax.nn.sigmoid(self.ensemble_params['mu'])
                current_setups_phys = np.array(jax.vmap(self.unnormalize_setup)(current_setups_norm))
                valid_mask = np.array(safeties) > 0
                if np.any(valid_mask):
                    self.archive_setups.extend(current_setups_phys[valid_mask])
                    self.archive_grips.extend(np.array(grips)[valid_mask])
                    self.archive_stabs.extend(np.array(stabs)[valid_mask])

            if i % 10 == 0 and i > 0:
                grip_np, stab_np = np.array(grips), np.array(stabs)
                pareto_indices = self.get_non_dominated_indices(grip_np, stab_np)
                
                if len(pareto_indices) > 0 and len(pareto_indices) < self.ensemble_size:
                    for j in range(self.ensemble_size):
                        if j not in pareto_indices:
                            elite_idx = np.random.choice(pareto_indices)
                            mu_copy = np.array(new_params['mu'])
                            log_std_copy = np.array(new_params['log_std'])
                            mu_copy[j] = mu_copy[elite_idx]
                            log_std_copy[j] = np.full(self.dim, -0.5) 
                            new_params['mu'] = jnp.array(mu_copy)
                            new_params['log_std'] = jnp.array(log_std_copy)
                            omegas_np = np.array(self.omegas)
                            omegas_np[j] = np.random.uniform(0.0, 1.0)
                            self.omegas = jnp.array(omegas_np)
                            
            old_params = self.ensemble_params
            self.ensemble_params = new_params
            
            if i % 20 == 0:
                safe_count = np.sum(np.array(safeties) > 0)
                print(f"   [SB-TRPO] Active Policies Safe: {safe_count}/{self.ensemble_size} | Max Grip: {np.max(grips):.3f} G | Max Stab: {np.max(stabs):.3f}")

        # Post-Processing: Return the top 100 setups from the historical archive
        if len(self.archive_grips) > 0:
            all_setups = np.array(self.archive_setups)
            all_grips = np.array(self.archive_grips)
            all_stabs = -np.array(self.archive_stabs) # Invert for UI

            # Drop duplicates and sort by Grip
            df_archive = pd.DataFrame(all_setups, columns=self.var_keys)
            df_archive['grip'] = all_grips
            df_archive['stab'] = all_stabs
            df_archive = df_archive.drop_duplicates().sort_values('grip', ascending=False).head(150)
            
            return df_archive[self.var_keys].values, df_archive['grip'].values, df_archive['stab'].values
            
        else:
            return np.zeros((1,7)), np.array([0]), np.array([0])