import sys
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
from functools import partial

# Import the new Differentiable Physics Engine
from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class MORL_SB_TRPO_Optimizer:
    """
    Multi-Objective Reinforcement Learning (Dominance-Based) with Safety-Biased TRPO.
    Replaces TuRBO with an ensemble of differentiable active policies.
    Calculates exact analytical gradients through the physics engine using the Reparameterization Trick.
    """
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
        
        # Initialize the Pareto Policy Ensemble
        # Each policy in the ensemble targets a different trade-off between Grip and Stability
        self.omegas = jnp.linspace(0.0, 1.0, self.ensemble_size)
        
        k1, k2 = jax.random.split(self.key)
        self.ensemble_params = {
            # Start policies near the center of the normalized [0,1] space
            'mu': jax.random.uniform(k1, (self.ensemble_size, self.dim), minval=-0.5, maxval=0.5),
            # Initial exploration variance
            'log_std': jnp.full((self.ensemble_size, self.dim), -1.0) 
        }

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_setup(self, x_norm):
        return self.raw_bounds[0] + x_norm * (self.raw_bounds[1] - self.raw_bounds[0])

    @partial(jax.jit, static_argnums=(0,))
    def simulate_slalom_jax(self, setup_norm):
        """
        Differentiable Physics Evaluation.
        Returns multiple independent objectives required for MORL-DB.
        """
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
            
            # Safety Margin: Distance to loss-of-control boundary
            safety_margin = 5.0 - yaw_rate 
            return x_next, (lat_g, yaw_rate, safety_margin)

        t_array = jnp.linspace(0, T_max, steps)
        _, (lat_gs, yaw_rates, safety_margins) = jax.lax.scan(step_fn, x_init, t_array)
        
        # MORL-DB Objectives (Both formulated to be Maximized)
        obj_grip = jnp.mean(lat_gs)
        obj_stability = -jnp.max(yaw_rates) 
        
        # Absolute worst safety violation across the simulation
        min_safety = jnp.min(safety_margins) 
        return obj_grip, obj_stability, min_safety

    @partial(jax.jit, static_argnums=(0,))
    def sb_trpo_policy_loss(self, params, old_params, omega, key):
        """
        Safety-Biased TRPO Core Update.
        Calculates exact policy gradients bounded by KL divergence and strict log-barrier safety constraints.
        """
        mu, log_std = params['mu'], params['log_std']
        old_mu, old_log_std = old_params['mu'], old_params['log_std']
        
        # 1. Reparameterization Trick (Differentiable Exploration)
        eps = jax.random.normal(key, mu.shape)
        # Bounded sigmoid mapping to guarantee setup values remain physically valid
        setup_norm = jax.nn.sigmoid(mu + jnp.exp(log_std) * eps)
        
        grip, stability, safety = self.simulate_slalom_jax(setup_norm)
        
        # 2. Scalarized MORL Reward (Guided by preference vector omega)
        reward = omega * grip + (1.0 - omega) * stability
        
        # 3. SB-TRPO Safety Log-Barrier
        # If the policy breaches safety constraints (safety < 0), penalty explodes exponentially
        safety_violation = jnp.minimum(0.0, safety)
        safety_cost = -1000.0 * safety_violation**2 
        
        # 4. TRPO KL-Divergence Constraint (Analytical for Gaussians)
        var, old_var = jnp.exp(2*log_std), jnp.exp(2*old_log_std)
        kl = jnp.sum(old_log_std - log_std + (var + (mu - old_mu)**2) / (2 * old_var) - 0.5)
        
        # Strict Trust Region threshold delta = 0.05
        kl_penalty = 50.0 * jnp.maximum(0.0, kl - 0.05) 
        
        # Minimize Negative Reward (Maximize Objective) while satisfying constraints
        loss = -reward + safety_cost + kl_penalty
        return loss, (grip, stability, safety, kl)

    @partial(jax.jit, static_argnums=(0,))
    def update_ensemble(self, ensemble_params, old_ensemble_params, omegas, opt_state, keys):
        """
        Vectorized execution of SB-TRPO across the entire population of policies simultaneously.
        """
        # Vectorize the value_and_grad function across the ensemble batch dimension
        vmap_loss_grad = vmap(value_and_grad(self.sb_trpo_policy_loss, has_aux=True), in_axes=(0, 0, 0, 0))
        
        (losses, aux), grads = vmap_loss_grad(ensemble_params, old_ensemble_params, omegas, keys)
        grip, stability, safety, kl = aux
        
        return grads, grip, stability, safety, kl

    def get_non_dominated_indices(self, grip_scores, stability_scores):
        """MORL-DB: Fast Pareto Dominance Sorting."""
        objs = np.stack([grip_scores, stability_scores], axis=1)
        is_efficient = np.ones(objs.shape[0], dtype=bool)
        for i, c in enumerate(objs):
            if is_efficient[i]:
                # Find if any other point dominates 'c'
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
            
            # Step 1: Execute JAX VMAP Policy Gradient Update
            grads, grips, stabs, safeties, kls = self.update_ensemble(
                self.ensemble_params, old_params, self.omegas, opt_state, keys
            )
            
            # Apply gradients via Optax
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(self.ensemble_params, updates)
            
            # Step 2: MORL-DB Pareto Culling (Every 10 iterations)
            if i % 10 == 0 and i > 0:
                grip_np, stab_np = np.array(grips), np.array(stabs)
                pareto_indices = self.get_non_dominated_indices(grip_np, stab_np)
                
                print(f" > Iter {i:03d} | Maintained Pareto Frontier Size: {len(pareto_indices)} / {self.ensemble_size}")
                
                # Dominance-Based Resampling: If a policy is dominated, reset it to the parameters of an elite policy
                # but randomize its target preference to force exploration of Pareto gaps.
                if len(pareto_indices) > 0 and len(pareto_indices) < self.ensemble_size:
                    for j in range(self.ensemble_size):
                        if j not in pareto_indices:
                            elite_idx = np.random.choice(pareto_indices)
                            # Copy elite weights to the dominated policy (JAX arrays are immutable, convert to numpy for routing)
                            mu_copy = np.array(new_params['mu'])
                            log_std_copy = np.array(new_params['log_std'])
                            
                            mu_copy[j] = mu_copy[elite_idx]
                            log_std_copy[j] = np.full(self.dim, -0.5) # Expand variance to force new exploration
                            
                            new_params['mu'] = jnp.array(mu_copy)
                            new_params['log_std'] = jnp.array(log_std_copy)
                            
                            # Shift the target preference vector to discover missing trade-offs
                            omegas_np = np.array(self.omegas)
                            omegas_np[j] = np.random.uniform(0.0, 1.0)
                            self.omegas = jnp.array(omegas_np)
                            
            old_params = self.ensemble_params
            self.ensemble_params = new_params
            
            if i % 20 == 0:
                safe_count = np.sum(np.array(safeties) > 0)
                print(f"   [SB-TRPO] Active Policies Safe: {safe_count}/{self.ensemble_size} | Max Grip: {np.max(grips):.3f} G | Max Stab: {np.max(stabs):.3f}")

        # Post-Processing: Extract Final Deterministic Pareto Front
        final_setups_norm = jax.nn.sigmoid(self.ensemble_params['mu'])
        final_setups_phys = np.array(jax.vmap(self.unnormalize_setup)(final_setups_norm))
        
        final_grips, final_stabs, _ = jax.vmap(self.simulate_slalom_jax)(final_setups_norm)
        pareto_idx = self.get_non_dominated_indices(np.array(final_grips), np.array(final_stabs))
        
        pareto_setups = final_setups_phys[pareto_idx]
        pareto_grips = np.array(final_grips)[pareto_idx]
        pareto_stabs = -np.array(final_stabs)[pareto_idx] # Convert back to positive overshoot for UI
        
        return pareto_setups, pareto_grips, pareto_stabs

if __name__ == "__main__":
    optimizer = MORL_SB_TRPO_Optimizer()
    pareto_setups, pareto_grips, pareto_stabs = optimizer.run(iterations=100)
    
    df = pd.DataFrame(pareto_setups, columns=optimizer.var_keys)
    df['Lat_G_Score'] = pareto_grips
    df['Stability_Overshoot'] = pareto_stabs
    
    print("\n[MORL-DB] Target-Fidelity Pareto Front Discovery Complete:")
    print(df.sort_values('Lat_G_Score', ascending=False).to_string(index=False))