import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import flax.linen as nn
import pandas as pd
import numpy as np

# --- FLAX NEURAL NETWORKS (Ghost Car Actor-Critic Framework) ---
class DiffWMPCActor(nn.Module):
    """
    Continuous Policy Network (The Actor).
    Maps the current Vehicle State + Human Error to an Active MPC Cost Surface.
    Outputs the weights that parameterize the Diff-WMPC Solver: 
    [w_steer, w_accel, w_mu_uncertainty]
    """
    @nn.compact
    def __call__(self, state):
        x = nn.Dense(128)(state)
        x = nn.swish(x)
        x = nn.Dense(128)(x)
        x = nn.swish(x)
        
        # Output means for the stochastic Gaussian policy
        mean = nn.Dense(3)(x)
        log_std = nn.Dense(3)(x)
        log_std = jnp.clip(log_std, -20, 2) 
        
        # We use softplus to ensure MPC cost weights are strictly positive
        deterministic_action = jax.nn.softplus(mean)
        return deterministic_action, log_std

class SoftQNetwork(nn.Module):
    """
    Soft Q-Network (The Critic).
    Evaluates the Expected Future Reward (Lap Time Reduction + Tube Stability)
    given the 38-D Kinematic State and the 3-D MPC Weight Action.
    """
    @nn.compact
    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)
        x = nn.Dense(256)(sa)
        x = nn.swish(x)
        x = nn.Dense(128)(x)
        x = nn.swish(x)
        q_value = nn.Dense(1)(x)
        return q_value

class GhostCarEvaluator:
    """
    State-of-the-Art Reinforcement Learning Driver & Vehicle Evaluator.
    Executes Continuous Temporal Difference (TD) analysis via JAX vmap, 
    dynamically reshaping the Diff-WMPC solver weights based on human deviation.
    """
    def __init__(self, state_dim=5, rng_seed=42):
        self.key = jax.random.PRNGKey(rng_seed)
        
        self.actor = DiffWMPCActor()
        self.critic_1 = SoftQNetwork()
        self.critic_2 = SoftQNetwork()
        
        # State: [Velocity_Error, Lateral_Error, Yaw_Error, Slip_Proxy, Base_Friction]
        dummy_state = jnp.ones((1, state_dim))
        dummy_action = jnp.ones((1, 3)) # [w_steer, w_accel, w_mu]
        
        key_a, key_c1, key_c2 = jax.random.split(self.key, 3)
        self.actor_params = self.actor.init(key_a, dummy_state)
        self.critic_1_params = self.critic_1.init(key_c1, dummy_state, dummy_action)
        self.critic_2_params = self.critic_2.init(key_c2, dummy_state, dummy_action)

    @partial(jit, static_argnums=(0,))
    def _evaluate_node_advantage(self, state, human_action, optimal_action, c1_p, c2_p):
        """Calculates Q-Value Advantage for a single spatial node."""
        q1_opt = self.critic_1.apply(c1_p, state, optimal_action)
        q2_opt = self.critic_2.apply(c2_p, state, optimal_action)
        q_opt = jnp.minimum(q1_opt, q2_opt)
        
        q1_hum = self.critic_1.apply(c1_p, state, human_action)
        q2_hum = self.critic_2.apply(c2_p, state, human_action)
        q_hum = jnp.minimum(q1_hum, q2_hum)
        
        # Advantage: How much worse was the human's implicit cost landscape?
        return (q_hum - q_opt)[0]

    @partial(jit, static_argnums=(0,))
    def _get_active_mpc_weights(self, state, actor_p):
        """Queries the Actor network to dynamically adjust the WMPC cost bounds."""
        deterministic_weights, _ = self.actor.apply(actor_p, state)
        return deterministic_weights

    def infer_driver_intent_irl(self, state_matrix, optimal_actions, iterations=100):
        """
        Maximum Entropy Inverse RL (MaxEnt IRL).
        Reverse-engineers the human driver's implicit cost weights by finding the 
        MPC parameters that mathematically explain their observed telemetry errors.
        """
        print("[Ghost-Car AI] Executing MaxEnt IRL: Inferring driver's implicit MPC cost weights...")
        
        # Start with the optimal weights as the prior guess
        inferred_actions = jnp.array(optimal_actions)
        
        import optax
        optimizer = optax.adam(learning_rate=0.05)
        opt_state = optimizer.init(inferred_actions)
        
        @jax.jit
        def irl_step(actions, opt_st):
            def loss_fn(act):
                # Enforce strictly positive weights
                valid_act = jax.nn.softplus(act)
                
                # 1. Critic Evaluation of the inferred human actions vs optimal
                q_hum = jax.vmap(self.critic_1.apply, in_axes=(None, 0, 0))(self.critic_1_params, state_matrix, valid_act)
                q_opt = jax.vmap(self.critic_1.apply, in_axes=(None, 0, 0))(self.critic_1_params, state_matrix, optimal_actions)
                
                # 2. Observed Physical Severity (The true empirical disadvantage)
                # state_matrix: [err_v, err_n, err_yaw, ...]
                # Heavily penalize lateral deviation (missing apex) and velocity deviation (overshooting)
                physical_disadvantage = - (0.5 * state_matrix[:, 0]**2 + 2.0 * state_matrix[:, 1]**2)
                physical_disadvantage = jnp.expand_dims(physical_disadvantage, axis=-1)
                
                # 3. IRL Objective: The predicted Q-value difference (Advantage) must match the physical reality.
                # We also add a MaxEnt regularization term (L2 on the actions) to prevent weight explosion.
                advantage_loss = jnp.mean(((q_hum - q_opt) - physical_disadvantage)**2)
                entropy_reg = 0.01 * jnp.mean(valid_act**2)
                
                return advantage_loss + entropy_reg
                
            loss, grads = jax.value_and_grad(loss_fn)(actions)
            updates, opt_st = optimizer.update(grads, opt_st)
            return optax.apply_updates(actions, updates), opt_st, loss

        for _ in range(iterations):
            inferred_actions, opt_state, _ = irl_step(inferred_actions, opt_state)
            
        return jax.nn.softplus(inferred_actions)

    def pre_train_critic(self, dict_stochastic_mpc, iterations=300):
        """
        Generates synthetic suboptimal rollouts to train the Q-Networks 
        before analyzing real driver telemetry.
        """
        print("[Ghost-Car AI] Pre-training SoftQNetwork Critic on synthetic perturbations...")
        
        s_mpc = dict_stochastic_mpc['s']
        v_mpc = dict_stochastic_mpc['v']
        n_mpc = dict_stochastic_mpc['n']
        
        # Synthetic errors: Late braking (velocity overshoot), Apex miss (lateral error)
        err_v = jnp.sin(s_mpc / 20.0) * 2.0 
        err_n = jnp.cos(s_mpc / 30.0) * 0.5
        
        state_matrix = jnp.stack([err_v, err_n, jnp.zeros_like(s_mpc), jnp.zeros_like(s_mpc), jnp.ones_like(s_mpc)*1.4], axis=-1)
        suboptimal_actions = jnp.tile(jnp.array([5e-2, 5e-7, 0.05]), (len(s_mpc), 1))
        
        # Target TD Reward: Estimated time lost due to synthetic velocity error
        target_q = jnp.expand_dims(err_v * -0.1, axis=-1) 
        
        import optax
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state_1 = optimizer.init(self.critic_1_params)
        
        @jax.jit
        def train_step(c1_p, opt_st):
            def loss_fn(p):
                q_preds = jax.vmap(self.critic_1.apply, in_axes=(None, 0, 0))(p, state_matrix, suboptimal_actions)
                return jnp.mean((q_preds - target_q)**2)
            
            loss, grads = jax.value_and_grad(loss_fn)(c1_p)
            updates, opt_st = optimizer.update(grads, opt_st)
            return optax.apply_updates(c1_p, updates), opt_st, loss

        for i in range(iterations):
            self.critic_1_params, opt_state_1, loss = train_step(self.critic_1_params, opt_state_1)
            
        print(f"[Ghost-Car AI] Pre-training complete. Final MSE Loss: {loss:.4f}")

    def evaluate_continuous_policy(self, df_human, dict_stochastic_mpc):
        """
        Generates the dynamic AC-MPC coaching report using fully vectorized JAX ops.
        """
        print("[Ghost-Car AI] Vector-mapping Human Policy vs. Diff-WMPC Manifold...")
        
        # Extract Ghost Car (Optimal) Manifold
        s_mpc = dict_stochastic_mpc['s']
        v_mpc = dict_stochastic_mpc['v']
        n_mpc = dict_stochastic_mpc['n']
        
        # JAX-native interpolation to synchronize human telemetry to MPC nodes
        v_hum = jnp.interp(s_mpc, jnp.array(df_human['s']), jnp.array(df_human['v']))
        n_hum = jnp.interp(s_mpc, jnp.array(df_human['s']), jnp.array(df_human['n']))
        yaw_hum = jnp.interp(s_mpc, jnp.array(df_human['s']), jnp.array(df_human['yaw']))
        
        # Compute dynamic error states for the RL Actor
        err_v = v_hum - v_mpc
        err_n = n_hum - n_mpc
        err_yaw = yaw_hum # Simplified assuming MPC yaw aligns with track tangent

        # Construct full state tensor [N_nodes, state_dim]
        # State: [Velocity_Error, Lateral_Error, Yaw_Error, Slip_Proxy, Base_Friction]
        state_matrix = jnp.stack([err_v, err_n, err_yaw, jnp.zeros_like(s_mpc), jnp.ones_like(s_mpc)*1.4], axis=-1)

        # Base MPC nominal weights vs human implicit deviation weights (proxy for evaluation)
        optimal_actions = jnp.tile(jnp.array([1e-2, 1e-7, 0.02]), (len(s_mpc), 1))
        
        # --- IRL FIX: Dynamically Infer Human Actions ---
        human_actions = self.infer_driver_intent_irl(state_matrix, optimal_actions)

        # --- JAX VMAP: Instantaneous parallel evaluation across the entire track ---
        batch_advantage_fn = vmap(self._evaluate_node_advantage, in_axes=(0, 0, 0, None, None))
        advantages = batch_advantage_fn(
            state_matrix, human_actions, optimal_actions, 
            self.critic_1_params, self.critic_2_params
        )
        
        batch_actor_fn = vmap(self._get_active_mpc_weights, in_axes=(0, None))
        active_weights = batch_actor_fn(state_matrix, self.actor_params)

        # Compile report for critical deviations
        advantages_np = np.array(advantages)
        active_weights_np = np.array(active_weights)
        
        # Filter indices where the human advantage is severely negative (bad driving error)
        critical_indices = np.where(advantages_np < -0.5)[0]
        
        report = []
        for i in critical_indices:
            w_steer_mod, w_accel_mod, w_mu_mod = active_weights_np[i]
            
            report.append({
                'S_Node': round(float(s_mpc[i]), 1),
                'Critic_Advantage': round(float(advantages_np[i]), 3),
                'Driver_Error_State': f"dV: {err_v[i]:.1f}m/s | dLat: {err_n[i]:.2f}m",
                'Ghost_Car_Correction': f"Set W_Steer: {w_steer_mod:.4f}, W_Accel: {w_accel_mod:.8f}, Mu_Uncert: {w_mu_mod:.3f}"
            })

        df_report = pd.DataFrame(report)
        print(f"[Ghost-Car AI] Evaluation Complete. {len(df_report)} Critical Interventions Flagged via JAX vmap.")
        
        # We can now feed `active_weights_np` directly back into the `DiffWMPCSolver.solve(ai_cost_map=...)`
        output_cost_map = {
            'w_steer': active_weights_np[:, 0],
            'w_accel': active_weights_np[:, 1],
            'w_mu': active_weights_np[:, 2]
        }
        
        return df_report, output_cost_map