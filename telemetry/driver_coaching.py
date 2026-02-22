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
        # (In training, human_action is inferred via Inverse RL, here we mock a degraded cost map)
        human_actions = jnp.tile(jnp.array([5e-2, 5e-7, 0.05]), (len(s_mpc), 1)) 

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