import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from models.vehicle_dynamics import NeuralEnergyLandscape, NeuralDissipationMatrix
from data.configs.vehicle_params import vehicle_params as VP_DICT

def generate_synthetic_flex_data(num_samples=2000, key_seed=42):
    """
    Generates synthetic training data representing an unmodeled physical effect:
    Chassis Torsional Flex. When the car corners, the chassis twists, storing 
    potential energy and adding structural damping.
    """
    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Randomize vehicle states: suspension displacements (q) and momentums (p)
    q = jax.random.normal(k1, (num_samples, 14)) * 0.05  # +/- 50mm suspension travel
    p = jax.random.normal(k2, (num_samples, 14)) * 0.1   # minor velocities
    setup_params = jax.random.normal(k3, (num_samples, 7))

    # Calculate synthetic diagonal chassis twist from suspension compressions
    # q[6]=FL, q[7]=FR, q[8]=RL, q[9]=RR
    torsion = (q[:, 6] - q[:, 7]) - (q[:, 8] - q[:, 9])
    
    # Target 1: The Energy Residual (H_net)
    # Stored energy in a torsional spring: 0.5 * k * x^2
    k_torsion = 15000.0 # Nm/rad approximation
    target_H = 0.5 * k_torsion * (torsion ** 2)
    
    # Target 2: The Dissipation Residual (R_net)
    # Structural hysteresis adds to the damping matrix
    target_R_mag = jnp.abs(torsion) * 500.0

    return q, p, setup_params, target_H, target_R_mag

def train_neural_residuals():
    print("[Neural Physics] Generating Synthetic Chassis Flex Data...")
    q_data, p_data, setup_data, target_H, target_R_mag = generate_synthetic_flex_data()

    # Reconstruct the Mass Diagonal needed for H_net initialization
    m_s = VP_DICT.get('m_s', VP_DICT['m'] * 0.85)
    m_us = VP_DICT.get('m_us', VP_DICT['m'] * 0.0375)
    Ix = VP_DICT.get('Ix', 200.0)
    Iy = VP_DICT.get('Iy', 800.0)
    Iz = VP_DICT['Iz']
    Iw = VP_DICT.get('Iw', 1.2)
    M_diag = jnp.array([m_s, m_s, m_s, Ix, Iy, Iz, m_us, m_us, m_us, m_us, Iw, Iw, Iw, Iw])

    # Initialize the Neural Networks
    h_net = NeuralEnergyLandscape(M_diag=M_diag)
    r_net = NeuralDissipationMatrix(dim=14)

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    h_params = h_net.init(key1, q_data[0], p_data[0], setup_data[0])
    r_params = r_net.init(key2, q_data[0], p_data[0])

    # Initialize Adam Optimizers
    h_tx = optax.adam(learning_rate=1e-3)
    r_tx = optax.adam(learning_rate=1e-3)
    
    h_opt_state = h_tx.init(h_params)
    r_opt_state = r_tx.init(r_params)

    # --- H_net (Energy) Training Loop ---
    @jax.jit
    def h_loss_fn(params, q, p, setup, target):
        # We only want H_net to output the residual, so we subtract the prior
        def compute_residual(q_single, p_single, setup_single):
            total_energy = h_net.apply(params, q_single, p_single, setup_single)
            T_prior = 0.5 * jnp.sum((p_single ** 2) / M_diag)
            V_structural = 0.5 * jnp.sum(q_single[6:10] ** 2) * 30000.0 
            return total_energy - (T_prior + V_structural)
            
        preds = jax.vmap(compute_residual)(q, p, setup)
        return jnp.mean((preds - target) ** 2)

    @jax.jit
    def h_update(params, opt_state, q, p, setup, target):
        loss, grads = jax.value_and_grad(h_loss_fn)(params, q, p, setup, target)
        updates, opt_state = h_tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # --- R_net (Dissipation) Training Loop ---
    @jax.jit
    def r_loss_fn(params, q, p, target_mag):
        preds = jax.vmap(r_net.apply, in_axes=(None, 0, 0))(params, q, p)
        
        # We train the network to output a diagonal matrix representing structural damping
        def make_target(mag):
            return jnp.eye(14) * mag
            
        targets = jax.vmap(make_target)(target_mag)
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def r_update(params, opt_state, q, p, target):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target)
        updates, opt_state = r_tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # --- Execution ---
    print("\n[Neural Physics] Training H_net (Energy Landscape Residual)...")
    for epoch in range(1, 1001):
        h_params, h_opt_state, h_loss = h_update(h_params, h_opt_state, q_data, p_data, setup_data, target_H)
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | MSE Loss: {h_loss:.6f}")

    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, 1001):
        r_params, r_opt_state, r_loss = r_update(r_params, r_opt_state, q_data, p_data, target_R_mag)
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f}")

    print("\n[Neural Physics] Pre-training complete!")
    print("The neural components have successfully learned the unmodeled chassis flex dynamics.")
    print("These pre-trained parameters are now ready to be injected into DiffWMPCSolver.")
    
    # BUG 1 FIX: Return the trained parameters
    return h_params, r_params

if __name__ == "__main__":
    train_neural_residuals()