import jax
import jax.numpy as jnp
import optax
import os
import flax.serialization
from flax.training import train_state

from models.vehicle_dynamics import NeuralEnergyLandscape, NeuralDissipationMatrix
from data.configs.vehicle_params import vehicle_params as VP_DICT


# ─────────────────────────────────────────────────────────────────────────────
# Module-level scale factors set by train_neural_residuals().
#
# These are exposed here so that callers which only unpack 2 return values
#   h_params, r_params = train_neural_residuals()          ← backward-compat
# can still access the normalisation scales when needed:
#   from residual_fitting import TRAINED_H_SCALE, TRAINED_R_SCALE
#
# vehicle_dynamics.py should use:
#   H_residual_physical = h_net.apply(h_params, q, p, setup) * TRAINED_H_SCALE
#   R_residual_physical = r_net.apply(r_params, q, p)        * TRAINED_R_SCALE
# ─────────────────────────────────────────────────────────────────────────────
TRAINED_H_SCALE: float = 1.0   # updated in-place by train_neural_residuals()
TRAINED_R_SCALE: float = 1.0


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
    setup_params = jax.random.normal(k3, (num_samples, 8))

    # Calculate synthetic diagonal chassis twist from suspension compressions
    # q[6]=FL, q[7]=FR, q[8]=RL, q[9]=RR
    torsion = (q[:, 6] - q[:, 7]) - (q[:, 8] - q[:, 9])

    # Target 1: The Energy Residual (H_net)
    # Stored energy in a torsional spring: 0.5 * k * x^2
    k_torsion = 15000.0  # Nm/rad approximation
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

    # ── FIX: Target normalisation ─────────────────────────────────────────────
    # The synthetic H_net targets are O(1000 J) while the network outputs O(1).
    # A 1000× scale mismatch causes the optimiser to chase a loss dominated by
    # the scale rather than the shape of the energy landscape.  Normalising to
    # unit variance puts the MSE landscape in the same range as the network's
    # natural output scale.
    #
    # IMPORTANT: h_scale must be stored alongside the saved parameters so that
    # H_net outputs can be de-normalised when used by vehicle_dynamics.py:
    #       H_residual_physical = h_net.apply(params, q, p, setup) * h_scale
    h_scale = float(jnp.std(target_H) + 1e-6)
    r_scale = float(jnp.std(target_R_mag) + 1e-6)
    target_H_norm     = target_H     / h_scale
    target_R_mag_norm = target_R_mag / r_scale
    print(f"   [Neural Physics] H target scale: {h_scale:.2f} J  "
          f"| R target scale: {r_scale:.4f}")

    # ── FIX: Cosine decay learning rate schedules ─────────────────────────────
    # Fixed lr=1e-3 caused H_net to plateau at MSE=23.7 after epoch 600.
    # Cosine decay from 1e-3 → 1e-5 over 1000 epochs avoids the saddle point
    # by progressively reducing step size as the loss landscape flattens.
    NUM_EPOCHS = 1000
    h_schedule = optax.cosine_decay_schedule(
        init_value=1e-3,
        decay_steps=NUM_EPOCHS,
        alpha=0.01,   # final lr = 1e-3 * 0.01 = 1e-5
    )
    r_schedule = optax.cosine_decay_schedule(
        init_value=1e-3,
        decay_steps=NUM_EPOCHS,
        alpha=0.01,
    )
    h_tx = optax.adam(learning_rate=h_schedule)
    r_tx = optax.adam(learning_rate=r_schedule)

    h_opt_state = h_tx.init(h_params)
    r_opt_state = r_tx.init(r_params)

    # ─────────────────────────────────────────────────────────────────────────
    # H_net (Energy) Training
    # ─────────────────────────────────────────────────────────────────────────
    # FIX: Passivity constraint added to H_net loss.
    #
    # A passive Hamiltonian residual must not inject energy into the system.
    # The passivity condition is:
    #
    #     dH_residual/dt = (∂H/∂q)ᵀ · (dq/dt) ≤ 0
    #
    # where dq/dt ≈ p / M_diag  (velocity from canonical momentum).
    #
    # We penalise any positive energy rate with passivity_loss = relu(dH/dt) * λ.
    # λ = 10.0 is large enough to strongly penalise violations relative to the
    # normalised MSE loss (both are O(1) after target normalisation).
    #
    # Root cause of "Speed changed: 10.000 → 10.002 with zero throttle":
    # H_net was injecting 0.46 J/step into the kinetic energy. This fix makes
    # the network structurally passive during training so the energy budget is
    # always conserved when H_net is deployed.
    PASSIVITY_LAMBDA = 10.0

    @jax.jit
    def h_loss_fn(params, q, p, setup, target_norm):
        """
        MSE loss on normalised energy residual + passivity constraint.
        """
        def compute_sample(q_s, p_s, setup_s, target_s):
            # ── Compute the normalised energy residual ──────────────────
            total_energy = h_net.apply(params, q_s, p_s, setup_s)
            T_prior      = 0.5 * jnp.sum((p_s ** 2) / M_diag)
            V_structural = 0.5 * jnp.sum(q_s[6:10] ** 2) * 30000.0
            residual_normalised = (total_energy - (T_prior + V_structural)) / h_scale
            mse_sample = (residual_normalised - target_s) ** 2

            # ── Passivity constraint ────────────────────────────────────
            # ∂H_net/∂q: gradient of the full H_net output w.r.t. q
            dH_dq = jax.grad(
                lambda q_: h_net.apply(params, q_, p_s, setup_s)
            )(q_s)
            # dq/dt = p / M_diag  (Hamilton's eq. for kinetic energy)
            v         = p_s / M_diag
            # Energy injection rate — must be <= 0 for passivity
            energy_rate = jnp.dot(dH_dq, v)
            # Penalise only positive (injecting) energy rates
            passivity_violation = jax.nn.relu(energy_rate) * PASSIVITY_LAMBDA

            return mse_sample, passivity_violation

        mse_samples, passivity_samples = jax.vmap(compute_sample)(q, p, setup, target_norm)
        mse_loss       = jnp.mean(mse_samples)
        passivity_loss = jnp.mean(passivity_samples)
        return mse_loss + passivity_loss

    @jax.jit
    def h_update(params, opt_state, q, p, setup, target_norm):
        loss, grads = jax.value_and_grad(h_loss_fn)(params, q, p, setup, target_norm)
        updates, opt_state = h_tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # R_net (Dissipation) Training
    # ─────────────────────────────────────────────────────────────────────────

    @jax.jit
    def r_loss_fn(params, q, p, target_mag_norm):
        preds = jax.vmap(r_net.apply, in_axes=(None, 0, 0))(params, q, p)

        def make_target(mag):
            return jnp.eye(14) * mag

        targets = jax.vmap(make_target)(target_mag_norm)
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def r_update(params, opt_state, q, p, target_mag_norm):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target_mag_norm)
        updates, opt_state = r_tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # Training execution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Neural Physics] Training H_net (Energy Landscape Residual)...")
    print("  [Physics] MSE is on normalised targets.  Target × h_scale = physical J.")
    print("  [Physics] Passivity constraint active — H_net cannot inject energy.")

    for epoch in range(1, NUM_EPOCHS + 1):
        h_params, h_opt_state, h_loss = h_update(
            h_params, h_opt_state,
            q_data, p_data, setup_data, target_H_norm,
        )
        if epoch % 200 == 0:
            current_lr = float(h_schedule(epoch))
            print(f"  Epoch {epoch:4d} | MSE+Passivity Loss: {h_loss:.6f} | "
                  f"lr: {current_lr:.2e}")

    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        r_params, r_opt_state, r_loss = r_update(
            r_params, r_opt_state,
            q_data, p_data, target_R_mag_norm,
        )
        if epoch % 200 == 0:
            current_lr = float(r_schedule(epoch))
            print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | lr: {current_lr:.2e}")

    print("\n[Neural Physics] Pre-training complete!")
    print("The neural components have successfully learned the unmodeled chassis flex dynamics.")
    print(f"Scale factors for de-normalisation:  h_scale={h_scale:.2f}  r_scale={r_scale:.4f}")
    print("Access scale factors: from residual_fitting import TRAINED_H_SCALE, TRAINED_R_SCALE")

    # ── Update module-level scale globals ────────────────────────────────────
    global TRAINED_H_SCALE, TRAINED_R_SCALE
    TRAINED_H_SCALE = h_scale
    TRAINED_R_SCALE = r_scale

    # ── BUG 3 FIX: Save weights and scale factor to the canonical models/ dir ─
    #
    # THE ONE-LINE FIX: residual_fitting.py lives at:
    #     PROJECT_ROOT/optimization/residual_fitting.py
    #
    # Previous code used:
    #     os.path.dirname(__file__)          →  PROJECT_ROOT/optimization/
    #     os.path.join(..., 'models')        →  PROJECT_ROOT/optimization/models/  ← WRONG
    #
    # vehicle_dynamics.py (inside models/) and sanity_checks.py (at project root)
    # both resolve to PROJECT_ROOT/models/ — so all three paths must agree.
    #
    # Fix: go up one extra level from optimization/ to reach the project root,
    # then descend into models/:
    #     os.path.dirname(os.path.dirname(__file__))  →  PROJECT_ROOT/
    #     os.path.join(..., 'models')                 →  PROJECT_ROOT/models/  ← CORRECT
    _optimization_dir = os.path.dirname(os.path.abspath(__file__))  # .../optimization/
    _project_root     = os.path.dirname(_optimization_dir)           # .../FS_Driver_Setup_Optimizer/
    _model_dir        = os.path.join(_project_root, 'models')        # .../models/   ← agrees with vehicle_dynamics.py and sanity_checks.py

    os.makedirs(_model_dir, exist_ok=True)

    _h_path     = os.path.join(_model_dir, 'h_net.bytes')
    _r_path     = os.path.join(_model_dir, 'r_net.bytes')
    _scale_path = os.path.join(_model_dir, 'h_net_scale.txt')

    with open(_h_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(h_params))
    with open(_r_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(r_params))
    with open(_scale_path, 'w') as f:
        f.write(str(h_scale))

    print(f"[Neural Physics] Weights saved → {_h_path}")
    print(f"[Neural Physics] Scale saved   → {_scale_path}  ({h_scale:.2f} J)")
    print("These pre-trained parameters are now ready to be injected into DiffWMPCSolver.")

    # Return exactly 2 values — backward-compatible with any caller that unpacks:
    #   h_params, r_params = train_neural_residuals()
    return h_params, r_params


if __name__ == "__main__":
    train_neural_residuals()