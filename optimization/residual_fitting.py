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
# Callers that only unpack 2 return values:
#   h_params, r_params = train_neural_residuals()          ← backward-compat
# can still access normalisation scales when needed:
#   from residual_fitting import TRAINED_H_SCALE, TRAINED_R_SCALE
# ─────────────────────────────────────────────────────────────────────────────
TRAINED_H_SCALE: float = 1.0
TRAINED_R_SCALE: float = 1.0


def generate_synthetic_flex_data(num_samples=2000, key_seed=42):
    """
    Generates synthetic training data for chassis torsional flex.
    Unchanged — the data generation is correct.
    """
    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3 = jax.random.split(key, 3)

    q            = jax.random.normal(k1, (num_samples, 14)) * 0.05
    p            = jax.random.normal(k2, (num_samples, 14)) * 0.1
    setup_params = jax.random.normal(k3, (num_samples, 8))

    torsion   = (q[:, 6] - q[:, 7]) - (q[:, 8] - q[:, 9])
    k_torsion = 15000.0
    target_H     = 0.5 * k_torsion * (torsion ** 2)
    target_R_mag = jnp.abs(torsion) * 500.0

    return q, p, setup_params, target_H, target_R_mag


def train_neural_residuals():
    print("[Neural Physics] Generating Synthetic Chassis Flex Data...")
    q_data, p_data, setup_data, target_H, target_R_mag = generate_synthetic_flex_data()

    m_s  = VP_DICT.get('m_s', VP_DICT['m'] * 0.85)
    m_us = VP_DICT.get('m_us', VP_DICT['m'] * 0.0375)
    Ix   = VP_DICT.get('Ix', 200.0)
    Iy   = VP_DICT.get('Iy', 800.0)
    Iz   = VP_DICT['Iz']
    Iw   = VP_DICT.get('Iw', 1.2)
    M_diag = jnp.array([m_s, m_s, m_s, Ix, Iy, Iz,
                         m_us, m_us, m_us, m_us, Iw, Iw, Iw, Iw])

    h_net = NeuralEnergyLandscape(M_diag=M_diag)
    r_net = NeuralDissipationMatrix(dim=14)

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    h_params = h_net.init(key1, q_data[0], p_data[0], setup_data[0])
    r_params = r_net.init(key2, q_data[0], p_data[0])

    # ── Target normalisation (unchanged) ─────────────────────────────────────
    h_scale = float(jnp.std(target_H) + 1e-6)
    r_scale = float(jnp.std(target_R_mag) + 1e-6)
    target_H_norm     = target_H     / h_scale
    target_R_mag_norm = target_R_mag / r_scale
    print(f"   [Neural Physics] H target scale: {h_scale:.2f} J  "
          f"| R target scale: {r_scale:.4f}")

    NUM_EPOCHS = 2000

    h_schedule = optax.cosine_decay_schedule(init_value=1e-3,
                                              decay_steps=NUM_EPOCHS,
                                              alpha=0.01)
    r_schedule = optax.cosine_decay_schedule(init_value=1e-3,
                                              decay_steps=NUM_EPOCHS,
                                              alpha=0.01)

    # AdamW with weight decay — directly penalises ‖∂H/∂q‖ via weight norm
    h_tx = optax.adamw(learning_rate=h_schedule, weight_decay=1e-4)
    r_tx = optax.adamw(learning_rate=r_schedule, weight_decay=1e-4)

    h_opt_state = h_tx.init(h_params)
    r_opt_state = r_tx.init(r_params)

    # ─────────────────────────────────────────────────────────────────────────
    # CURRICULUM PASSIVITY LAMBDA — CORE FIX
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Problem with fixed λ=1000 from epoch 1 (the previous value):
    # ─────────────────────────────────────────────────────────────
    # At epoch 1 the MSE loss is O(1) (random initialisation).
    # The passivity gradient is O(λ) = O(1000).
    # The optimiser sees a gradient that is 3–4 orders of magnitude larger
    # in the passivity direction than the MSE direction.  It immediately
    # annihilates the residual weights to produce H_residual ≈ 0 everywhere
    # — this trivially satisfies passivity (zero gradient = zero energy rate)
    # but means the network NEVER learns the actual chassis-flex energy
    # landscape.  The resulting MSE plateau at ~17.28 (observed in logs) is
    # the textbook signature: the network converges to the trivial zero-residual
    # solution rather than the target energy function.
    #
    # Why the trivial solution satisfies passivity but is wrong:
    #   If H_residual = 0 everywhere:
    #     ∂H/∂q = 0  →  energy_rate = ∂H/∂q · v = 0  →  passivity satisfied ✓
    #     residual = 0 ≠ target (torsional spring energy)  →  MSE not satisfied ✗
    #   The loss = MSE(0) + λ×0 = MSE(0), which is large but gradient is zero
    #   in the passivity direction, so passivity penalty stops improving and
    #   MSE plateaus.
    #
    # Fix: CURRICULUM SCHEDULING — quadratic ramp from λ_min=1 to λ_max=1000
    # over the FIRST HALF of training, then hold λ=1000 for the second half.
    #
    # Phase 1 (epochs 1 → NUM_EPOCHS/2):
    #   λ ramps: 1 → 250 → 1000  (quadratic: t² where t = epoch/half_epochs)
    #   The network primarily fits the MSE target (learns the energy shape).
    #   Late Phase 1: λ is large enough that aggressive energy injection is
    #   penalised, but not so large that it immediately destroys the residual.
    #
    # Phase 2 (epochs NUM_EPOCHS/2+1 → NUM_EPOCHS):
    #   λ = 1000.0 held constant.
    #   The network is now near a good MSE solution — small weight adjustments
    #   eliminate energy injection WITHOUT blowing up the MSE because the
    #   network is already close to the true passive energy landscape.
    #
    # Quadratic ramp formula:
    #   t = min(epoch / (NUM_EPOCHS * 0.5), 1.0)   ∈ [0, 1]
    #   λ(t) = λ_min + (λ_max − λ_min) × t²
    #
    # Sample values:
    #   epoch  1    → t ≈ 0.001 → λ ≈   1.0
    #   epoch 200   → t = 0.20  → λ ≈  41.0
    #   epoch 500   → t = 0.50  → λ = 250.3
    #   epoch 800   → t = 0.80  → λ = 640.0
    #   epoch 1000  → t = 1.00  → λ = 1000.0
    #   epoch 1001+ → t = 1.00  → λ = 1000.0  (held)
    #
    # Implementation note on jit:
    #   passivity_lambda is passed as a JAX scalar jnp.float32 into h_loss_fn.
    #   JAX jit traces through the VALUE dynamically — retracing only occurs on
    #   shape or dtype changes, not float value changes.  The first jit call
    #   produces one traced graph that works for all λ values from 1 to 1000.
    #   No performance penalty vs the fixed-lambda version.
    # ─────────────────────────────────────────────────────────────────────────
    PASSIVITY_LAMBDA_MIN = 1.0
    PASSIVITY_LAMBDA_MAX = 1000.0
    CURRICULUM_HALF      = float(NUM_EPOCHS) * 0.5   # 1000.0

    @jax.jit
    def h_loss_fn(params, q, p, setup, target_norm, passivity_lambda):
        """
        MSE + curriculum passivity loss.
        passivity_lambda : JAX scalar — dynamic, no retracing.
        """
        def compute_sample(q_s, p_s, setup_s, target_s):
            # ── Normalised energy residual ────────────────────────────────
            total_energy     = h_net.apply(params, q_s, p_s, setup_s)
            T_prior          = 0.5 * jnp.sum((p_s ** 2) / M_diag)
            V_structural     = 0.5 * jnp.sum(q_s[6:10] ** 2) * 30000.0
            residual_norm    = (total_energy - (T_prior + V_structural)) / h_scale
            mse_sample       = (residual_norm - target_s) ** 2

            # ── Passivity constraint ──────────────────────────────────────
            # Condition: ∂H_net/∂q · (p/M) ≤ 0  (no energy injection)
            dH_dq       = jax.grad(
                lambda q_: h_net.apply(params, q_, p_s, setup_s)
            )(q_s)
            v           = p_s / M_diag
            energy_rate = jnp.dot(dH_dq, v)
            # Only penalise positive (injecting) energy rates;
            # negative rates (dissipative) are free.
            passivity_violation = jax.nn.relu(energy_rate) * passivity_lambda

            return mse_sample, passivity_violation

        mse_samples, passivity_samples = jax.vmap(
            compute_sample)(q, p, setup, target_norm)
        return jnp.mean(mse_samples) + jnp.mean(passivity_samples)

    @jax.jit
    def h_update(params, opt_state, q, p, setup, target_norm, passivity_lambda):
        """Gradient step with curriculum lambda threaded through."""
        loss, grads = jax.value_and_grad(h_loss_fn)(
            params, q, p, setup, target_norm, passivity_lambda
        )
        # AdamW requires params as third argument to apply weight decay correctly
        updates, new_opt_state = h_tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # R_net (Dissipation) Training — unchanged
    # ─────────────────────────────────────────────────────────────────────────

    @jax.jit
    def r_loss_fn(params, q, p, target_mag_norm):
        preds = jax.vmap(r_net.apply, in_axes=(None, 0, 0))(params, q, p)
        targets = jax.vmap(lambda mag: jnp.eye(14) * mag)(target_mag_norm)
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def r_update(params, opt_state, q, p, target_mag_norm):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target_mag_norm)
        updates, new_opt_state = r_tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # Training execution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Neural Physics] Training H_net (Energy Landscape Residual)...")
    print(f"  [Config] Epochs: {NUM_EPOCHS} | "
          f"PASSIVITY_LAMBDA: curriculum {PASSIVITY_LAMBDA_MIN:.0f}→{PASSIVITY_LAMBDA_MAX:.0f} "
          f"(quadratic ramp over first {int(CURRICULUM_HALF)} epochs, held thereafter) | "
          f"Weight decay: 1e-4 (AdamW) | LR: cosine 1e-3→1e-5")
    print("  [Physics] Phase 1 (ep 1→1000): λ ramps 1→1000 — network learns shape FIRST.")
    print("  [Physics] Phase 2 (ep 1001→2000): λ=1000 held — passivity locked in.")
    print("  [Physics] This prevents the trivial zero-residual solution seen at fixed λ=1000.")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Curriculum ramp: Python float arithmetic, no JAX retracing
        t            = min(float(epoch) / CURRICULUM_HALF, 1.0)
        lambda_val   = PASSIVITY_LAMBDA_MIN + (PASSIVITY_LAMBDA_MAX - PASSIVITY_LAMBDA_MIN) * (t ** 2)
        p_lambda     = jnp.array(lambda_val, dtype=jnp.float32)

        h_params, h_opt_state, h_loss = h_update(
            h_params, h_opt_state,
            q_data, p_data, setup_data, target_H_norm,
            p_lambda,
        )
        if epoch % 200 == 0:
            phase = "Phase 1 (ramp)" if epoch <= int(CURRICULUM_HALF) else "Phase 2 (hold)"
            print(f"  Epoch {epoch:4d} | Loss: {h_loss:.6f} | "
                  f"λ={float(p_lambda):7.1f} | lr: {float(h_schedule(epoch)):.2e} | {phase}")

    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        r_params, r_opt_state, r_loss = r_update(
            r_params, r_opt_state,
            q_data, p_data, target_R_mag_norm,
        )
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | "
                  f"lr: {float(r_schedule(epoch)):.2e}")

    print("\n[Neural Physics] Pre-training complete!")
    print(f"Scale factors:  h_scale={h_scale:.2f} J  |  r_scale={r_scale:.4f}")
    print("Access via: from residual_fitting import TRAINED_H_SCALE, TRAINED_R_SCALE")

    global TRAINED_H_SCALE, TRAINED_R_SCALE
    TRAINED_H_SCALE = h_scale
    TRAINED_R_SCALE = r_scale

    # ── BUG 3 FIX (retained): save to correct models/ directory ─────────────
    _optimization_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root     = os.path.dirname(_optimization_dir)
    _model_dir        = os.path.join(_project_root, 'models')
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

    return h_params, r_params


if __name__ == "__main__":
    train_neural_residuals()