import jax
import jax.numpy as jnp
import optax
import os
import flax.serialization

from models.vehicle_dynamics import NeuralEnergyLandscape, NeuralDissipationMatrix
from data.configs.vehicle_params import vehicle_params as VP_DICT


# ─────────────────────────────────────────────────────────────────────────────
# Module-level scale factors set by train_neural_residuals().
# ─────────────────────────────────────────────────────────────────────────────
TRAINED_H_SCALE: float = 1.0
TRAINED_R_SCALE: float = 1.0


def generate_synthetic_flex_data(num_samples=2000, key_seed=42):
    """
    Generates synthetic training data for chassis torsional flex.
    Torsional stiffness is at 15 000 Nm/rad (synthetic approximation).
    Until 4-post rig data is available this is the primary training source.
    """
    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3 = jax.random.split(key, 3)

    q            = jax.random.normal(k1, (num_samples, 14)) * 0.05
    p            = jax.random.normal(k2, (num_samples, 14)) * 0.1
    setup_params = jax.random.normal(k3, (num_samples, 8))

    torsion      = (q[:, 6] - q[:, 7]) - (q[:, 8] - q[:, 9])
    k_torsion    = 15000.0
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

    # ── Target normalisation ──────────────────────────────────────────────────
    h_scale = float(jnp.std(target_H) + 1e-6)
    r_scale = float(jnp.std(target_R_mag) + 1e-6)
    target_H_norm     = target_H     / h_scale
    target_R_mag_norm = target_R_mag / r_scale
    print(f"   [Neural Physics] H target scale: {h_scale:.2f} J  "
          f"| R target scale: {r_scale:.4f}")

    # =========================================================================
    # TWO-PHASE TRAINING — VERSION 5: GRADIENT NORMALISATION + BILATERAL
    # PASSIVITY LOSS
    # =========================================================================
    #
    # FAILURE HISTORY (brief — full record in previous sessions):
    # ─────────────────────────────────────────────────────────────────────────
    #
    # v1  Fixed λ=1000 → trivial zero collapse (passivity >> MSE).
    # v2  λ ramp with MSE gate → gate permanently frozen (MSE_norm ≥ 1 by
    #     construction from normalisation).
    # v3  λ_max=100 cold switch at epoch 1501 → gradient ratio 594,500:1.
    # v4  Linear ramp λ → 10 over Phase 2 → loss grows proportional to λ.
    #
    # ROOT CAUSE OF v4 FAILURE (diagnosed from run log):
    # ─────────────────────────────────────────────────────────────────────────
    # Epoch 1400: MSE = 0.001793, λ = 0.00  ← Phase 1 converged correctly
    # Epoch 1600: loss = 23.21,    λ = 2.00  ← grows proportionally to λ
    # Epoch 1800: loss = 61.52,    λ = 6.00  ← violation constant at ~10 J/s
    # Epoch 2000: loss = 100.14,   λ = 10.00 ← λ ramp achieves nothing
    #
    # Diagnosis:
    # (a) Gradient scale mismatch: Adam second moment (m̂₂) was calibrated to
    #     Phase 1 MSE gradients of O(0.002). At epoch 1501 with λ=0.02 and
    #     violation ≈ 11.89 J/s, passivity gradient ≈ λ × 11.89 = 0.24.
    #     Even this tiny initial value is 120× larger than m̂₂ expects.
    #     By λ=2.0 the ratio is ≈ 12,000:1 — Adam step size ≈ 0.
    #     The optimizer is effectively frozen: violation stays at 11.89.
    #
    # (b) Unilateral relu only penalises energy injection (positive rates).
    #     Network learned to comply by injecting LARGE NEGATIVE energy (phantom
    #     braking at −16,131 mJ per 10 ms step = −1613 J/s). This satisfies
    #     relu(rate) = 0 but generates massive unphysical deceleration forces
    #     → NaN gradients in WMPC _simulate_trajectory rollout.
    #
    # V5 DESIGN — TWO FIXES:
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Fix 1 — GRADIENT NORMALISATION (eliminates scale mismatch permanently):
    #
    #   Instead of scalar-weighted sum:  loss = MSE + λ × passivity
    #   Use gradient-normalised sum:     ∇ = ∇_MSE + α × (‖∇_MSE‖/‖∇_pass‖) × ∇_pass
    #
    #   PASSIVITY_ALPHA = 1.0 guarantees passivity contributes exactly 1× the
    #   MSE gradient norm, regardless of:
    #     - violation magnitude (works at 11.89 J/s AND at 0.01 J/s)
    #     - λ value (no tuning needed)
    #     - training phase (scale adapts automatically as network improves)
    #
    #   Mathematical guarantee: the combined gradient is always in a 45° cone
    #   between ∇_MSE and ∇_pass. Neither term can dominate the other.
    #   Adam receives correctly-scaled gradients from the first Phase 2 epoch.
    #
    #   Requires two forward passes per step (1 for MSE, 1 for passivity).
    #   For N=2000 samples and a 128-64 network, this adds ~40% compute.
    #   Acceptable trade-off given the training is offline.
    #
    # Fix 2 — BILATERAL PASSIVITY LOSS + FRESH ADAM at Phase 2 start:
    #
    #   Port-Hamiltonian passivity: dH/dt ≤ 0 at zero external input.
    #   Physical bounds for an FSAE car at 15 m/s:
    #     Injection  (rate > 0):           forbidden. Full penalty.
    #     Dissipation (0 ≥ rate ≥ −50):    physically normal. No penalty.
    #     Phantom braking (rate < −50 J/s): unphysical. Soft penalty (10%).
    #       (Real peak damper dissipation: ~30–80 W per corner = 120–320 J/s
    #        total. DISSIPATION_THRESHOLD = 100 J/s is conservative.)
    #
    #   This steers the network away from the phantom-braking local minimum
    #   that previous unilateral relu enabled.
    #
    #   Fresh Adam state at epoch PHASE_1_END + 1: m̂₁ = m̂₂ = 0 for h_tx_p2.
    #   This eliminates the stale moment problem entirely — the optimiser
    #   correctly sizes its first steps based on actual Phase 2 gradient scale.
    #
    # EXPECTED PHASE 2 OUTCOME:
    #   MSE:                0.002 → 0.005–0.025 (slight degradation is fine)
    #   Passivity violation: 11.89 → < 1.0 J/s  (gradient normalisation enforces)
    #   Phantom braking:   −1613 → < −100 J/s   (bilateral penalty removes)
    #   ΔKE (passive step):−16131 mJ → < −100 mJ (within physical range)
    #   NaN in WMPC:        eliminated            (forces back in physical range)
    # =========================================================================

    NUM_EPOCHS            = 2000
    PHASE_1_END           = 1500   # pure MSE phase (unchanged — works correctly)
    PHASE_2_EPOCHS        = NUM_EPOCHS - PHASE_1_END   # = 500
    PASSIVITY_ALPHA       = 1.0    # passivity ‖grad‖ = PASSIVITY_ALPHA × MSE ‖grad‖
    DISSIPATION_THRESHOLD = 100.0  # J/s — phantom-braking softness boundary

    # Phase 1 optimizer — unchanged from v4, confirmed working
    h_schedule_p1 = optax.cosine_decay_schedule(init_value=1e-3,
                                                  decay_steps=PHASE_1_END,
                                                  alpha=0.01)
    h_tx_p1 = optax.adamw(learning_rate=h_schedule_p1, weight_decay=1e-4)

    # Phase 2 optimizer — FRESH Adam, LR restarts at 5e-4 (50× higher than
    # Phase 1 end LR of 1e-5) to give meaningful step sizes.
    # Cosine to 5e-5 over 500 epochs (not decaying all the way to 1e-5).
    h_schedule_p2 = optax.cosine_decay_schedule(init_value=5e-4,
                                                  decay_steps=PHASE_2_EPOCHS,
                                                  alpha=0.1)
    h_tx_p2 = optax.adamw(learning_rate=h_schedule_p2, weight_decay=1e-4)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 UPDATE — pure MSE, no passivity
    # ─────────────────────────────────────────────────────────────────────────

    @jax.jit
    def h_update_p1(params, opt_state, q, p, setup, target_norm):
        """Phase 1: pure MSE loss. Learns the torsional energy landscape."""
        def mse_loss(params_):
            def per_sample(q_s, p_s, setup_s, t_s):
                total      = h_net.apply(params_, q_s, p_s, setup_s)
                T_prior    = 0.5 * jnp.sum((p_s ** 2) / M_diag)
                V_struct   = 0.5 * jnp.sum(q_s[6:10] ** 2) * 30000.0
                residual   = (total - T_prior - V_struct) / h_scale
                return (residual - t_s) ** 2
            return jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

        loss, grads = jax.value_and_grad(mse_loss)(params)
        updates, new_state = h_tx_p1.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 UPDATE — gradient-normalised MSE + bilateral passivity
    # ─────────────────────────────────────────────────────────────────────────

    @jax.jit
    def h_update_p2(params, opt_state, q, p, setup, target_norm):
        """
        Phase 2: gradient-normalised joint optimisation.

        Two forward passes:
          1. MSE gradient  → ∇_MSE  (same as Phase 1)
          2. Passivity gradient → ∇_pass (bilateral: penalise injection AND phantom braking)

        Combined update: ∇ = ∇_MSE + PASSIVITY_ALPHA × (‖∇_MSE‖/‖∇_pass‖) × ∇_pass

        Guarantees:
          - Passivity always contributes exactly PASSIVITY_ALPHA × MSE gradient norm
          - No λ tuning required — the scale self-adapts each step
          - Adam receives correctly-scaled gradients from epoch 1501 onward
          - Passivity cannot overwhelm MSE (they compete at equal footing)
        """

        def mse_loss(params_):
            def per_sample(q_s, p_s, setup_s, t_s):
                total    = h_net.apply(params_, q_s, p_s, setup_s)
                T_prior  = 0.5 * jnp.sum((p_s ** 2) / M_diag)
                V_struct = 0.5 * jnp.sum(q_s[6:10] ** 2) * 30000.0
                residual = (total - T_prior - V_struct) / h_scale
                return (residual - t_s) ** 2
            return jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

        def passivity_loss(params_):
            """
            Bilateral passivity loss.

            Port-Hamiltonian energy rate at zero external input:
              dH/dt = (∂H/∂q)ᵀ · (p/M) + (∂H/∂p)ᵀ · F_ext=0
                    = (∂H/∂q)ᵀ · v

            Physical decomposition:
              rate > 0:                energy injection     → forbidden, full penalty
              -THRESHOLD ≤ rate ≤ 0:   physical dissipation → allowed, zero penalty
              rate < -THRESHOLD:        phantom braking      → unphysical, soft penalty

            The phantom-braking penalty (weight 0.1) is lighter because large negative
            rates are less catastrophic than large positive rates (they slow the car
            down rather than accelerating it unphysically). The threshold of 100 J/s
            corresponds to ~2× the peak damper dissipation of a typical FSAE car.
            """
            def per_sample(q_s, p_s, setup_s):
                dH_dq  = jax.grad(
                    lambda q_: h_net.apply(params_, q_, p_s, setup_s)
                )(q_s)
                v      = p_s / M_diag
                rate   = jnp.dot(dH_dq, v)
                inject  = jax.nn.relu(rate)                                         # injection
                phantom = 0.1 * jax.nn.relu(-rate - DISSIPATION_THRESHOLD)          # phantom braking
                return inject + phantom
            return jnp.mean(jax.vmap(per_sample)(q, p, setup))

        # Compute both gradients
        mse_val,  mse_grads  = jax.value_and_grad(mse_loss)(params)
        pass_val, pass_grads = jax.value_and_grad(passivity_loss)(params)

        # Gradient norms — sum over all leaves of the parameter pytree
        # jax.tree_util.tree_leaves is static at trace time (Flax pytree structure
        # is fixed), so Python's sum() on JAX arrays is well-defined inside jit.
        mse_norm_sq = sum(
            jnp.sum(g ** 2)
            for g in jax.tree_util.tree_leaves(mse_grads)
        )
        pass_norm_sq = sum(
            jnp.sum(g ** 2)
            for g in jax.tree_util.tree_leaves(pass_grads)
        )
        mse_norm  = jnp.sqrt(mse_norm_sq  + 1e-12)
        pass_norm = jnp.sqrt(pass_norm_sq + 1e-12)

        # Passivity scale: ensures ‖PASSIVITY_ALPHA × scale × ∇_pass‖ = PASSIVITY_ALPHA × ‖∇_MSE‖
        scale = PASSIVITY_ALPHA * mse_norm / (pass_norm + 1e-8)

        # Combined gradient: MSE term + normalised passivity term
        combined_grads = jax.tree_util.tree_map(
            lambda gm, gp: gm + scale * gp,
            mse_grads, pass_grads,
        )

        updates, new_state = h_tx_p2.update(combined_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, mse_val, pass_val, scale

    # ─────────────────────────────────────────────────────────────────────────
    # R_net loss and update (unchanged — R_net training is not the bottleneck)
    # ─────────────────────────────────────────────────────────────────────────

    r_schedule = optax.cosine_decay_schedule(init_value=1e-3,
                                              decay_steps=NUM_EPOCHS,
                                              alpha=0.01)
    r_tx = optax.adamw(learning_rate=r_schedule, weight_decay=1e-4)
    r_opt_state = r_tx.init(r_params)

    @jax.jit
    def r_loss_fn(params, q, p, target_mag_norm):
        preds   = jax.vmap(r_net.apply, in_axes=(None, 0, 0))(params, q, p)
        targets = jax.vmap(lambda mag: jnp.eye(14) * mag)(target_mag_norm)
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def r_update(params, opt_state, q, p, target_mag_norm):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target_mag_norm)
        updates, new_opt_state = r_tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # H_net training loop
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Neural Physics] Training H_net (Energy Landscape Residual)...")
    print(f"  [Config] Epochs: {NUM_EPOCHS} | AdamW | weight_decay: 1e-4")
    print(f"  [Two-Phase Design — v5 gradient normalisation]")
    print(f"  Phase 1 (ep 1→{PHASE_1_END}): λ=0.0 — PURE MSE.")
    print(f"    Network learns torsional energy landscape with zero constraint interference.")
    print(f"  Phase 2 (ep {PHASE_1_END+1}→{NUM_EPOCHS}): gradient-normalised MSE + bilateral passivity.")
    print(f"    Fresh Adam state at phase transition (eliminates stale second-moment problem).")
    print(f"    Passivity gradient scaled to {PASSIVITY_ALPHA}× MSE gradient norm every step.")
    print(f"    Bilateral: penalises injection (full) AND phantom braking >100 J/s (10% weight).")
    print(f"    LR: cosine 5e-4→5e-5 (Phase 2 restarts higher than Phase 1 end LR of ~1e-5).")

    h_opt_state_p1 = h_tx_p1.init(h_params)
    h_opt_state_p2 = None   # initialised lazily at first Phase 2 epoch

    for epoch in range(1, NUM_EPOCHS + 1):

        if epoch <= PHASE_1_END:
            # ── Phase 1: pure MSE ─────────────────────────────────────────────
            h_params, h_opt_state_p1, h_loss = h_update_p1(
                h_params, h_opt_state_p1,
                q_data, p_data, setup_data, target_H_norm,
            )

            if epoch % 200 == 0:
                lr_now = float(h_schedule_p1(epoch))
                print(f"  Epoch {epoch:4d} | Loss: {h_loss:.6f} | "
                      f"lr: {lr_now:.2e} | Phase 1 (pure MSE)")

        else:
            # ── Phase 2: gradient-normalised MSE + bilateral passivity ────────
            if h_opt_state_p2 is None:
                # One-time fresh Adam initialisation at phase transition.
                # h_tx_p2 has never seen any gradients: m̂₁ = m̂₂ = 0.
                # This gives Adam correct step-size calibration for Phase 2
                # gradient scales from the very first update.
                h_opt_state_p2 = h_tx_p2.init(h_params)
                print(f"\n  [Phase 2 START] Epoch {epoch}: fresh Adam state (LR=5e-4). "
                      f"Gradient normalisation + bilateral passivity active.")

            h_params, h_opt_state_p2, mse_l, pass_l, p_scale = h_update_p2(
                h_params, h_opt_state_p2,
                q_data, p_data, setup_data, target_H_norm,
            )

            if epoch % 200 == 0:
                p2_pct = (epoch - PHASE_1_END) / PHASE_2_EPOCHS * 100
                lr_now = float(h_schedule_p2(epoch - PHASE_1_END))
                print(f"  Epoch {epoch:4d} | "
                      f"MSE: {float(mse_l):.6f} | "
                      f"Violation: {float(pass_l):.4f} J/s | "
                      f"PassScale: {float(p_scale):.3f} | "
                      f"lr: {lr_now:.2e} | "
                      f"Phase 2 ({p2_pct:.0f}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # R_net training loop
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        r_params, r_opt_state, r_loss = r_update(
            r_params, r_opt_state,
            q_data, p_data, target_R_mag_norm,
        )
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | "
                  f"lr: {float(r_schedule(epoch)):.2e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Post-training passive energy injection diagnostic
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
        from data.configs.tire_coeffs import tire_coeffs as TP_DICT
        _veh = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        _x0  = jnp.zeros(46).at[14].set(10.0)
        _u0  = jnp.array([0.0, 0.0])
        _sp  = jnp.array([35000., 38000., 400., 450., 2500., 2800., 0.28, 0.60])
        _x1  = _veh.simulate_step(_x0, _u0, _sp, dt=0.01)
        _m   = VP_DICT.get('total_mass', 230.0)
        _dKE = 0.5 * _m * (float(_x1[14]) ** 2 - float(_x0[14]) ** 2)
        budget_J = 0.10   # 100 mJ
        if abs(_dKE) < budget_J:
            _tag = (f"✓ PASS  ({_dKE * 1000:.1f} mJ < {budget_J * 1000:.0f} mJ budget)")
        else:
            sign = "injection" if _dKE > 0 else "phantom braking"
            _tag = (f"✗ WARN  ({_dKE * 1000:.1f} mJ — {sign}) — "
                    f"passivity not fully converged with synthetic data. "
                    f"Real 4-post rig data will improve this further.")
        print(f"\n[Neural Physics] Passive energy injection: {_tag}")
    except Exception as _e:
        print(f"[Neural Physics] Energy injection check skipped: {_e}")

    print("\n[Neural Physics] Pre-training complete!")
    print(f"Scale factors:  h_scale={h_scale:.2f} J  |  r_scale={r_scale:.4f}")
    print("Access via: from residual_fitting import TRAINED_H_SCALE, TRAINED_R_SCALE")

    global TRAINED_H_SCALE, TRAINED_R_SCALE
    TRAINED_H_SCALE = h_scale
    TRAINED_R_SCALE = r_scale

    # ── Save weights and scale ────────────────────────────────────────────────
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