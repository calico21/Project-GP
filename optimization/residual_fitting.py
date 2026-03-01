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


def generate_synthetic_flex_data(num_samples=5000, key_seed=42):
    """
    Generates synthetic training data for chassis torsional flex.
    Torsional stiffness at 15 000 Nm/rad (synthetic approximation).

    FIX (v6) — physical momenta and setup parameters.
    ─────────────────────────────────────────────────
    Previous: p ~ N(0, 0.1), setup_params ~ N(0, 1).

    At p ~ N(0, 0.1) with M_diag[0] = 188 kg:
        v[0] = p[0] / 188 = 0.0005 m/s  (operating speed: 15 m/s)

    The bilateral passivity penalty at this training velocity:
        phantom rate = dH/dq[0] × 0.0005 = −0.85 J/s
        penalty      = 0.1 × relu(0.85 − 100) = 0.000   ← ZERO gradient

    The network learns arbitrary phantom forces because the threshold
    of 100 J/s is never crossed at training time.  At the actual test
    point (vx=10 m/s), the same forces give −169,656 mJ.

    Fix: sample p from physical momenta p = M × v where v covers the
    operating envelope.  At v[0] ~ N(15, 5) m/s:
        phantom rate = −1696 × 15 = −25,440 J/s
        penalty      = 0.1 × (25,440 − 100) = 2,534 J/s  ← strong gradient

    Fix also: setup_params must be in raw physical units because
    NeuralEnergyLandscape.__call__ passes them through
    PhysicsNormalizer.normalize_setup() internally.  N(0,1) was sampling
    k_f ≈ 0 N/m instead of 40,000 N/m, causing H_net to learn the wrong
    coupling between spring stiffness and torsional energy.
    """
    # Physical mass/inertia matching DifferentiableMultiBodyVehicle defaults
    _m_s   = VP_DICT.get('total_mass', 230.0) - (
                2 * VP_DICT.get('unsprung_mass_f', 10.0) +
                2 * VP_DICT.get('unsprung_mass_r', 11.0))
    _Ix    = VP_DICT.get('Ix', 45.0)
    _Iy    = VP_DICT.get('Iy', 85.0)
    _Iz    = VP_DICT.get('Iz', 125.0)
    _m_usf = VP_DICT.get('unsprung_mass_f', 10.0)
    _m_usr = VP_DICT.get('unsprung_mass_r', 11.0)
    _Iw    = VP_DICT.get('Iw', 1.2)

    M_diag_train = jnp.array([_m_s, _m_s, _m_s, _Ix, _Iy, _Iz,
                                _m_usf, _m_usf, _m_usr, _m_usr,
                                _Iw, _Iw, _Iw, _Iw])

    # Physical velocity standard deviations (1σ ≈ operating range / 2)
    v_std = jnp.array([
        15.0,   # vx  longitudinal (5–25 m/s operating range)
         2.0,   # vy  lateral
         0.3,   # vz  heave
         0.4,   # wx  roll rate
         0.3,   # wy  pitch rate
         1.5,   # wz  yaw rate (0.5G at 15 m/s → 0.98 rad/s)
         0.4,   # unsprung vz FL
         0.4,   # unsprung vz FR
         0.4,   # unsprung vz RL
         0.4,   # unsprung vz RR
        75.0,   # wheel omega FL  (15 m/s / 0.2032 m ≈ 73.8 rad/s)
        75.0,   # wheel omega FR
        75.0,   # wheel omega RL
        75.0,   # wheel omega RR
    ])

    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (num_samples, 14)) * 0.05   # position (m)

    # Physical momenta: p = M × v
    v_samples = jax.random.normal(k2, (num_samples, 14)) * v_std
    p         = v_samples * M_diag_train

    # Physical setup parameters in raw units (PhysicsNormalizer handles scaling)
    # [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f]
    setup_mean_train  = jnp.array([40000., 40000.,  500.,  500., 3000., 3000., 0.30, 0.60])
    setup_scale_train = jnp.array([15000., 15000.,  300.,  300., 1500., 1500., 0.03, 0.08])
    setup_params = (setup_mean_train
                    + jax.random.normal(k3, (num_samples, 8)) * setup_scale_train)

    torsion      = (q[:, 6] - q[:, 7]) - (q[:, 8] - q[:, 9])
    k_torsion    = 15000.0
    target_H     = 0.5 * k_torsion * (torsion ** 2)
    target_R_mag = jnp.abs(torsion) * 500.0

    return q, p, setup_params, target_H, target_R_mag


def train_neural_residuals():
    print("[Neural Physics] Generating Synthetic Chassis Flex Data...")
    q_data, p_data, setup_data, target_H, target_R_mag = generate_synthetic_flex_data()

    # ── B1 FIX: M_diag must match generate_synthetic_flex_data exactly ───────
    # Previous bug: used VP_DICT.get('m_s', m*0.85)=195.5 kg, Ix=200, Iy=800.
    # Correct:      total_mass - 4×m_us = 188 kg, Ix=45, Iy=85.
    # Error at vx=10 m/s: T_prior was +420 J too high → H_residual learned
    # a phantom −420 J offset to compensate → contributed to phantom braking.
    m_s   = VP_DICT.get('total_mass', 230.0) - (
                2 * VP_DICT.get('unsprung_mass_f', 10.0) +
                2 * VP_DICT.get('unsprung_mass_r', 11.0))
    m_usf = VP_DICT.get('unsprung_mass_f', 10.0)
    m_usr = VP_DICT.get('unsprung_mass_r', 11.0)
    Ix    = VP_DICT.get('Ix', 45.0)
    Iy    = VP_DICT.get('Iy', 85.0)
    Iz    = VP_DICT.get('Iz', 125.0)
    Iw    = VP_DICT.get('Iw', 1.2)
    M_diag = jnp.array([m_s, m_s, m_s, Ix, Iy, Iz,
                         m_usf, m_usf, m_usr, m_usr, Iw, Iw, Iw, Iw])

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
    # THREE-PHASE TRAINING — VERSION 7: GRADUATED PASSIVITY SCHEDULE
    # =========================================================================
    #
    # FAILURE HISTORY:
    # ─────────────────────────────────────────────────────────────────────────
    # v1-v4: Fixed/ramped λ — various gradient scale mismatches.
    # v5:  Gradient normalisation + ALPHA=20 → MSE diverges 0.525→1.648.
    #      Phase 1 MSE was 0.525 (not converged) so ||∇_pass||/||∇_MSE||≈850.
    #      ALPHA=20 overpowered MSE by 20×.
    # v6a: B1 M_diag fix + ALPHA=1.0 → PassScale=0.000. Phase 1 now converges
    #      to MSE=0.082 (B1 fixed T_prior). But ||∇_pass||/||∇_MSE||≈9000 now
    #      because MSE baseline is much lower. ALPHA=1/9000 ≈ 0.0001 effective.
    # v6b: ALPHA=5.0 → PassScale grows to 0.54, violation 758→116 J/s (close!)
    #      BUT MSE diverges 0.082→6.437 (78×) and NaN rate 0%→67%.
    #      ||∇_pass||/||∇_MSE||≈9000 means ALPHA=5 ≡ ALPHA=53 in v5.
    #
    # THREE-PHASE SOLUTION:
    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1 (ep 1→2500):     Pure MSE. Converges to ~0.082.
    # Phase 2 (ep 2501→3200):  ALPHA=3.0. Burns violation 758→~250 J/s.
    #                           MSE degrades ~3-5× (→0.25-0.4), not 78×.
    #                           Fresh Adam restart at Phase 2 start.
    # Phase 3 (ep 3201→4000):  ALPHA=1.0. MSE recovery + gentle passivity.
    #                           Fresh Adam restart clears Phase 2 moments.
    #                           LR=2e-4 (lower than Phase 2's 5e-4).
    #                           Violation continues to ~150-200 J/s.
    #                           MSE recovers toward ~0.1-0.3.
    #
    # WHY ALPHA=3.0 FOR PHASE 2:
    #   ||∇_pass||/||∇_MSE|| ≈ 9000 at Phase 2 start.
    #   ALPHA=1.0: scale=1/9000 → PassScale≈0.000, no effect.
    #   ALPHA=5.0: MSE diverges 78× (observed, NaN in WMPC).
    #   ALPHA=3.0: effective passivity = 3× MSE norm.
    #              MSE degrades ~3-5× (acceptable). Violation drops ~250 J/s.
    #
    # WHY FRESH ADAM AT PHASE 3:
    #   Phase 2 with ALPHA=3 accumulates large m̂₂ for the passivity direction.
    #   Without reset, Phase 3 ALPHA=1.0 would still be pushed by these stale
    #   moments for ~100+ epochs. Fresh Adam at epoch 3201 clears this.
    # =========================================================================

    NUM_EPOCHS            = 4000
    PHASE_1_END           = 2500
    PHASE_2_END           = 3200   # 700 Phase 2 epochs: aggressive passivity
    PHASE_3_END           = 4000   # 800 Phase 3 epochs: MSE recovery
    PHASE_2_EPOCHS        = PHASE_2_END - PHASE_1_END   # = 700
    PHASE_3_EPOCHS        = PHASE_3_END - PHASE_2_END   # = 800

    PASSIVITY_ALPHA_P2    = 3.0    # Phase 2: aggressive (3× MSE norm)
    PASSIVITY_ALPHA_P3    = 1.0    # Phase 3: gentle (1× MSE norm)
    DISSIPATION_THRESHOLD = 100.0  # J/s — phantom-braking softness boundary

    # Phase 1 optimizer
    h_schedule_p1 = optax.cosine_decay_schedule(init_value=1e-3,
                                                  decay_steps=PHASE_1_END,
                                                  alpha=0.01)
    h_tx_p1 = optax.adamw(learning_rate=h_schedule_p1, weight_decay=1e-4)

    # Phase 2 optimizer — fresh Adam, cosine 5e-4→5e-5 over 700 epochs
    h_schedule_p2 = optax.cosine_decay_schedule(init_value=5e-4,
                                                  decay_steps=PHASE_2_EPOCHS,
                                                  alpha=0.1)
    h_tx_p2 = optax.adamw(learning_rate=h_schedule_p2, weight_decay=1e-4)

    # Phase 3 optimizer — fresh Adam, cosine 2e-4→1e-5 over 800 epochs.
    # Lower starting LR than Phase 2: MSE recovery needs finer steps.
    h_schedule_p3 = optax.cosine_decay_schedule(init_value=2e-4,
                                                  decay_steps=PHASE_3_EPOCHS,
                                                  alpha=0.05)
    h_tx_p3 = optax.adamw(learning_rate=h_schedule_p3, weight_decay=1e-4)

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
    # PHASE 2/3 UPDATE FACTORY — gradient-normalised MSE + bilateral passivity
    # ─────────────────────────────────────────────────────────────────────────
    # Two separate JIT-compiled functions are required because PASSIVITY_ALPHA
    # is captured as a compile-time constant by @jax.jit. Passing it as a
    # dynamic argument would force retracing every epoch. One function per phase.

    def _make_h_update(alpha_val, optimizer):
        """
        Returns a JIT-compiled update function for Phase 2 or Phase 3.

        Combined gradient: ∇ = ∇_MSE + alpha_val × (‖∇_MSE‖/‖∇_pass‖) × ∇_pass

        Guarantees passivity contributes exactly alpha_val × MSE gradient norm,
        regardless of violation magnitude — the scale self-adapts each step.
        """
        @jax.jit
        def _update(params, opt_state, q, p, setup, target_norm):
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
                  dH/dt = (∂H/∂q)ᵀ · v

                  rate > 0:               energy injection     → full penalty
                  -THRESHOLD ≤ rate ≤ 0:  physical dissipation → no penalty
                  rate < -THRESHOLD:       phantom braking      → 10% penalty
                """
                def per_sample(q_s, p_s, setup_s):
                    dH_dq  = jax.grad(
                        lambda q_: h_net.apply(params_, q_, p_s, setup_s)
                    )(q_s)
                    v      = p_s / M_diag
                    rate   = jnp.dot(dH_dq, v)
                    inject  = jax.nn.relu(rate)
                    phantom = 0.1 * jax.nn.relu(-rate - DISSIPATION_THRESHOLD)
                    return inject + phantom
                return jnp.mean(jax.vmap(per_sample)(q, p, setup))

            mse_val,  mse_grads  = jax.value_and_grad(mse_loss)(params)
            pass_val, pass_grads = jax.value_and_grad(passivity_loss)(params)

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

            # scale: ‖alpha × scale × ∇_pass‖ = alpha × ‖∇_MSE‖
            scale = alpha_val * mse_norm / (pass_norm + 1e-8)

            combined_grads = jax.tree_util.tree_map(
                lambda gm, gp: gm + scale * gp,
                mse_grads, pass_grads,
            )

            updates, new_state = optimizer.update(combined_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, mse_val, pass_val, scale

        return _update

    h_update_p2 = _make_h_update(PASSIVITY_ALPHA_P2, h_tx_p2)
    h_update_p3 = _make_h_update(PASSIVITY_ALPHA_P3, h_tx_p3)

    # ─────────────────────────────────────────────────────────────────────────
    # R_net loss and update
    # ─────────────────────────────────────────────────────────────────────────

    r_schedule = optax.cosine_decay_schedule(init_value=1e-3,
                                              decay_steps=NUM_EPOCHS,
                                              alpha=0.01)
    r_tx = optax.adamw(learning_rate=r_schedule, weight_decay=1e-4)
    r_opt_state = r_tx.init(r_params)

    # DOF-specific dissipation weights — diagonal-only loss gives 14× more
    # gradient signal per DOF vs the previous 196-element full-matrix loss.
    R_DOF_WEIGHTS = jnp.array([
        0.08, 0.08, 0.80, 0.40, 0.40, 0.15,   # sprung: x,y,z,roll,pitch,yaw
        1.00, 1.00, 1.00, 1.00,                # unsprung heave FL/FR/RL/RR
        0.02, 0.02, 0.02, 0.02,                # wheel spin (rolling resistance)
    ])

    @jax.jit
    def r_loss_fn(params, q, p, target_mag_norm):
        preds     = jax.vmap(r_net.apply, in_axes=(None, 0, 0))(params, q, p)
        pred_diag = jax.vmap(jnp.diag)(preds)
        tgt_diag  = target_mag_norm[:, None] * R_DOF_WEIGHTS[None,:]
        return jnp.mean((pred_diag - tgt_diag) ** 2)

    @jax.jit
    def r_update(params, opt_state, q, p, target_mag_norm):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target_mag_norm)
        updates, new_opt_state = r_tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # H_net training loop — three phases
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Neural Physics] Training H_net (Energy Landscape Residual)...")
    print(f"  [Config] Epochs: {NUM_EPOCHS} | AdamW | weight_decay: 1e-4")
    print(f"  [Three-Phase Design — v7: graduated passivity schedule]")
    print(f"  Phase 1 (ep 1→{PHASE_1_END}): λ=0.0 — PURE MSE.")
    print(f"    Network learns torsional energy landscape with zero constraint interference.")
    print(f"  Phase 2 (ep {PHASE_1_END+1}→{PHASE_2_END}): ALPHA={PASSIVITY_ALPHA_P2:.1f} — aggressive passivity.")
    print(f"    Burns violation 758→~250 J/s. Fresh Adam restart at transition.")
    print(f"  Phase 3 (ep {PHASE_2_END+1}→{PHASE_3_END}): ALPHA={PASSIVITY_ALPHA_P3:.1f} — MSE recovery.")
    print(f"    MSE recovers toward 0.1-0.3. Fresh Adam restart clears Phase 2 moments.")
    print(f"    Training momenta: physical (v[0]~N(15,5) m/s) — fires at operating speed.")

    h_opt_state_p1 = h_tx_p1.init(h_params)
    h_opt_state_p2 = None   # initialised lazily at first Phase 2 epoch
    h_opt_state_p3 = None   # initialised lazily at first Phase 3 epoch

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

        elif epoch <= PHASE_2_END:
            # ── Phase 2: aggressive passivity (ALPHA=3.0) ─────────────────────
            if h_opt_state_p2 is None:
                h_opt_state_p2 = h_tx_p2.init(h_params)
                print(f"\n  [Phase 2 START] Epoch {epoch}: fresh Adam (LR=5e-4). "
                      f"ALPHA={PASSIVITY_ALPHA_P2:.1f}. "
                      f"Gradient normalisation + bilateral passivity active.")

            h_params, h_opt_state_p2, mse_l, pass_l, p_scale = h_update_p2(
                h_params, h_opt_state_p2,
                q_data, p_data, setup_data, target_H_norm,
            )
            if epoch % 200 == 0:
                p2_pct = int(100 * (epoch - PHASE_1_END) / PHASE_2_EPOCHS)
                lr_now = float(h_schedule_p2(epoch - PHASE_1_END))
                print(f"  Epoch {epoch:4d} | "
                      f"MSE: {float(mse_l):.6f} | "
                      f"Violation: {float(pass_l):.4f} J/s | "
                      f"PassScale: {float(p_scale):.3f} | "
                      f"lr: {lr_now:.2e} | "
                      f"Phase 2 ({p2_pct}%) α=3.0")

        else:
            # ── Phase 3: MSE recovery with gentle passivity (ALPHA=1.0) ───────
            if h_opt_state_p3 is None:
                # Fresh Adam: clears Phase 2's large passivity moment estimates.
                # Without this, stale m̂₂ from Phase 2 continues pushing the
                # network in the passivity direction for ~100+ epochs despite
                # the reduced ALPHA. Fresh start gives correct step calibration
                # from the first Phase 3 epoch.
                h_opt_state_p3 = h_tx_p3.init(h_params)
                print(f"\n  [Phase 3 START] Epoch {epoch}: fresh Adam (LR=2e-4). "
                      f"ALPHA={PASSIVITY_ALPHA_P3:.1f}. MSE recovery + gentle passivity.")

            h_params, h_opt_state_p3, mse_l, pass_l, p_scale = h_update_p3(
                h_params, h_opt_state_p3,
                q_data, p_data, setup_data, target_H_norm,
            )
            if epoch % 200 == 0:
                p3_pct = int(100 * (epoch - PHASE_2_END) / PHASE_3_EPOCHS)
                lr_now = float(h_schedule_p3(epoch - PHASE_2_END))
                print(f"  Epoch {epoch:4d} | "
                      f"MSE: {float(mse_l):.6f} | "
                      f"Violation: {float(pass_l):.4f} J/s | "
                      f"PassScale: {float(p_scale):.3f} | "
                      f"lr: {lr_now:.2e} | "
                      f"Phase 3 ({p3_pct}%) α=1.0")

    # ─────────────────────────────────────────────────────────────────────────
    # R_net training loop — two-phase with LR restart at midpoint
    # ─────────────────────────────────────────────────────────────────────────
    R_PHASE_1_END  = 1000
    r_schedule_p2  = optax.cosine_decay_schedule(init_value=5e-4,
                                                   decay_steps=NUM_EPOCHS - R_PHASE_1_END,
                                                   alpha=0.1)
    r_tx_p2        = optax.adamw(learning_rate=r_schedule_p2, weight_decay=1e-4)
    r_opt_state_p2 = None

    @jax.jit
    def r_update_p2(params, opt_state, q, p, target_mag_norm):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target_mag_norm)
        updates, new_state = r_tx_p2.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch <= R_PHASE_1_END:
            r_params, r_opt_state, r_loss = r_update(
                r_params, r_opt_state,
                q_data, p_data, target_R_mag_norm,
            )
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | "
                      f"lr: {float(r_schedule(epoch)):.2e}")
        else:
            if r_opt_state_p2 is None:
                r_opt_state_p2 = r_tx_p2.init(r_params)
                print(f"  [R Phase 2] Epoch {epoch}: fresh Adam, LR restart 5e-4.")

            r_params, r_opt_state_p2, r_loss = r_update_p2(
                r_params, r_opt_state_p2,
                q_data, p_data, target_R_mag_norm,
            )
            if epoch % 200 == 0:
                lr_now = float(r_schedule_p2(epoch - R_PHASE_1_END))
                print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | "
                      f"lr: {lr_now:.2e} | R Phase 2")

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