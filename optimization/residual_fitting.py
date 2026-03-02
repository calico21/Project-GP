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

    Fix: sample p from physical momenta p = M × v where v covers the
    operating envelope.  At v[0] ~ N(15, 5) m/s:
        phantom rate = −1696 × 15 = −25,440 J/s  ← strong gradient

    Fix also: setup_params in raw physical units (PhysicsNormalizer handles
    scaling internally). N(0,1) was sampling k_f ≈ 0 N/m.
    """
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

    v_std = jnp.array([
        15.0,   # vx  longitudinal
         2.0,   # vy  lateral
         0.3,   # vz  heave
         0.4,   # wx  roll rate
         0.3,   # wy  pitch rate
         1.5,   # wz  yaw rate
         0.4,   # unsprung vz FL
         0.4,   # unsprung vz FR
         0.4,   # unsprung vz RL
         0.4,   # unsprung vz RR
        75.0,   # wheel omega FL
        75.0,   # wheel omega FR
        75.0,   # wheel omega RL
        75.0,   # wheel omega RR
    ])

    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (num_samples, 14)) * 0.05

    v_samples = jax.random.normal(k2, (num_samples, 14)) * v_std
    p         = v_samples * M_diag_train

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

    # ── B1 FIX: M_diag identical to generate_synthetic_flex_data ─────────────
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

    h_scale = float(jnp.std(target_H) + 1e-6)
    r_scale = float(jnp.std(target_R_mag) + 1e-6)
    target_H_norm     = target_H     / h_scale
    target_R_mag_norm = target_R_mag / r_scale
    print(f"   [Neural Physics] H target scale: {h_scale:.2f} J  "
          f"| R target scale: {r_scale:.4f}")

    # =========================================================================
    # TWO-PHASE TRAINING — VERSION 9: PHANTOM RATE AT EQUILIBRIUM
    # =========================================================================
    #
    # FAILURE HISTORY (v5–v8):
    # ─────────────────────────────────────────────────────────────────────────
    # v5  ALPHA=20                → MSE diverges 0.525→1.648
    # v6a ALPHA=1                 → PassScale=0.000 (ratio 9000:1)
    # v6b ALPHA=5                 → NaN=67%, MSE diverges 78×
    # v7  3-phase ALPHA=3/1       → NaN=100%, ΔKE=-16160 unchanged
    # v8  EqGrad²(q=0,p=0)        → EqGrad²=0.0000 immediately
    #     ΔKE=-16160 unchanged    → regulariser solved already-solved problem
    #
    # CONFIRMED ROOT CAUSE (identified from v8 null result):
    # ─────────────────────────────────────────────────────────────────────────
    # H_net is f(q, p, setup). Phase 1 correctly zeroed dH/dq at (q=0, p=0).
    # But the network learned spurious p-DEPENDENT q-gradients:
    #
    #   dH_residual/dq[0] at (q=0, p=0):       ≈ 0        ← Phase 1 fixed this
    #   dH_residual/dq[0] at (q=0, p[0]=1880): ≈ −161.6 N ← THE PROBLEM
    #
    # The diagnostic test uses q=zeros(46).at[14].set(10.0), which maps to
    # state q_14D=0, p[0] = m_s × 10 = 188 × 10 = 1880 kg·m/s.
    # The phantom braking force is this momentum-modulated q-gradient.
    #
    # Every previous loss evaluated the wrong point:
    #   passivity_loss:    correct p range, but random q (not q=0)
    #   equilibrium_loss:  correct q=0, but p=0 (problem doesn't exist there)
    #
    # THE FIX — PHANTOM RATE AT EQUILIBRIUM LOSS:
    # ─────────────────────────────────────────────────────────────────────────
    #   L_phantom = E_{p,setup} [ (dH/dq(q=0, p, setup) · v)² ]
    #
    # Three properties that make this work where v5–v8 did not:
    #
    #   1. Evaluated AT q=0  — the exact diagnostic test configuration.
    #      No distribution shift between training and evaluation.
    #
    #   2. Uses PHYSICAL p values from the training batch — where the
    #      spurious p-q coupling actually lives (p[0]~N(2820,940) kg·m/s
    #      for vx~N(15,5) m/s). This is the opposite error from v8.
    #
    #   3. Squared penalty (rate²) — no threshold, no dead zone, symmetric.
    #      Penalises BOTH phantom braking (rate < 0) AND injection (rate > 0).
    #      relu-based losses have zero gradient when rate is in the dead zone.
    #
    # Target_H = 0.5×k×torsion² has no p-dependence → ideal dH/dq is
    # p-independent. L_phantom forces the residual to respect this.
    #
    # WHY MSE WILL NOT DIVERGE this time:
    #   L_phantom and MSE are consistent: both want the residual to be
    #   flat (zero gradient) at q=0 across all p. They are not competing.
    #   ALPHA_PHANTOM=3.0 gives 3× MSE norm contribution. With consistent
    #   gradients, this accelerates convergence rather than causing divergence.
    #   Predicted Phase 2 MSE degradation: ≤2× (vs 78× with ALPHA=5 passivity).
    #
    # BILATERAL PASSIVITY REMAINS at ALPHA=1.0 as a secondary guard against
    # energy injection at random (q, p) points during rollout.
    #
    # WHAT TO WATCH IN THE LOG:
    #   PhRate (mean squared phantom rate at q=0) — must decrease monotonically.
    #   PhScale — grows as PhRate falls (self-stabilising).
    #   Expected: PhRate 2.5M → <10k over 1000 epochs.
    #   ΔKE: -16160 mJ should move for the first time.
    # =========================================================================

    NUM_EPOCHS     = 3500
    PHASE_1_END    = 2500
    PHASE_2_EPOCHS = NUM_EPOCHS - PHASE_1_END   # = 1000

    ALPHA_PHANTOM  = 3.0   # phantom rate at eq: 3× MSE norm
    ALPHA_PASS     = 1.0   # bilateral passivity guard: 1× MSE norm
    DISSIPATION_THRESHOLD = 100.0  # J/s — passivity softness boundary

    # Phase 1 — pure MSE
    h_schedule_p1 = optax.cosine_decay_schedule(init_value=1e-3,
                                                  decay_steps=PHASE_1_END,
                                                  alpha=0.01)
    h_tx_p1 = optax.adamw(learning_rate=h_schedule_p1, weight_decay=1e-4)

    # Phase 2 — fresh Adam, cosine 5e-4→5e-5
    h_schedule_p2 = optax.cosine_decay_schedule(init_value=5e-4,
                                                  decay_steps=PHASE_2_EPOCHS,
                                                  alpha=0.1)
    h_tx_p2 = optax.adamw(learning_rate=h_schedule_p2, weight_decay=1e-4)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 UPDATE — pure MSE
    # ─────────────────────────────────────────────────────────────────────────

    @jax.jit
    def h_update_p1(params, opt_state, q, p, setup, target_norm):
        def mse_loss(params_):
            def per_sample(q_s, p_s, setup_s, t_s):
                total    = h_net.apply(params_, q_s, p_s, setup_s)
                T_prior  = 0.5 * jnp.sum((p_s ** 2) / M_diag)
                V_struct = 0.5 * jnp.sum(q_s[6:10] ** 2) * 30000.0
                residual = (total - T_prior - V_struct) / h_scale
                return (residual - t_s) ** 2
            return jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

        loss, grads = jax.value_and_grad(mse_loss)(params)
        updates, new_state = h_tx_p1.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 UPDATE — MSE + phantom rate at eq + bilateral passivity
    # ─────────────────────────────────────────────────────────────────────────

    @jax.jit
    def h_update_p2(params, opt_state, q, p, setup, target_norm):
        """
        Three-term gradient-normalised update:

        Term 1 — MSE: energy landscape on training (q, p, setup) samples.

        Term 2 — Phantom rate at equilibrium (THE NEW TERM):
            L_phantom = E_{p,setup}[ (dH/dq(q=0, p, setup) · v)² ]
            Evaluated at q=0 with PHYSICAL p values from the training batch.
            This is the exact point where ΔKE=-16160 mJ is measured.
            Forces the p-dependent q-gradient to zero across the full
            operating momentum range — eliminating the spurious coupling.

        Term 3 — Bilateral passivity: secondary injection guard at random
            (q, p) points. rate>0 fully penalised; rate<-100 J/s softly.

        All terms scaled to ||∇_MSE|| via gradient normalisation.
        Adam receives correctly-calibrated gradients from epoch 2501 onward.
        """

        # ── Term 1: MSE ───────────────────────────────────────────────────────
        def mse_loss(params_):
            def per_sample(q_s, p_s, setup_s, t_s):
                total    = h_net.apply(params_, q_s, p_s, setup_s)
                T_prior  = 0.5 * jnp.sum((p_s ** 2) / M_diag)
                V_struct = 0.5 * jnp.sum(q_s[6:10] ** 2) * 30000.0
                residual = (total - T_prior - V_struct) / h_scale
                return (residual - t_s) ** 2
            return jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

        # ── Term 2: Phantom rate at equilibrium ───────────────────────────────
        # Evaluates (dH/dq · v)² at q=0 for every (p, setup) in the batch.
        #
        # q=0 is the equilibrium configuration — identical to the diagnostic:
        #   _x0 = jnp.zeros(46).at[14].set(10.0)  → q_14D = zeros(14)
        #
        # p comes from the physical training distribution, so it covers the
        # full operating range including p[0]~N(2820,940) kg·m/s where the
        # phantom braking force was learned.
        #
        # rate² penalises BOTH phantom braking and injection with no threshold
        # dead zone — the network cannot escape by staying just under 100 J/s.
        def phantom_rate_at_eq_loss(params_):
            q_eq = jnp.zeros(14)
            def per_sample(p_s, setup_s):
                dH_dq = jax.grad(
                    lambda q_: h_net.apply(params_, q_, p_s, setup_s)
                )(q_eq)
                v    = p_s / M_diag
                rate = jnp.dot(dH_dq, v)
                return rate ** 2
            return jnp.mean(jax.vmap(per_sample)(p, setup))

        # ── Term 3: Bilateral passivity ───────────────────────────────────────
        # Secondary guard: prevents energy injection at random (q, p) during
        # rollout (not just at q=0). Complements L_phantom.
        def passivity_loss(params_):
            def per_sample(q_s, p_s, setup_s):
                dH_dq = jax.grad(
                    lambda q_: h_net.apply(params_, q_, p_s, setup_s)
                )(q_s)
                v      = p_s / M_diag
                rate   = jnp.dot(dH_dq, v)
                inject  = jax.nn.relu(rate)
                phantom = 0.1 * jax.nn.relu(-rate - DISSIPATION_THRESHOLD)
                return inject + phantom
            return jnp.mean(jax.vmap(per_sample)(q, p, setup))

        # ── Compute all gradients ─────────────────────────────────────────────
        mse_val,  mse_grads  = jax.value_and_grad(mse_loss)(params)
        ph_val,   ph_grads   = jax.value_and_grad(phantom_rate_at_eq_loss)(params)
        pass_val, pass_grads = jax.value_and_grad(passivity_loss)(params)

        # ── Gradient norms ────────────────────────────────────────────────────
        def _norm(tree):
            return jnp.sqrt(sum(
                jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(tree)
            ) + 1e-12)

        mse_norm  = _norm(mse_grads)
        ph_norm   = _norm(ph_grads)
        pass_norm = _norm(pass_grads)

        # ── Gradient-normalised scales ────────────────────────────────────────
        scale_ph   = ALPHA_PHANTOM * mse_norm / (ph_norm   + 1e-8)
        scale_pass = ALPHA_PASS    * mse_norm / (pass_norm + 1e-8)

        # ── Combined gradient ─────────────────────────────────────────────────
        combined_grads = jax.tree_util.tree_map(
            lambda gm, gph, gp: gm + scale_ph * gph + scale_pass * gp,
            mse_grads, ph_grads, pass_grads,
        )

        updates, new_state = h_tx_p2.update(combined_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, mse_val, ph_val, pass_val, scale_ph, scale_pass

    # ─────────────────────────────────────────────────────────────────────────
    # R_net — DOF-weighted diagonal loss, two-phase
    # ─────────────────────────────────────────────────────────────────────────
    R_DOF_WEIGHTS = jnp.array([
        0.08, 0.08, 0.80, 0.40, 0.40, 0.15,
        1.00, 1.00, 1.00, 1.00,
        0.02, 0.02, 0.02, 0.02,
    ])

    r_schedule = optax.cosine_decay_schedule(init_value=1e-3,
                                              decay_steps=NUM_EPOCHS,
                                              alpha=0.01)
    r_tx = optax.adamw(learning_rate=r_schedule, weight_decay=1e-4)
    r_opt_state = r_tx.init(r_params)

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

    # ─────────────────────────────────────────────────────────────────────────
    # H_net training loop
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Neural Physics] Training H_net (Energy Landscape Residual)...")
    print(f"  [Config] Epochs: {NUM_EPOCHS} | AdamW | weight_decay: 1e-4")
    print(f"  [Two-Phase Design — v9: phantom rate at equilibrium]")
    print(f"  Phase 1 (ep 1→{PHASE_1_END}): PURE MSE.")
    print(f"    Network learns torsional energy landscape.")
    print(f"  Phase 2 (ep {PHASE_1_END+1}→{NUM_EPOCHS}): MSE + phantom-rate-at-eq + passivity.")
    print(f"    PhantomRate: (dH/dq·v)² at q=0 across physical p  [ALPHA={ALPHA_PHANTOM:.1f}]")
    print(f"      → directly targets spurious p-dependent q-gradient at diagnostic point.")
    print(f"    Passivity:   injection guard at random (q,p)        [ALPHA={ALPHA_PASS:.1f}]")
    print(f"    All terms gradient-normalised. Fresh Adam at Phase 2 start.")

    h_opt_state_p1 = h_tx_p1.init(h_params)
    h_opt_state_p2 = None

    for epoch in range(1, NUM_EPOCHS + 1):

        if epoch <= PHASE_1_END:
            h_params, h_opt_state_p1, h_loss = h_update_p1(
                h_params, h_opt_state_p1,
                q_data, p_data, setup_data, target_H_norm,
            )
            if epoch % 200 == 0:
                lr_now = float(h_schedule_p1(epoch))
                print(f"  Epoch {epoch:4d} | Loss: {h_loss:.6f} | "
                      f"lr: {lr_now:.2e} | Phase 1 (pure MSE)")

        else:
            if h_opt_state_p2 is None:
                h_opt_state_p2 = h_tx_p2.init(h_params)
                print(f"\n  [Phase 2 START] Epoch {epoch}: fresh Adam (LR=5e-4). "
                      f"PhantomRate@eq (α={ALPHA_PHANTOM:.1f}) + passivity (α={ALPHA_PASS:.1f}).")

            h_params, h_opt_state_p2, mse_l, ph_l, pass_l, sc_ph, sc_pass = h_update_p2(
                h_params, h_opt_state_p2,
                q_data, p_data, setup_data, target_H_norm,
            )
            if epoch % 200 == 0:
                p2_pct = int(100 * (epoch - PHASE_1_END) / PHASE_2_EPOCHS)
                lr_now = float(h_schedule_p2(epoch - PHASE_1_END))
                print(f"  Epoch {epoch:4d} | "
                      f"MSE: {float(mse_l):.6f} | "
                      f"PhRate: {float(ph_l):.1f} | "
                      f"Violation: {float(pass_l):.2f} J/s | "
                      f"PhScale: {float(sc_ph):.4f} | "
                      f"PassScale: {float(sc_pass):.3f} | "
                      f"lr: {lr_now:.2e} | "
                      f"Phase 2 ({p2_pct}%)")

    # ─────────────────────────────────────────────────────────────────────────
    # R_net training loop
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch <= R_PHASE_1_END:
            r_params, r_opt_state, r_loss = r_update(
                r_params, r_opt_state, q_data, p_data, target_R_mag_norm,
            )
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | "
                      f"lr: {float(r_schedule(epoch)):.2e}")
        else:
            if r_opt_state_p2 is None:
                r_opt_state_p2 = r_tx_p2.init(r_params)
                print(f"  [R Phase 2] Epoch {epoch}: fresh Adam, LR restart 5e-4.")
            r_params, r_opt_state_p2, r_loss = r_update_p2(
                r_params, r_opt_state_p2, q_data, p_data, target_R_mag_norm,
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
        budget_J = 0.10
        if abs(_dKE) < budget_J:
            _tag = f"✓ PASS  ({_dKE * 1000:.1f} mJ < {budget_J * 1000:.0f} mJ budget)"
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