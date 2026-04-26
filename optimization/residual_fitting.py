# optimization/residual_fitting.py
# Project-GP — Neural Port-Hamiltonian Residual Training
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX3 — Batch 1 Passive Architecture)
# ─────────────────────────────────────────────────────────────────────────────
# CHANGE : Phase-2 Augmented Lagrangian passivity training RETIRED.
#   Passivity guaranteed algebraically by PassiveHNet (physics/h_net_icnn.py).
#
# CHANGE : NeuralEnergyLandscape replaced by PassiveHNet.
#   PassiveHNet.apply() → H_neural  (the neural residual ONLY, in J/m²).
#   Vehicle sees: H_neural * susp_sq  (via _full_H wrapper in vehicle_dynamics).
#   Training must match: train H_neural to predict target_H / susp_sq_per_sample.
#
# WHY J/m² (energy density):
#   target_H = V_spring_dev + V_arb + V_torsion  [Joules].
#   Vehicle uses H_neural * susp_sq to add this to H_total.
#   So H_neural * susp_sq ≈ target_H → H_neural ≈ target_H / susp_sq.
#   Training on target_H directly would mean the network outputs Joules but
#   the vehicle multiplies by susp_sq (~0.0016), giving forces 625× too small.
#   Training on target_H / susp_sq aligns training and inference exactly.
#   The susp_sq gate also attenuates q-gradients near equilibrium, preventing
#   phantom forces from an undertrained network (fixes the -1822 mJ bug).
#
# RETAINED (GP-vX2):
#   · generate_synthetic_flex_data() — unchanged
#   · All R_net code — unchanged
#   · Post-training diagnostics (energy injection, FiLM sensitivity)
#   · Serialisation to h_net.bytes / r_net.bytes / h_net_scale.txt
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import os
import flax.serialization

from physics.h_net_icnn import PassiveHNet
from models.vehicle_dynamics import (NeuralDissipationMatrix, PhysicsNormalizer)
from config.vehicles.ter26 import vehicle_params as VP_DICT


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

_V_STRUCT_PRIOR_K: float = 30_000.0
_Z_EQ: jnp.ndarray = jnp.array([0.0128, 0.0128, 0.0142, 0.0142])

TRAINED_H_SCALE: float = 1.0
TRAINED_R_SCALE: float = 1.0
LAST_TRAIN_MSE:  float = float('nan')
LAST_PHRATE:     float = float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Synthetic data generation  (unchanged from GP-vX2)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_flex_data(num_samples: int = 5000, key_seed: int = 42):
    _m_s   = VP_DICT.get('total_mass', 230.0) - (
                 2 * VP_DICT.get('unsprung_mass_f', 10.0) +
                 2 * VP_DICT.get('unsprung_mass_r', 11.0))
    _m_usf = VP_DICT.get('unsprung_mass_f', 10.0)
    _m_usr = VP_DICT.get('unsprung_mass_r', 11.0)
    _Ix    = VP_DICT.get('Ix',  45.0)
    _Iy    = VP_DICT.get('Iy',  85.0)
    _Iz    = VP_DICT.get('Iz', 125.0)
    _Iw    = VP_DICT.get('Iw',   1.2)

    M_diag_train = jnp.array([_m_s, _m_s, _m_s, _Ix, _Iy, _Iz,
                               _m_usf, _m_usf, _m_usr, _m_usr,
                               _Iw, _Iw, _Iw, _Iw])

    v_std = jnp.array([15.0, 2.0, 0.3, 0.4, 0.3, 1.5,
                        0.4, 0.4, 0.4, 0.4, 75.0, 75.0, 75.0, 75.0])

    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    k1a, k1b, k1c  = jax.random.split(k1, 3)

    q_body  = jax.random.normal(k1a, (num_samples, 6)) * jnp.array(
                  [0.01, 0.01, 0.005, 0.02, 0.015, 0.03])
    q_susp  = _Z_EQ[None, :] + jax.random.normal(k1b, (num_samples, 4)) * 0.020
    q_wheel = jax.random.normal(k1c, (num_samples, 4)) * 0.05
    q = jnp.concatenate([q_body, q_susp, q_wheel], axis=1)

    p = jax.random.normal(k2, (num_samples, 14)) * v_std * M_diag_train

    setup_noise_tight = jnp.clip(jax.random.normal(k3, (num_samples, 4)), -1.0, 1.0)
    setup_noise_wide  = jnp.clip(jax.random.normal(k4, (num_samples, 28)), -2.0, 2.0)
    setup_params = (PhysicsNormalizer.setup_mean
                    + setup_noise_wide * PhysicsNormalizer.setup_scale)
    setup_params = setup_params.at[:, 0:4].set(
        PhysicsNormalizer.setup_mean[0:4]
        + setup_noise_tight * PhysicsNormalizer.setup_scale[0:4])
    setup_params = setup_params.at[:, 0:2].set(jnp.maximum(setup_params[:, 0:2], 5_000.0))
    setup_params = setup_params.at[:, 2:4].set(jnp.maximum(setup_params[:, 2:4], 50.0))
    setup_params = setup_params.at[:, 4:8].set(jnp.maximum(setup_params[:, 4:8], 50.0))
    setup_params = setup_params.at[:, 26:28].set(jnp.maximum(setup_params[:, 26:28], 0.005))

    k_f   = setup_params[:, 0];  k_r   = setup_params[:, 1]
    arb_f = setup_params[:, 2];  arb_r = setup_params[:, 3]
    z_fl  = q[:, 6];  z_fr = q[:, 7]
    z_rl  = q[:, 8];  z_rr = q[:, 9]

    V_spring_dev = (
        0.5 * jax.nn.relu(k_f - _V_STRUCT_PRIOR_K) * ((z_fl - _Z_EQ[0])**2 + (z_fr - _Z_EQ[1])**2)
      + 0.5 * jax.nn.relu(k_r - _V_STRUCT_PRIOR_K) * ((z_rl - _Z_EQ[2])**2 + (z_rr - _Z_EQ[3])**2))
    V_arb     = 0.5 * arb_f * (z_fl - z_fr)**2 + 0.5 * arb_r * (z_rl - z_rr)**2
    torsion   = (z_fl - z_fr) - (z_rl - z_rr)
    V_torsion = 0.5 * 5_000.0 * torsion**2
    target_H  = V_spring_dev + V_arb + V_torsion   # [Joules], always ≥ 0

    # ── Convert to energy density: what PassiveHNet should output ─────────────
    # Vehicle sees H_neural * susp_sq. So H_neural = target_H / susp_sq.
    # susp_sq floor 1e-4 prevents division by zero at exact equilibrium.
    susp_sq = jnp.sum((q[:, 6:10] - _Z_EQ[None, :]) ** 2, axis=1) + 1e-4
    target_H_density = target_H / susp_sq   # [J/m²]

    v_fl = p[:, 6] / (_m_usf + 1e-8);  v_fr = p[:, 7] / (_m_usf + 1e-8)
    v_rl = p[:, 8] / (_m_usr + 1e-8);  v_rr = p[:, 9] / (_m_usr + 1e-8)
    v_susp_rms     = jnp.sqrt((v_fl**2 + v_fr**2 + v_rl**2 + v_rr**2) * 0.25 + 1e-8)
    v_torsion_rate = (v_fl - v_fr) - (v_rl - v_rr)
    target_R_mag   = (jnp.abs(torsion) * 200.0
                      + v_susp_rms * 300.0
                      + jnp.abs(v_torsion_rate) * 100.0)

    return q, p, setup_params, target_H_density, target_R_mag


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Neural residual training
# ═══════════════════════════════════════════════════════════════════════════════

def train_neural_residuals():
    global TRAINED_H_SCALE, TRAINED_R_SCALE, LAST_TRAIN_MSE, LAST_PHRATE

    print("[Neural Physics] Generating Synthetic Chassis Flex Data...")
    print("[Neural Physics] GP-vX3 — PassiveHNet (ICNN + gauge), Phase 2 AL retired.")
    print("[Neural Physics] Training target: H_density = target_H / susp_sq  [J/m²]")

    q_data, p_data, setup_data, target_H_density, target_R_mag = generate_synthetic_flex_data()

    assert setup_data.shape[1] == 28, f"setup_data shape {setup_data.shape} — expected (N, 28)."
    print(f"   [P10 check] setup_data shape: {setup_data.shape} ✓")

    # Diagnostic: target composition
    _z_fl = q_data[:, 6];  _z_fr = q_data[:, 7]
    _z_rl = q_data[:, 8];  _z_rr = q_data[:, 9]
    _tors  = (_z_fl - _z_fr) - (_z_rl - _z_rr)
    _k_f   = setup_data[:, 0];  _k_r  = setup_data[:, 1]
    _arb_f = setup_data[:, 2];  _arb_r = setup_data[:, 3]
    _susp_sq = jnp.sum((q_data[:, 6:10] - _Z_EQ[None, :]) ** 2, axis=1) + 1e-4
    _V_sd  = (0.5 * jax.nn.relu(_k_f - _V_STRUCT_PRIOR_K) * ((_z_fl - _Z_EQ[0])**2 + (_z_fr - _Z_EQ[1])**2)
            + 0.5 * jax.nn.relu(_k_r - _V_STRUCT_PRIOR_K) * ((_z_rl - _Z_EQ[2])**2 + (_z_rr - _Z_EQ[3])**2))
    _V_arb = 0.5 * _arb_f * (_z_fl - _z_fr)**2 + 0.5 * _arb_r * (_z_rl - _z_rr)**2
    _V_tor = 0.5 * 5_000.0 * _tors**2
    _target_J = _V_sd + _V_arb + _V_tor
    print(f"   [Target H]       mean={float(jnp.mean(_target_J)):.3f} J  "
          f"| V_spring_dev={float(jnp.mean(_V_sd)):.3f} J  "
          f"| V_arb={float(jnp.mean(_V_arb)):.3f} J  "
          f"| V_torsion={float(jnp.mean(_V_tor)):.3f} J")
    print(f"   [susp_sq]        mean={float(jnp.mean(_susp_sq)):.4e} m²")
    print(f"   [Target density] mean={float(jnp.mean(target_H_density)):.1f} J/m²  "
          f"| std={float(jnp.std(target_H_density)):.1f} J/m²")
    if float(jnp.std(_V_sd)) < 0.01:
        print("   [WARN] V_spring_dev near-zero variance — FiLM spring signal absent.")
    if float(jnp.std(_V_arb)) < 0.01:
        print("   [WARN] V_arb near-zero variance — FiLM ARB signal absent.")

    m_s   = VP_DICT.get('total_mass', 230.0) - (
                2 * VP_DICT.get('unsprung_mass_f', 10.0) +
                2 * VP_DICT.get('unsprung_mass_r', 11.0))
    m_usf = VP_DICT.get('unsprung_mass_f', 10.0)

    h_net  = PassiveHNet(q_dim=14, p_dim=14, setup_dim=28)
    r_net  = NeuralDissipationMatrix(dim=14)

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    h_params   = h_net.init(key1, q_data[0], p_data[0], setup_data[0])
    r_params   = r_net.init(key2, q_data[0], p_data[0])

    # h_scale is std of the DENSITY target [J/m²]
    h_scale           = float(jnp.std(target_H_density) + 1e-6)
    r_scale           = float(jnp.std(target_R_mag) + 1e-6)
    target_H_norm     = target_H_density / h_scale   # normalised density
    target_R_mag_norm = target_R_mag / r_scale
    print(f"   [Neural Physics] H density scale: {h_scale:.2f} J/m²  "
          f"| R target scale: {r_scale:.4f}")

    NUM_EPOCHS = 6000   # 2500→6000: P4 convergence requires more iterations
    h_schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=NUM_EPOCHS, alpha=0.005)
    h_tx       = optax.adamw(learning_rate=h_schedule, weight_decay=1e-4)

    # ── Phase 1: pure MSE on density ──────────────────────────────────────────
    # h_net outputs H_neural [J/m²]. Loss: (H_neural/h_scale - target_density_norm)²
    # h_scale ~ std(target_H / susp_sq) ~ 5000 J/m²
    @jax.jit
    def h_update_p1(params, opt_state, q, p, setup, target_norm):
        def combined_loss(params_):
            def per_sample(q_s, p_s, setup_s, t_s):
                H_neural = h_net.apply(params_, q_s, p_s, setup_s)
                l_mse = (H_neural / h_scale - t_s) ** 2

                # Explicit P4: p·∇_p H ≥ 0  (kinetic energy must be non-negative)
                # grad_H_p: (14,) gradient of H w.r.t. momenta p_s
                grad_H_p = jax.grad(
                    lambda p_: h_net.apply(params_, q_s, p_, setup_s)
                )(p_s)
                p_dot_grad = jnp.dot(p_s, grad_H_p)
                # softplus penalty: zero when p·∇H ≥ 0, positive when violated
                # λ_P4=100 makes this ~50× MSE weight at a 0.1 J violation
                l_p4 = 100.0 * jax.nn.softplus(-p_dot_grad * 0.01)
                return l_mse + l_p4

            return jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

        loss, grads = jax.value_and_grad(combined_loss)(params)
        updates, new_state = h_tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    # ── R_net ─────────────────────────────────────────────────────────────────
    R_DOF_WEIGHTS = jnp.array([
        0.08, 0.08, 0.80, 0.40, 0.40, 0.15,
        1.00, 1.00, 1.00, 1.00,
        0.02, 0.02, 0.02, 0.02,
    ])
    R_PHASE_1_END  = 1000
    r_schedule_p1  = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=NUM_EPOCHS, alpha=0.01)
    r_tx_p1        = optax.adamw(learning_rate=r_schedule_p1, weight_decay=1e-4)
    r_opt_state    = r_tx_p1.init(r_params)
    r_schedule_p2  = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=NUM_EPOCHS - R_PHASE_1_END, alpha=0.1)
    r_tx_p2        = optax.adamw(learning_rate=r_schedule_p2, weight_decay=1e-4)
    r_opt_state_p2 = None

    @jax.jit
    def r_loss_fn(params, q, p, target_mag_norm):
        preds     = jax.vmap(r_net.apply, in_axes=(None, 0, 0))(params, q, p)
        pred_diag = jax.vmap(jnp.diag)(preds)
        tgt_diag  = target_mag_norm[:, None] * R_DOF_WEIGHTS[None, :]
        return jnp.mean((pred_diag - tgt_diag) ** 2)

    @jax.jit
    def r_update_p1(params, opt_state, q, p, target_mag_norm):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target_mag_norm)
        updates, new_state = r_tx_p1.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    @jax.jit
    def r_update_p2(params, opt_state, q, p, target_mag_norm):
        loss, grads = jax.value_and_grad(r_loss_fn)(params, q, p, target_mag_norm)
        updates, new_state = r_tx_p2.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    # ═════════════════════════════════════════════════════════════════════════
    # H_net training loop
    # ═════════════════════════════════════════════════════════════════════════
    # ── FiLM contrastive pairs: same (q,p), different setups ─────────────────
    # Pre-compute paired indices where setup differs but state is identical.
    # Loss: H(q,p,s1) ≠ H(q,p,s2) when |target(s1)-target(s2)| is large.
    # This forces FiLM to modulate the network output based on setup.
    _rng_film = jax.random.PRNGKey(7777)
    _idx_a    = jax.random.randint(_rng_film, (512,), 0, len(q_data))
    _idx_b    = jax.random.randint(jax.random.fold_in(_rng_film, 1), (512,), 0, len(q_data))
    _q_film   = q_data[_idx_a];    _p_film   = p_data[_idx_a]
    _s_film_a = setup_data[_idx_a]; _s_film_b = setup_data[_idx_b]
    _t_film_a = target_H_norm[_idx_a]; _t_film_b = target_H_norm[_idx_b]

    @jax.jit
    def h_update_film(params, opt_state, q, p, setup, target_norm,
                      q_f, p_f, s_a, s_b, t_a, t_b):
        def film_loss(params_):
            # Standard MSE
            def per_sample(q_s, p_s, setup_s, t_s):
                H = h_net.apply(params_, q_s, p_s, setup_s)
                l_mse = (H / h_scale - t_s) ** 2
                grad_H_p = jax.grad(
                    lambda p_: h_net.apply(params_, q_s, p_, setup_s)
                )(p_s)
                l_p4 = 100.0 * jax.nn.softplus(-jnp.dot(p_s, grad_H_p) * 0.01)
                return l_mse + l_p4
            l_main = jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

            # FiLM contrastive: H(q,p,s_a) - H(q,p,s_b) ≈ t_a - t_b
            def film_pair(q_s, p_s, s_a_, s_b_, t_a_, t_b_):
                Ha = h_net.apply(params_, q_s, p_s, s_a_) / h_scale
                Hb = h_net.apply(params_, q_s, p_s, s_b_) / h_scale
                return ((Ha - Hb) - (t_a_ - t_b_)) ** 2
            l_film = jnp.mean(jax.vmap(film_pair)(q_f, p_f, s_a, s_b, t_a, t_b))

            return l_main + 2.0 * l_film   # 2× weight on FiLM contrastive

        loss, grads = jax.value_and_grad(film_loss)(params)
        updates, new_state = h_tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    print("\n[Neural Physics] Training H_net (Energy Density Residual)...")
    print(f"  [Config] Epochs: {NUM_EPOCHS} | AdamW | weight_decay: 1e-4")
    print(f"  [GP-vX3] PassiveHNet: ICNN + gauge. Passivity structural.")
    print(f"  [GP-vX3] Target: H_density = target_H / susp_sq  [J/m²]")
    print(f"  [GP-vX3] P4 explicit penalty λ=100 + FiLM contrastive loss (2×)")

    h_opt_state   = h_tx.init(h_params)
    _last_mse_val = float('nan')

    for epoch in range(1, NUM_EPOCHS + 1):
        h_params, h_opt_state, h_loss = h_update_film(
            h_params, h_opt_state,
            q_data, p_data, setup_data, target_H_norm,
            _q_film, _p_film, _s_film_a, _s_film_b, _t_film_a, _t_film_b,
        )
        _last_mse_val = float(h_loss)
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:5d} | Loss: {h_loss:.6f} | "
                  f"lr: {float(h_schedule(epoch)):.2e}")

    # ── Algebraic passivity check ─────────────────────────────────────────────
    print("\n[Neural Physics] Verifying structural passivity properties (P1–P4)...")
    try:
        from physics.passivity_verification import make_checkers
        from physics.h_net_icnn import _Z_EQ_DEFAULT
        import jax.random as jr

        check_P1, check_P2, check_P3, check_P4 = make_checkers(h_net)
        _rng = jr.PRNGKey(42)
        _kq, _kp, _ks = jr.split(_rng, 3)
        _q = 0.15 * jr.normal(_kq, (512, 14)); _q = _q.at[:, 6:10].add(_Z_EQ_DEFAULT[6:10])
        _p = 50.0 * jr.normal(_kp, (512, 14))
        _s = jr.uniform(_ks, (512, 28))
        _pp = h_params["params"] if isinstance(h_params, dict) and "params" in h_params else h_params

        _p1_min, _ = check_P1(_pp, _q, _p, _s)
        _p2_err    = check_P2(_pp, _s)
        _p3_err    = check_P3(_pp, _q, _s)
        _p4_min, _ = check_P4(_pp, _q, _p, _s)
        print(f"  P1 H≥0:         min={float(_p1_min):+.2e}  {'✓' if float(_p1_min) >= -1e-4 else '✗'}")
        print(f"  P2 H(eq,0)=0:   max|err|={float(_p2_err):.2e}  {'✓' if float(_p2_err) < 1e-4 else '✗'}")
        print(f"  P3 ∇pH(·,0)=0:  max‖∇‖={float(_p3_err):.2e}  {'✓' if float(_p3_err) < 1e-4 else '✗'}")
        print(f"  P4 p·∇pH≥0:     min={float(_p4_min):+.2e}  {'✓' if float(_p4_min) >= -1e-3 else '✗'}")
    except Exception as _ve:
        print(f"  [WARN] Passivity verification skipped: {_ve}")

    # ═════════════════════════════════════════════════════════════════════════
    # R_net training loop  (unchanged from GP-vX2)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch <= R_PHASE_1_END:
            r_params, r_opt_state, r_loss = r_update_p1(
                r_params, r_opt_state, q_data, p_data, target_R_mag_norm)
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:5d} | MSE Loss: {r_loss:.6f} | "
                      f"lr: {float(r_schedule_p1(epoch)):.2e}")
        else:
            if r_opt_state_p2 is None:
                r_opt_state_p2 = r_tx_p2.init(r_params)
                print(f"  [R Phase 2] Epoch {epoch}: fresh Adam, LR 5e-4.")
            r_params, r_opt_state_p2, r_loss = r_update_p2(
                r_params, r_opt_state_p2, q_data, p_data, target_R_mag_norm)
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:5d} | MSE Loss: {r_loss:.6f} | "
                      f"lr: {float(r_schedule_p2(epoch - R_PHASE_1_END)):.2e} | R Phase 2")

    # ═════════════════════════════════════════════════════════════════════════
    # Post-training diagnostics
    # ═════════════════════════════════════════════════════════════════════════
    try:
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle, build_default_setup_28
        from config.tire_coeffs import tire_coeffs as TP_DICT
        _veh = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        _sp  = build_default_setup_28(VP_DICT).at[1].set(38000.).at[21].set(0.28)
        _x0  = jnp.zeros(46).at[14].set(10.0)
        _x1  = _veh.simulate_step(_x0, jnp.array([0.0, 0.0]), _sp, dt=0.01)
        _dKE = 0.5 * VP_DICT.get('total_mass', 230.0) * (float(_x1[14])**2 - float(_x0[14])**2)
        budget_J = 0.10
        if abs(_dKE) < budget_J:
            _tag = f"✓ PASS  ({_dKE * 1000:.1f} mJ < {budget_J * 1000:.0f} mJ budget)"
        else:
            _sign = "injection" if _dKE > 0 else "phantom braking"
            _tag  = f"✗ WARN  ({_dKE * 1000:.1f} mJ — {_sign})"
        print(f"\n[Neural Physics] Passive energy injection: {_tag}")
    except Exception as _e:
        print(f"[Neural Physics] Energy injection check skipped: {_e}")

    try:
        _key_diag   = jax.random.PRNGKey(9999)
        _kd1, _kd2  = jax.random.split(_key_diag)
        _q_eq_diag  = jnp.zeros(14).at[6:10].set(_Z_EQ + 0.010)
        _p_nom_diag = jnp.zeros(14).at[0].set(m_s * 15.0)
        _setup_diag = (PhysicsNormalizer.setup_mean[None, :]
                       + jax.random.normal(_kd1, (200, 28)) * PhysicsNormalizer.setup_scale * 0.5)
        _H_diag     = jax.vmap(
            lambda s: h_net.apply(h_params, _q_eq_diag, _p_nom_diag, s)
        )(_setup_diag)
        _film_std   = float(jnp.std(_H_diag))
        _film_thresh = 0.10 * h_scale   # 10% of density scale
        _film_ok    = _film_std > _film_thresh
        print(f"[Neural Physics] FiLM sensitivity: "
              f"std(H_density | varied_setup) = {_film_std:.2f} J/m²  "
              f"(threshold: {_film_thresh:.2f} J/m²)  "
              f"{'✓ PASS' if _film_ok else '✗ WARN — FiLM underutilized'}")
    except Exception as _e:
        print(f"[Neural Physics] FiLM sensitivity check skipped: {_e}")

    print("\n[Neural Physics] Pre-training complete!")
    print(f"Scale factors:  h_scale={h_scale:.2f} J/m²  |  r_scale={r_scale:.4f}")

    TRAINED_H_SCALE = h_scale
    TRAINED_R_SCALE = r_scale
    LAST_TRAIN_MSE  = _last_mse_val
    LAST_PHRATE     = float('nan')

    _opt_dir   = os.path.dirname(os.path.abspath(__file__))
    _root      = os.path.dirname(_opt_dir)
    _model_dir = os.path.join(_root, 'models')
    os.makedirs(_model_dir, exist_ok=True)

    _h_path     = os.path.join(_model_dir, 'h_net.bytes')
    _r_path     = os.path.join(_model_dir, 'r_net.bytes')
    _scale_path = os.path.join(_model_dir, 'h_net_scale.txt')

    with open(_h_path,     'wb') as f: f.write(flax.serialization.to_bytes(h_params))
    with open(_r_path,     'wb') as f: f.write(flax.serialization.to_bytes(r_params))
    with open(_scale_path, 'w')  as f: f.write(str(h_scale))

    print(f"[Neural Physics] Weights saved → {_h_path}")
    print(f"[Neural Physics] LAST_TRAIN_MSE={LAST_TRAIN_MSE:.6f}")

    return h_params, r_params


if __name__ == "__main__":
    train_neural_residuals()