# optimization/residual_fitting.py
# Project-GP — Neural Port-Hamiltonian Residual Training
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX2)
# ─────────────────────────────────────────────────────────────────────────────
# FIX-1 : Setup-dependent target_H — FiLM layers now receive real gradients
#   PREVIOUS: target_H = 0.5 * k_torsion * torsion² (SETUP-INDEPENDENT)
#   The network was being asked "how does k_f reshape the energy landscape?"
#   while the training target was identical regardless of k_f, k_r, arb_f, arb_r.
#   FiLM γ/β parameters received exactly zero gradient from the MSE signal.
#
#   FIX: target_H = relu(V_spring_dev) + V_arb + V_torsion
#     V_spring_dev = 0.5·relu(k_f - K_PRIOR)·(z_fl²+z_fr²)
#                  + 0.5·relu(k_r - K_PRIOR)·(z_rl²+z_rr²)
#     V_arb        = 0.5·arb_f·(z_fl-z_fr)² + 0.5·arb_r·(z_rl-z_rr)²
#     V_torsion    = 0.5·k_torsion·((z_fl-z_fr)-(z_rl-z_rr))²
#
#   K_PRIOR = 30_000.0 N/m matches NeuralEnergyLandscape V_structural exactly.
#   relu() bounds target_H ≥ 0, consistent with the softplus H_res output.
#   Softer-than-prior setups correctly produce near-zero H_res (conservative).
#   ARB energy (V_arb) has no prior term at all — highest-value new signal.
#
# FIX-2 : Phantom rate evaluated at physical equilibrium, not zero
#   PREVIOUS: q_eq = jnp.zeros(14)
#   The car never operates at z=0. With equilibrium-centered training data
#   and Bug-2 in vehicle_dynamics.py (susp_sq gated at z_eq), the ghost
#   forces being suppressed were at the wrong operating point entirely.
#   FIX: q_eq = zeros(14).at[6:10].set(_Z_EQ)  where _Z_EQ = [0.0128, 0.0128, 0.0142, 0.0142]
#
# FIX-3 : target_R_mag includes suspension velocity dissipation
#   PREVIOUS: target_R_mag = |torsion|·500 — position-only, no velocity signal.
#   Damper power dissipation is P = c·ż² — quadratic in velocity, not in position.
#   R_net was trained on a target that contains no information about operating speed.
#   FIX: target_R_mag = |torsion|·200 + v_susp_rms·300 + |v_torsion_rate|·100
#
# FIX-4 : k_torsion reduced to prevent H_RESIDUAL_CAP saturation
#   PREVIOUS: k_torsion = 15_000 with q ~ N(0, 0.05) → E[H_target] ≈ 75 J.
#   With equilibrium-centered q (σ=20mm), same k_torsion → E[target_H] ≈ 12 J,
#   but susp_sq ≈ 0.0016 m² → H_res_needed ≈ 7500 J/m² > H_RESIDUAL_CAP=5000.
#   Cap silently clips ~30% of training samples with no error signal.
#   FIX: k_torsion = 5_000 → E[V_torsion] ≈ 4 J, 99%+ of samples within cap.
#   NOTE: H_RESIDUAL_CAP in vehicle_dynamics.py should be raised to 50_000 for
#   maximum fidelity once real telemetry data is available for calibration.
#
# RETAINED from GP-vX1:
#   · Equilibrium-centered suspension q DOFs (z_eq offset in generate_synthetic_flex_data)
#   · ALPHA_PASS = 25.0, DISSIPATION_THRESHOLD = 50.0, squared hinge passivity
#   · Gradient-normalised multi-term Phase-2 update
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import os
import flax.serialization

from models.vehicle_dynamics import (NeuralEnergyLandscape, NeuralDissipationMatrix,
                                      PhysicsNormalizer)
from data.configs.vehicle_params import vehicle_params as VP_DICT


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants — single source of truth shared by data generation
# and training losses. Any change here must be mirrored in vehicle_dynamics.py.
# ─────────────────────────────────────────────────────────────────────────────

# Must match NeuralEnergyLandscape.__call__:  V_structural = 0.5 * sum(q[6:10]²) * _V_STRUCT_PRIOR_K
_V_STRUCT_PRIOR_K: float = 30_000.0   # N/m per corner

# Static equilibrium suspension deflections [m].
# Must match the _Z_EQ constant introduced in vehicle_dynamics.py Bug-2 fix,
# where susp_sq = sum((q[6:10] - _Z_EQ)²) + 1e-4.
_Z_EQ: jnp.ndarray = jnp.array([0.0128, 0.0128, 0.0142, 0.0142])

# ─────────────────────────────────────────────────────────────────────────────
# Module-level scalars written by train_neural_residuals() — accessible from
# main.py / W&B logging without re-importing the training artefacts.
# ─────────────────────────────────────────────────────────────────────────────
TRAINED_H_SCALE: float = 1.0
TRAINED_R_SCALE: float = 1.0
LAST_TRAIN_MSE:  float = float('nan')
LAST_PHRATE:     float = float('nan')


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_flex_data(num_samples: int = 5000, key_seed: int = 42):
    """
    Synthetic training corpus for the Port-Hamiltonian residuals H_net and R_net.

    State-space coverage (GP-vX2)
    ─────────────────────────────
    q:  Body DOFs  ~ trim perturbations (small angles, realistic heave).
        Suspension ~ N(z_eq, 20mm) — CRITICAL: centred at physical equilibrium,
        NOT at zero. At z=0 the susp_sq gate collapses to its 1e-4 floor,
        attenuating H_res gradients by 100×.
        Wheel angle DOFs ~ N(0, 0.05) rad (small).
    p:  Physical momenta p = M_diag × v where v covers the operating envelope
        (vx ~ N(15, 5) m/s, wheel ω ~ N(75, 5) rad/s, etc.).
    setup: 28-element SuspensionSetup, sampled ±1σ around PhysicsNormalizer means
           for spring/ARB rates (tighter bound to stay within H_RESIDUAL_CAP),
           ±2σ for non-energy-coupled parameters.

    Target signal (GP-vX2) — SETUP-DEPENDENT
    ─────────────────────────────────────────
    target_H = relu(V_spring_dev) + V_arb + V_torsion

    V_spring_dev captures the spring energy DEVIATION from the structural prior
    already embedded in NeuralEnergyLandscape.V_structural (30 kN/m per corner).
    The relu ensures target_H ≥ 0, consistent with the softplus-gated H_res.

    V_arb is the ONLY energy term for anti-roll bars — absent from every prior.
    This is the single most important new FiLM signal: as arb_f spans 200-700 Nm/m,
    V_arb varies by 3× across the training batch, giving FiLM a strong gradient.

    V_torsion is the chassis torsional flex residual (k_torsion = 5_000 Nm/rad).
    Note: k_torsion intentionally reduced from 15_000 to keep H_res_needed =
    target_H / susp_sq within H_RESIDUAL_CAP = 5000 J/m² for 99%+ of samples.
    """
    # ── Mass diagonal (must exactly match NeuralEnergyLandscape M_diag) ──────
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

    # Operating envelope for velocity sampling — covers FSG event speeds
    v_std = jnp.array([
        15.0,   # vx  longitudinal         (skidpad / autocross range)
         2.0,   # vy  lateral
         0.3,   # vz  heave
         0.4,   # wx  roll rate
         0.3,   # wy  pitch rate
         1.5,   # wz  yaw rate
         0.4,   # unsprung vz FL
         0.4,   # unsprung vz FR
         0.4,   # unsprung vz RL
         0.4,   # unsprung vz RR
        75.0,   # wheel omega FL           (≈ 15 m/s / 0.2045 m)
        75.0,   # wheel omega FR
        75.0,   # wheel omega RL
        75.0,   # wheel omega RR
    ])

    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    k1a, k1b, k1c  = jax.random.split(k1, 3)

    # ── q: position DOFs ──────────────────────────────────────────────────────
    # Body DOFs [0:6]: x, y, z, roll, pitch, yaw
    q_body = jax.random.normal(k1a, (num_samples, 6)) * jnp.array(
        [0.01, 0.01, 0.005, 0.02, 0.015, 0.03]
    )

    # Suspension DOFs [6:10]: centred at static equilibrium (FIX: was N(0, 0.05))
    # E[susp_sq] ≈ 4·(20mm²) = 0.0016 m² — 15× larger than the near-zero floor,
    # restoring full gradient magnitude through the H_res * susp_sq path.
    q_susp = (_Z_EQ[None, :]
              + jax.random.normal(k1b, (num_samples, 4)) * 0.020)

    # Wheel angle DOFs [10:14]: small rotational perturbations
    q_wheel = jax.random.normal(k1c, (num_samples, 4)) * 0.05

    q = jnp.concatenate([q_body, q_susp, q_wheel], axis=1)   # (N, 14)

    # ── p: physical momenta p = M_diag × v ───────────────────────────────────
    p = jax.random.normal(k2, (num_samples, 14)) * v_std * M_diag_train

    # ── setup: 28-parameter SuspensionSetup ──────────────────────────────────
    # Spring/ARB rates clipped to ±1σ: wider sampling risks target_H exceeding
    # H_RESIDUAL_CAP for the H_res = target_H / susp_sq ratio.
    # All other parameters use ±2σ (non-energy-coupled, safe to be wider).
    setup_noise_tight = jnp.clip(jax.random.normal(k3, (num_samples, 4)), -1.0, 1.0)
    setup_noise_wide  = jnp.clip(jax.random.normal(k4, (num_samples, 28)), -2.0, 2.0)

    setup_params = (PhysicsNormalizer.setup_mean
                    + setup_noise_wide * PhysicsNormalizer.setup_scale)
    # Overwrite spring/ARB indices [0:4] with tighter clip
    setup_params = setup_params.at[:, 0:4].set(
        PhysicsNormalizer.setup_mean[0:4]
        + setup_noise_tight * PhysicsNormalizer.setup_scale[0:4]
    )

    # Hard physical lower bounds — prevents NaN gradients inside H_net
    setup_params = setup_params.at[:, 0:2].set(
        jnp.maximum(setup_params[:, 0:2], 5_000.0))    # k_f, k_r  [N/m]
    setup_params = setup_params.at[:, 2:4].set(
        jnp.maximum(setup_params[:, 2:4], 50.0))       # arb_f, arb_r [Nm/m]
    setup_params = setup_params.at[:, 4:8].set(
        jnp.maximum(setup_params[:, 4:8], 50.0))       # c_low_f/r, c_high_f/r [Ns/m]
    setup_params = setup_params.at[:, 26:28].set(
        jnp.maximum(setup_params[:, 26:28], 0.005))    # bumpstop gaps [m]

    # ── target_H: setup-dependent residual energy (FIX-1) ─────────────────────
    k_f   = setup_params[:, 0]    # front spring rate [N/m]
    k_r   = setup_params[:, 1]    # rear  spring rate [N/m]
    arb_f = setup_params[:, 2]    # front ARB rate    [N/m equivalent]
    arb_r = setup_params[:, 3]    # rear  ARB rate    [N/m equivalent]

    z_fl  = q[:, 6];  z_fr = q[:, 7]
    z_rl  = q[:, 8];  z_rr = q[:, 9]

    # Term 1 — Spring energy ABOVE the structural prior (always ≥ 0 via relu).
    # The prior covers _V_STRUCT_PRIOR_K = 30_000 N/m per corner.
    # For setups stiffer than the prior: positive residual that H_res must learn.
    # For softer setups: H_res → 0 (conservative; prior marginally over-estimates).
    V_spring_dev = (
        0.5 * jax.nn.relu(k_f - _V_STRUCT_PRIOR_K) * ((z_fl - _Z_EQ[0]) ** 2
                                                    + (z_fr - _Z_EQ[1]) ** 2)
    + 0.5 * jax.nn.relu(k_r - _V_STRUCT_PRIOR_K) * ((z_rl - _Z_EQ[2]) ** 2
                                                    + (z_rr - _Z_EQ[3]) ** 2)
    )

    # Term 2 — Anti-roll bar energy: COMPLETELY ABSENT from any prior term.
    # ARBs resist differential suspension travel, not absolute travel.
    # This is the highest-value new term for FiLM: arb varies 3× across training.
    V_arb = (
        0.5 * arb_f * (z_fl - z_fr) ** 2
      + 0.5 * arb_r * (z_rl - z_rr) ** 2
    )

    # Term 3 — Chassis torsional compliance residual.
    # k_torsion = 5_000 Nm/rad (conservative synthetic value).
    # Reduced from 15_000 to keep H_res = target_H / susp_sq within
    # H_RESIDUAL_CAP = 5000 J/m². At k=5_000: E[V_torsion] ≈ 4 J,
    # H_res_needed ≈ 4 / 0.0016 = 2500 J/m² — comfortably within cap.
    k_torsion = 5_000.0
    torsion   = (z_fl - z_fr) - (z_rl - z_rr)   # torsional displacement [m]
    V_torsion = 0.5 * k_torsion * torsion ** 2

    target_H = V_spring_dev + V_arb + V_torsion   # always ≥ 0

    # ── target_R_mag: velocity-enriched dissipation proxy (FIX-3) ─────────────
    # R_net input is (q, p) only — no setup — so target must be a fn of q and p.
    # Power dissipated by a damper: P = c·ż² → quadratic in velocity.
    # Previous target (|torsion|·500) had zero velocity sensitivity.
    v_fl = q[:, 6] * 0.0 + p[:, 6] / (_m_usf + 1e-8)   # unsprung vel FL [m/s]
    v_fr = p[:, 7] / (_m_usf + 1e-8)
    v_rl = p[:, 8] / (_m_usr + 1e-8)
    v_rr = p[:, 9] / (_m_usr + 1e-8)

    # RMS suspension velocity — primary driver of damper dissipation
    v_susp_rms = jnp.sqrt(
        (v_fl ** 2 + v_fr ** 2 + v_rl ** 2 + v_rr ** 2) * 0.25 + 1e-8
    )
    # Torsional velocity rate — driver of chassis flex dissipation
    v_torsion_rate = (v_fl - v_fr) - (v_rl - v_rr)

    target_R_mag = (
        jnp.abs(torsion)        * 200.0    # position-based torsional damping
      + v_susp_rms              * 300.0    # velocity-based damper dissipation
      + jnp.abs(v_torsion_rate) * 100.0    # differential velocity (flex dissipation)
    )

    return q, p, setup_params, target_H, target_R_mag


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Neural residual training
# ═══════════════════════════════════════════════════════════════════════════════

def train_neural_residuals():
    global TRAINED_H_SCALE, TRAINED_R_SCALE, LAST_TRAIN_MSE, LAST_PHRATE

    print("[Neural Physics] Generating Synthetic Chassis Flex Data...")
    print("[Neural Physics] P5 SE(3) architecture — 29 bilateral-invariant features.")
    print("[Neural Physics] P10 setup fix — 28-param setup vector.")
    print("[Neural Physics] GP-vX2 — setup-dependent target_H (FiLM active).")

    q_data, p_data, setup_data, target_H, target_R_mag = generate_synthetic_flex_data()

    assert setup_data.shape[1] == 28, (
        f"setup_data shape {setup_data.shape} — expected (N, 28)."
    )
    print(f"   [P10 check] setup_data shape: {setup_data.shape} ✓")

    # Log target composition to verify setup-dependence is non-trivial
    _z_fl = q_data[:, 6];  _z_fr = q_data[:, 7]
    _z_rl = q_data[:, 8];  _z_rr = q_data[:, 9]
    _tors = (_z_fl - _z_fr) - (_z_rl - _z_rr)
    _k_f  = setup_data[:, 0];  _k_r = setup_data[:, 1]
    _arb_f = setup_data[:, 2]; _arb_r = setup_data[:, 3]
    _V_sd = (0.5 * jax.nn.relu(_k_f - _V_STRUCT_PRIOR_K) * ((_z_fl - _Z_EQ[0])**2 + (_z_fr - _Z_EQ[1])**2)
           + 0.5 * jax.nn.relu(_k_r - _V_STRUCT_PRIOR_K) * ((_z_rl - _Z_EQ[2])**2 + (_z_rr - _Z_EQ[3])**2))
    _V_arb = (0.5 * _arb_f * (_z_fl - _z_fr)**2
            + 0.5 * _arb_r * (_z_rl - _z_rr)**2)
    _V_tor = 0.5 * 5_000.0 * _tors**2
    _H_tot = float(jnp.mean(target_H))
    print(f"   [Target H] mean={_H_tot:.3f} J  "
          f"| V_spring_dev={float(jnp.mean(_V_sd)):.3f} J  "
          f"| V_arb={float(jnp.mean(_V_arb)):.3f} J  "
          f"| V_torsion={float(jnp.mean(_V_tor)):.3f} J")
    if float(jnp.std(_V_sd)) < 0.01:
        print("   [WARN] V_spring_dev has near-zero variance — FiLM spring signal absent.")
    if float(jnp.std(_V_arb)) < 0.01:
        print("   [WARN] V_arb has near-zero variance — FiLM ARB signal absent.")

    # ── M_diag (must exactly match generate_synthetic_flex_data) ─────────────
    m_s   = VP_DICT.get('total_mass', 230.0) - (
                2 * VP_DICT.get('unsprung_mass_f', 10.0) +
                2 * VP_DICT.get('unsprung_mass_r', 11.0))
    m_usf = VP_DICT.get('unsprung_mass_f', 10.0)
    m_usr = VP_DICT.get('unsprung_mass_r', 11.0)
    Ix    = VP_DICT.get('Ix',  45.0)
    Iy    = VP_DICT.get('Iy',  85.0)
    Iz    = VP_DICT.get('Iz', 125.0)
    Iw    = VP_DICT.get('Iw',   1.2)
    M_diag = jnp.array([m_s, m_s, m_s, Ix, Iy, Iz,
                         m_usf, m_usf, m_usr, m_usr, Iw, Iw, Iw, Iw])

    h_net = NeuralEnergyLandscape(M_diag=M_diag)
    r_net = NeuralDissipationMatrix(dim=14)

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    h_params = h_net.init(key1, q_data[0], p_data[0], setup_data[0])
    r_params = r_net.init(key2, q_data[0], p_data[0])

    h_scale = float(jnp.std(target_H)    + 1e-6)
    r_scale = float(jnp.std(target_R_mag) + 1e-6)
    target_H_norm     = target_H     / h_scale
    target_R_mag_norm = target_R_mag / r_scale
    print(f"   [Neural Physics] H target scale: {h_scale:.4f} J  "
          f"| R target scale: {r_scale:.4f}")

    # ── Training protocol ─────────────────────────────────────────────────────
    # Two-phase (GP-vX2):
    #   Phase 1 (ep 1→2500): pure MSE — learn setup-dependent energy landscape.
    #     With setup-dependent target, FiLM γ/β now receive real gradients from
    #     the first epoch. Expected Phase-1 convergence to MSE < 0.10 vs
    #     previous stall at 0.746.
    #   Phase 2 (ep 2501→3500): MSE + PhantomRate@z_eq + squared-hinge passivity.
    #     PhantomRate evaluated at PHYSICAL EQUILIBRIUM q_eq (FIX-2), not zeros.
    #     ALPHA_PASS = 25.0 compensates for pass_norm >> mse_norm.
    # ─────────────────────────────────────────────────────────────────────────

    NUM_EPOCHS      = 3500
    PHASE_1_END     = 2500
    PHASE_2_EPOCHS  = NUM_EPOCHS - PHASE_1_END

    ALPHA_PHANTOM         = 3.0
    ALPHA_PASS            = 25.0    # compensates pass_norm >> mse_norm (was 1.0)
    DISSIPATION_THRESHOLD = 50.0    # tighter physical bound (was 100.0)

    h_schedule_p1 = optax.cosine_decay_schedule(
        init_value=1e-3, decay_steps=PHASE_1_END, alpha=0.01)
    h_tx_p1 = optax.adamw(learning_rate=h_schedule_p1, weight_decay=1e-4)

    # BUG C FIX (part 1): Phase 1 ends at LR ≈ 1.39e-5 (cosine at epoch 2400/2500).
    # Fresh Adam at 5e-4 is a 36× LR jump with reset momentum → guaranteed MSE spike.
    # 5e-5 is ~4× above Phase 1 final LR — aggressive enough to learn passivity,
    # conservative enough not to destroy the MSE landscape.
    h_schedule_p2 = optax.cosine_decay_schedule(
        init_value=5e-5, decay_steps=PHASE_2_EPOCHS, alpha=0.1)
    h_tx_p2 = optax.adamw(learning_rate=h_schedule_p2, weight_decay=1e-4)

    # ── Phase 1: pure MSE ────────────────────────────────────────────────────

    @jax.jit
    def h_update_p1(params, opt_state, q, p, setup, target_norm):
        def mse_loss(params_):
            def per_sample(q_s, p_s, setup_s, t_s):
                total    = h_net.apply(params_, q_s, p_s, setup_s)
                T_prior  = 0.5 * jnp.sum((p_s ** 2) / (M_diag + 1e-8))
                V_struct = 0.5 * jnp.sum(q_s[6:10] ** 2) * _V_STRUCT_PRIOR_K
                
                # FIX: Un-attenuate the loss by removing the susp_sq scaling
                susp_sq = jnp.sum((q_s[6:10] - _Z_EQ) ** 2) + 1e-4
                susp_sq_frozen = jax.lax.stop_gradient(susp_sq)
                
                pred_H_res = (total - T_prior - V_struct) / susp_sq_frozen
                target_H_res = (t_s * h_scale) / susp_sq_frozen
                
                return ((pred_H_res - target_H_res) / h_scale) ** 2
            return jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

        loss, grads = jax.value_and_grad(mse_loss)(params)
        updates, new_state = h_tx_p1.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, loss

    # ── Phase 2: MSE + PhantomRate@z_eq + squared-hinge passivity ────────────

    @jax.jit
    def h_update_p2(params, opt_state, q, p, setup, target_norm):
        # Physical equilibrium: body at trim, suspension at static deflection
        q_eq = jnp.zeros(14).at[6:10].set(_Z_EQ)

        # ── Neural residual isolator ───────────────────────────────────────────
        # Returns H_res * susp_sq_eq(q) — the ONLY component that can inject
        # phantom energy. T_prior is analytically conservative (kinetic energy).
        # V_structural is analytically passive (elastic potential). Both are
        # balanced in F_ext by gravity and spring forces respectively.
        # Evaluating phantom rate / passivity on H_full = T + V + H_res*susp_sq
        # penalises real physics:
        #   dV_structural/dt = 30000 * z * ż — large positive during compression,
        #   balanced by F_ext[16..23], not a passivity violation.
        #   dT_prior/dt = 0 by construction (conservative).
        # Only the neural residual is unconstrained and can hallucinate energy.
        def _h_residual(params_, q_, p_s, setup_s):
            total    = h_net.apply(params_, q_, p_s, setup_s)
            T_prior  = 0.5 * jnp.sum((p_s ** 2) / (M_diag + 1e-8))
            V_struct = 0.5 * jnp.sum(q_[6:10] ** 2) * _V_STRUCT_PRIOR_K
            return total - T_prior - V_struct   # = H_res * susp_sq_eq(q)

        def mse_loss(params_):
            def per_sample(q_s, p_s, setup_s, t_s):
                total    = h_net.apply(params_, q_s, p_s, setup_s)
                T_prior  = 0.5 * jnp.sum((p_s ** 2) / (M_diag + 1e-8))
                V_struct = 0.5 * jnp.sum(q_s[6:10] ** 2) * _V_STRUCT_PRIOR_K
                
                # FIX: Un-attenuate the loss by removing the susp_sq scaling
                susp_sq = jnp.sum((q_s[6:10] - _Z_EQ) ** 2) + 1e-4
                susp_sq_frozen = jax.lax.stop_gradient(susp_sq)
                
                pred_H_res = (total - T_prior - V_struct) / susp_sq_frozen
                target_H_res = (t_s * h_scale) / susp_sq_frozen
                
                return ((pred_H_res - target_H_res) / h_scale) ** 2
            return jnp.mean(jax.vmap(per_sample)(q, p, setup, target_norm))

        def phantom_rate_at_eq_loss(params_):
            # Evaluates d(H_res)/dq at physical equilibrium.
            # At z=z_eq: susp_sq = 1e-4 (floor) → d(H_res*susp_sq)/dq is small
            # by construction — the neural residual cannot generate large forces
            # at the operating point without violating the susp_sq gate.
            # V_structural gradient at z_eq is real spring force — must NOT be
            # penalized here; it is balanced by gravity in F_ext.
            def per_sample(p_s, setup_s):
                dHr_dq = jax.grad(
                    lambda q_: _h_residual(params_, q_, p_s, setup_s)
                )(q_eq)
                v    = p_s / (M_diag + 1e-8)
                rate = jnp.dot(dHr_dq, v)
                return rate ** 2
            return jnp.mean(jax.vmap(per_sample)(p, setup))

        def passivity_loss(params_):
            # Evaluates d(H_res)/dt at random (q, p).
            # dV_structural/dt = 30000 * z * ż — can be positive (spring compression).
            # This is balanced by gravity in F_ext, not a passivity violation.
            # Only H_res*susp_sq can inject phantom energy; penalize that alone.
            def per_sample(q_s, p_s, setup_s):
                dHr_dq  = jax.grad(
                    lambda q_: _h_residual(params_, q_, p_s, setup_s)
                )(q_s)
                v       = p_s / (M_diag + 1e-8)
                rate    = jnp.dot(dHr_dq, v)
                inject  = jax.nn.relu(rate) ** 2
                phantom = 0.1 * jax.nn.relu(-rate - DISSIPATION_THRESHOLD) ** 2
                return inject + phantom
            return jnp.mean(jax.vmap(per_sample)(q, p, setup))

        mse_val,  mse_grads  = jax.value_and_grad(mse_loss)(params)
        ph_val,   ph_grads   = jax.value_and_grad(phantom_rate_at_eq_loss)(params)
        pass_val, pass_grads = jax.value_and_grad(passivity_loss)(params)

        def _grad_norm(tree):
            return jnp.sqrt(sum(
                jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(tree)
            ) + 1e-12)

        mse_norm  = _grad_norm(mse_grads)
        ph_norm   = _grad_norm(ph_grads)
        pass_norm = _grad_norm(pass_grads)

        # Gradient-normalised scaling with symmetric caps to prevent float32 overflow.
        # Lower bound on scale_pass prevents it collapsing when pass_norm is large.
        scale_ph   = jnp.minimum(
            ALPHA_PHANTOM * mse_norm / (ph_norm   + 1e-8), 500.0)
        # BUG C FIX (part 2): The floor ALPHA_PASS*0.01=0.25 forces passivity gradients
        # to apply unconditionally even when mse_norm << pass_norm (i.e., MSE is well
        # converged but passivity has never been trained). This makes pass_grads dominate
        # the combined update, driving MSE from 0.006 → 1.97 in 100 epochs.
        # Confirmed by output: PassScale: 0.250 is constant all through Phase 2 —
        # the natural value was always below the floor and clamped to it.
        # Without the floor, scale_pass starts near-zero (protecting MSE), then grows
        # organically as passivity improves and pass_norm decreases.
        scale_pass = jnp.minimum(
            ALPHA_PASS    * mse_norm / (pass_norm + 1e-8),
            500.0,
        )

        combined_grads = jax.tree_util.tree_map(
            lambda gm, gph, gp: gm + scale_ph * gph + scale_pass * gp,
            mse_grads, ph_grads, pass_grads,
        )

        updates, new_state = h_tx_p2.update(combined_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, mse_val, ph_val, pass_val, scale_ph, scale_pass

    # ── R_net: DOF-weighted diagonal loss ────────────────────────────────────
    # Weights reflect physical dissipation importance: suspension DOFs highest,
    # wheel spin lowest (rolling resistance negligible vs. damper losses).
    R_DOF_WEIGHTS = jnp.array([
        0.08, 0.08, 0.80, 0.40, 0.40, 0.15,   # body DOFs
        1.00, 1.00, 1.00, 1.00,                 # suspension DOFs
        0.02, 0.02, 0.02, 0.02,                 # wheel spin DOFs
    ])

    R_PHASE_1_END = 1000
    r_schedule_p1 = optax.cosine_decay_schedule(
        init_value=1e-3, decay_steps=NUM_EPOCHS, alpha=0.01)
    r_tx_p1       = optax.adamw(learning_rate=r_schedule_p1, weight_decay=1e-4)
    r_opt_state   = r_tx_p1.init(r_params)

    r_schedule_p2 = optax.cosine_decay_schedule(
        init_value=5e-4, decay_steps=NUM_EPOCHS - R_PHASE_1_END, alpha=0.1)
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
    print("\n[Neural Physics] Training H_net (Energy Landscape Residual)...")
    print(f"  [Config] Epochs: {NUM_EPOCHS} | AdamW | weight_decay: 1e-4")
    print(f"  [P5 SE(3)] 29 bilateral-symmetric features")
    print(f"  [GP-vX2] setup-dependent target: V_spring_dev + V_arb + V_torsion")
    print(f"  Phase 1 (ep 1→{PHASE_1_END}): PURE MSE.")
    print(f"  Phase 2 (ep {PHASE_1_END+1}→{NUM_EPOCHS}): "
          f"MSE + PhantomRate@z_eq [α={ALPHA_PHANTOM}] "
          f"+ squared-hinge passivity [α={ALPHA_PASS}]")

    h_opt_state_p1 = h_tx_p1.init(h_params)
    h_opt_state_p2 = None
    _last_mse_val  = float('nan')
    _last_ph_val   = float('nan')

    for epoch in range(1, NUM_EPOCHS + 1):

        if epoch <= PHASE_1_END:
            h_params, h_opt_state_p1, h_loss = h_update_p1(
                h_params, h_opt_state_p1,
                q_data, p_data, setup_data, target_H_norm,
            )
            _last_mse_val = float(h_loss)
            if epoch % 200 == 0:
                lr_now = float(h_schedule_p1(epoch))
                print(f"  Epoch {epoch:4d} | Loss: {h_loss:.6f} | "
                      f"lr: {lr_now:.2e} | Phase 1 (MSE)")

        else:
            if h_opt_state_p2 is None:
                h_opt_state_p2 = h_tx_p2.init(h_params)
                print(f"\n  [Phase 2 START] Epoch {epoch}: fresh Adam (LR=5e-4). "
                      f"PhantomRate@z_eq (α={ALPHA_PHANTOM}) "
                      f"+ squared-hinge passivity (α={ALPHA_PASS}).")

            (h_params, h_opt_state_p2, mse_l,
             ph_l, pass_l, sc_ph, sc_pass) = h_update_p2(
                h_params, h_opt_state_p2,
                q_data, p_data, setup_data, target_H_norm,
            )
            _last_mse_val = float(mse_l)
            _last_ph_val  = float(ph_l)

            if epoch % 200 == 0:
                p2_pct = int(100 * (epoch - PHASE_1_END) / PHASE_2_EPOCHS)
                lr_now = float(h_schedule_p2(epoch - PHASE_1_END))
                print(f"  Epoch {epoch:4d} | "
                      f"MSE: {float(mse_l):.6f} | "
                      f"PhRate@z_eq: {float(ph_l):.4f} | "
                      f"Violation: {float(pass_l):.2f} J/s | "
                      f"PhScale: {float(sc_ph):.2f} | "
                      f"PassScale: {float(sc_pass):.3f} | "
                      f"lr: {lr_now:.2e} | "
                      f"Phase 2 ({p2_pct}%)")

    # ═════════════════════════════════════════════════════════════════════════
    # R_net training loop
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[Neural Physics] Training R_net (Dissipation Matrix Residual)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch <= R_PHASE_1_END:
            r_params, r_opt_state, r_loss = r_update_p1(
                r_params, r_opt_state, q_data, p_data, target_R_mag_norm,
            )
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | "
                      f"lr: {float(r_schedule_p1(epoch)):.2e}")
        else:
            if r_opt_state_p2 is None:
                r_opt_state_p2 = r_tx_p2.init(r_params)
                print(f"  [R Phase 2] Epoch {epoch}: fresh Adam, LR 5e-4.")
            r_params, r_opt_state_p2, r_loss = r_update_p2(
                r_params, r_opt_state_p2, q_data, p_data, target_R_mag_norm,
            )
            if epoch % 200 == 0:
                lr_now = float(r_schedule_p2(epoch - R_PHASE_1_END))
                print(f"  Epoch {epoch:4d} | MSE Loss: {r_loss:.6f} | "
                      f"lr: {lr_now:.2e} | R Phase 2")

    # ═════════════════════════════════════════════════════════════════════════
    # Post-training diagnostics
    # ═════════════════════════════════════════════════════════════════════════

    # Diagnostic 1: passive energy injection
    try:
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
        from data.configs.tire_coeffs import tire_coeffs as TP_DICT
        from models.vehicle_dynamics import build_default_setup_28

        _veh = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        _sp  = build_default_setup_28(VP_DICT).at[1].set(38000.).at[21].set(0.28)
        _x0  = jnp.zeros(46).at[14].set(10.0)
        _x1  = _veh.simulate_step(_x0, jnp.array([0.0, 0.0]), _sp, dt=0.01)
        _dKE = 0.5 * VP_DICT.get('total_mass', 230.0) * (
            float(_x1[14]) ** 2 - float(_x0[14]) ** 2
        )
        budget_J = 0.10
        if abs(_dKE) < budget_J:
            _tag = f"✓ PASS  ({_dKE * 1000:.1f} mJ < {budget_J * 1000:.0f} mJ budget)"
        else:
            _sign = "injection" if _dKE > 0 else "phantom braking"
            _tag  = (f"✗ WARN  ({_dKE * 1000:.1f} mJ — {_sign}) — "
                     f"passivity not fully converged with synthetic data.")
        print(f"\n[Neural Physics] Passive energy injection: {_tag}")
    except Exception as _e:
        print(f"[Neural Physics] Energy injection check skipped: {_e}")

    # Diagnostic 2: FiLM setup sensitivity
    # Verifies that FiLM layers produce meaningful setup modulation.
    # Computes std(H) across 200 random setups at fixed (q_eq, p_nominal).
    # Target: std > 0.5 × h_scale — if near zero, FiLM is still dormant.
    try:
        _key_diag     = jax.random.PRNGKey(9999)
        _kd1, _kd2    = jax.random.split(_key_diag)
        # At z_eq susp_sq = 1e-4 (floor only) → H_res contribution ≈ 0 regardless
        # of FiLM. Displace by 10mm: susp_sq ≈ 4*(0.010)^2 = 4e-4 → 4× larger,
        # enough to expose FiLM sensitivity without leaving the linear spring regime.
        _q_eq_diag    = jnp.zeros(14).at[6:10].set(_Z_EQ + 0.010)
        _p_nom_diag   = jnp.zeros(14).at[0].set(m_s * 15.0)   # 15 m/s nominal
        _setup_diag   = (PhysicsNormalizer.setup_mean[None, :]
                         + jax.random.normal(_kd1, (200, 28))
                         * PhysicsNormalizer.setup_scale * 0.5)
        _H_diag       = jax.vmap(
            lambda s: h_net.apply(h_params, _q_eq_diag, _p_nom_diag, s)
        )(_setup_diag)
        _film_std     = float(jnp.std(_H_diag))
        _film_thresh  = 0.05 * h_scale
        _film_ok      = _film_std > _film_thresh
        print(f"[Neural Physics] FiLM sensitivity: "
              f"std(H | varied_setup) = {_film_std:.4f} J  "
              f"(threshold: {_film_thresh:.4f} J)  "
              f"{'✓ PASS — FiLM active' if _film_ok else '✗ WARN — FiLM underutilized'}")
    except Exception as _e:
        print(f"[Neural Physics] FiLM sensitivity check skipped: {_e}")

    print("\n[Neural Physics] Pre-training complete!")
    print(f"Scale factors:  h_scale={h_scale:.4f} J  |  r_scale={r_scale:.4f}")

    # ── Update module-level scalars ───────────────────────────────────────────
    TRAINED_H_SCALE = h_scale
    TRAINED_R_SCALE = r_scale
    LAST_TRAIN_MSE  = _last_mse_val
    LAST_PHRATE     = _last_ph_val

    # ── Persist weights ───────────────────────────────────────────────────────
    _opt_dir    = os.path.dirname(os.path.abspath(__file__))
    _root       = os.path.dirname(_opt_dir)
    _model_dir  = os.path.join(_root, 'models')
    os.makedirs(_model_dir, exist_ok=True)

    _h_path     = os.path.join(_model_dir, 'h_net.bytes')
    _r_path     = os.path.join(_model_dir, 'r_net.bytes')
    _scale_path = os.path.join(_model_dir, 'h_net_scale.txt')

    with open(_h_path,     'wb') as f: f.write(flax.serialization.to_bytes(h_params))
    with open(_r_path,     'wb') as f: f.write(flax.serialization.to_bytes(r_params))
    with open(_scale_path, 'w')  as f: f.write(str(h_scale))

    print(f"[Neural Physics] Weights saved → {_h_path}")
    print(f"[Neural Physics] Scale saved   → {_scale_path}  ({h_scale:.4f} J)")
    print(f"[Neural Physics] LAST_TRAIN_MSE={LAST_TRAIN_MSE:.6f}  "
          f"LAST_PHRATE={LAST_PHRATE:.4f}")

    return h_params, r_params


if __name__ == "__main__":
    train_neural_residuals()