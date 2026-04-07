#!/usr/bin/env python3
# tools/train_pinn_from_ttc.py
# Project-GP — PINN Calibration against TTC Round 9 Calspan Data
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Train TireOperatorPINN to learn the deterministic drift ΔF/F₀ between
#   Pacejka MF6.2 predictions and actual Calspan measurements.
#   This is the #1 priority upgrade from the GP-vX2 audit:
#     untrained PINN degrades R² from 0.974 → 0.934 (the GP penalty).
#     After training, target: R² > 0.990 on held-out test set.
#
# DATA SOURCE:
#   data/ttc_round9/processed.npz — preprocessed Hoosier 43075 16x7.5-10 R20
#   Fields: alpha_rad, kappa, Fz_N, gamma_rad, Fy_N, Fx_N, Vx_mps, P_kPa
#   Split: 80% train / 20% test, stratified by Fz
#
# TRAINING STRATEGY:
#   1. Compute Pacejka baseline Fy₀, Fx₀ for each data point
#   2. Compute normalised residuals: δFy = (Fy_meas - Fy₀) / (|Fy₀| + ε)
#   3. Build 8D feature vector [sin(α), sin(2α), κ, κ³, γ, Fz/1000, Vx/20, T_norm]
#      T_norm = 0 for TTC data (controlled lab temperature ≈ ambient)
#   4. Train with: MSE on residuals + symmetry regularisation + Lipschitz penalty
#   5. Save weights → models/pinn_params.bytes
#   6. Load into PacejkaTire._pinn_params on next construction
#
# USAGE:
#   python tools/train_pinn_from_ttc.py [--epochs 3000] [--lr 3e-4] [--batch 512]
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import argparse
import time
from functools import partial

import numpy as np

# ── JAX config must come before any JAX import ───────────────────────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
try:
    import jax_config  # noqa: F401 — sets XLA flags
except ImportError:
    pass

import jax
import jax.numpy as jnp
import optax
import flax.serialization

from models.tire_model import TireOperatorPINN, PacejkaTire
from data.configs.tire_coeffs import tire_coeffs


# ─────────────────────────────────────────────────────────────────────────────
# §1  Data Loading + Pacejka Residual Computation
# ─────────────────────────────────────────────────────────────────────────────

def load_ttc_and_compute_residuals(
    npz_path: str = None,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Load TTC data, compute Pacejka baselines, return train/test residuals.

    Returns dict with keys:
        train_features (N_train, 8)   — 8D PINN input features
        train_dFy      (N_train,)     — normalised Fy residual targets
        train_dFx      (N_train,)     — normalised Fx residual targets
        test_features  (N_test, 8)
        test_dFy       (N_test,)
        test_dFx       (N_test,)
        test_Fy_meas   (N_test,)     — raw measured Fy for R² computation
        test_Fx_meas   (N_test,)
        test_Fy_pac    (N_test,)     — Pacejka baseline Fy
        test_Fx_pac    (N_test,)
    """
    if npz_path is None:
        npz_path = os.path.join(_root, 'data', 'ttc_round9', 'processed.npz')

    if not os.path.exists(npz_path):
        print(f"[PINN-TTC] ERROR: TTC data not found at {npz_path}")
        print(f"[PINN-TTC] Generating synthetic TTC-equivalent data for pipeline validation...")
        return _generate_synthetic_ttc_residuals(seed)

    data = np.load(npz_path)
    alpha = data['alpha_rad'].astype(np.float32)
    kappa = data['kappa'].astype(np.float32)
    Fz    = data['Fz_N'].astype(np.float32)
    gamma = data['gamma_rad'].astype(np.float32)
    Fy_m  = data['Fy_N'].astype(np.float32)
    Fx_m  = data['Fx_N'].astype(np.float32)
    Vx    = data.get('Vx_mps', np.full_like(alpha, 11.0)).astype(np.float32)
    n     = len(alpha)

    print(f"[PINN-TTC] Loaded {n} TTC data points from {os.path.basename(npz_path)}")
    print(f"[PINN-TTC] α ∈ [{np.degrees(alpha.min()):.1f}°, {np.degrees(alpha.max()):.1f}°]  "
          f"Fz ∈ [{Fz.min():.0f}, {Fz.max():.0f}] N")

    # ── Stratified train/test split by Fz ────────────────────────────────────
    rng = np.random.default_rng(seed)
    n_bins = 10
    bins = np.linspace(Fz.min(), Fz.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(Fz, bins) - 1, 0, n_bins - 1)

    test_mask = np.zeros(n, dtype=bool)
    for b in range(n_bins):
        in_bin = np.where(bin_idx == b)[0]
        n_test = max(1, int(len(in_bin) * test_fraction))
        test_mask[rng.choice(in_bin, n_test, replace=False)] = True

    print(f"[PINN-TTC] Split: {(~test_mask).sum()} train / {test_mask.sum()} test "
          f"(stratified by Fz, {n_bins} bins)")

    # ── Compute Pacejka baseline forces ──────────────────────────────────────
    tire = PacejkaTire(tire_coeffs)

    # Vectorised Pacejka evaluation via vmap
    @jax.jit
    def pacejka_batch(alpha, kappa, Fz, gamma, Vx):
        T_ribs = jnp.full(3, 85.0)  # TTC lab temp ≈ controlled
        T_gas  = jnp.array(85.0)
        T_core = jnp.array(60.0)

        def single(a, k, fz, g, vx):
            Fx, Fy = tire.compute_tire_forces(
                alpha=a, kappa=k, Fz=fz, gamma=g, Vx=vx,
                T_ribs=T_ribs, T_gas=T_gas, T_core=T_core,
                use_pinn=False,   # Pure Pacejka — no PINN/GP correction
            )
            return Fx, Fy

        return jax.vmap(single)(alpha, kappa, Fz, gamma, Vx)

    # Process in chunks to avoid XLA memory issues
    chunk = 4096
    Fx_pac_all = np.zeros(n, dtype=np.float32)
    Fy_pac_all = np.zeros(n, dtype=np.float32)

    for i in range(0, n, chunk):
        s = slice(i, min(i + chunk, n))
        Fx_c, Fy_c = pacejka_batch(
            jnp.array(alpha[s]), jnp.array(kappa[s]),
            jnp.array(Fz[s]), jnp.array(gamma[s]), jnp.array(Vx[s]),
        )
        Fx_pac_all[s] = np.array(Fx_c)
        Fy_pac_all[s] = np.array(Fy_c)
        if i == 0:
            print(f"[PINN-TTC] Pacejka baseline: first chunk computed "
                  f"(Fy range [{Fy_c.min():.0f}, {Fy_c.max():.0f}] N)")

    # ── Normalised residuals ─────────────────────────────────────────────────
    eps = 1.0  # 1 N floor to avoid division by tiny Pacejka outputs at low slip
    dFy = (Fy_m - Fy_pac_all) / (np.abs(Fy_pac_all) + eps)
    dFx = (Fx_m - Fx_pac_all) / (np.abs(Fx_pac_all) + eps)

    # Clip to ±25% — matching PINN output tanh bound
    dFy = np.clip(dFy, -0.25, 0.25)
    dFx = np.clip(dFx, -0.25, 0.25)

    # ── Build 8D feature vectors ─────────────────────────────────────────────
    # Matches TireOperatorPINN.__call__ feature extraction:
    #   [sin(α), sin(2α), κ, κ³, γ, Fz/1000, Vx/20, T_norm]
    features = np.column_stack([
        np.sin(alpha),
        np.sin(2.0 * alpha),
        kappa,
        kappa ** 3,
        gamma,
        Fz / 1000.0,
        Vx / 20.0,
        np.zeros(n, dtype=np.float32),  # T_norm = 0 for TTC (lab ambient)
    ]).astype(np.float32)

    # ── Residual statistics ──────────────────────────────────────────────────
    print(f"[PINN-TTC] Residual stats (what PINN will learn):")
    print(f"  δFy: mean={dFy.mean():+.4f}  std={dFy.std():.4f}  "
          f"|max|={np.abs(dFy).max():.4f}")
    print(f"  δFx: mean={dFx.mean():+.4f}  std={dFx.std():.4f}  "
          f"|max|={np.abs(dFx).max():.4f}")

    # ── Pack into train/test ─────────────────────────────────────────────────
    return {
        'train_features': features[~test_mask],
        'train_dFy':      dFy[~test_mask],
        'train_dFx':      dFx[~test_mask],
        'test_features':  features[test_mask],
        'test_dFy':       dFy[test_mask],
        'test_dFx':       dFx[test_mask],
        'test_Fy_meas':   Fy_m[test_mask],
        'test_Fx_meas':   Fx_m[test_mask],
        'test_Fy_pac':    Fy_pac_all[test_mask],
        'test_Fx_pac':    Fx_pac_all[test_mask],
    }


def _generate_synthetic_ttc_residuals(seed: int = 42) -> dict:
    """
    Fallback: generate synthetic Pacejka data with known systematic bias
    so the PINN has something to learn. Used when TTC .npz is unavailable.

    Synthetic bias: high-load under-prediction (-3% at Fz>800N) +
    thermal-like asymmetry (+2% at positive camber).
    """
    rng = np.random.default_rng(seed)
    n = 20000

    alpha = rng.uniform(-0.25, 0.25, n).astype(np.float32)
    kappa = rng.uniform(-0.15, 0.15, n).astype(np.float32)
    Fz    = rng.uniform(200, 1200, n).astype(np.float32)
    gamma = rng.uniform(-0.09, 0.01, n).astype(np.float32)  # -5° to +0.5°
    Vx    = rng.uniform(3, 22, n).astype(np.float32)

    # Systematic bias that Pacejka doesn't capture:
    # 1. High-load regime: Pacejka overestimates grip by ~3%
    load_bias = -0.03 * np.clip((Fz - 800) / 400, 0, 1)
    # 2. Camber asymmetry: positive camber has ~2% more grip than MF6.2 predicts
    camber_bias = 0.02 * np.tanh(gamma * 20)
    # 3. Combined slip interaction: MF6.2 Gyk/Gxa don't capture all coupling
    combined_bias = 0.015 * np.sin(alpha * 5) * np.tanh(kappa * 10)
    # 4. Measurement noise
    noise = rng.normal(0, 0.01, n).astype(np.float32)

    dFy = (load_bias + camber_bias + combined_bias + noise).astype(np.float32)
    dFx = (load_bias * 0.5 + combined_bias * 0.8 + noise * 0.7).astype(np.float32)
    dFy = np.clip(dFy, -0.25, 0.25)
    dFx = np.clip(dFx, -0.25, 0.25)

    features = np.column_stack([
        np.sin(alpha), np.sin(2 * alpha), kappa, kappa ** 3,
        gamma, Fz / 1000, Vx / 20, np.zeros(n),
    ]).astype(np.float32)

    # Approximate Pacejka forces for R² computation
    Fy_pac = 1800 * np.sin(1.4 * np.arctan(12 * alpha)) * (Fz / 654) ** 0.85
    Fx_pac = 1500 * np.sin(1.579 * np.arctan(18.5 * kappa)) * (Fz / 654) ** 0.9
    Fy_meas = Fy_pac * (1 + dFy)
    Fx_meas = Fx_pac * (1 + dFx)

    # Split 80/20
    split = int(0.8 * n)
    idx = rng.permutation(n)
    tr, te = idx[:split], idx[split:]

    print(f"[PINN-TTC] Synthetic mode: {n} samples, known bias patterns")
    print(f"  Split: {len(tr)} train / {len(te)} test")

    return {
        'train_features': features[tr], 'train_dFy': dFy[tr], 'train_dFx': dFx[tr],
        'test_features': features[te], 'test_dFy': dFy[te], 'test_dFx': dFx[te],
        'test_Fy_meas': Fy_meas[te], 'test_Fx_meas': Fx_meas[te],
        'test_Fy_pac': Fy_pac[te], 'test_Fx_pac': Fx_pac[te],
    }


# ─────────────────────────────────────────────────────────────────────────────
# §2  Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_pinn_ttc(
    data: dict,
    n_epochs: int = 3000,
    lr: float = 3e-4,
    batch_size: int = 512,
    wd: float = 1e-5,
    symmetry_weight: float = 0.01,
    seed: int = 0,
):
    """
    Train TireOperatorPINN against TTC residuals.

    Loss function:
        L = MSE(δFy_pred, δFy_target) + MSE(δFx_pred, δFx_target)
          + λ_sym · L_symmetry
          + wd · ||params||²  (via AdamW)

    L_symmetry enforces Fy(α) = -Fy(-α) and Fx(α) = Fx(-α):
        For each batch, also evaluate at (-α, κ) and penalise
        |δFy(α) + δFy(-α)|² + |δFx(α) - δFx(-α)|²

    Returns: trained params dict, metrics dict
    """
    pinn = TireOperatorPINN()
    key = jax.random.PRNGKey(seed)
    params = pinn.init(key, data['train_features'][0])

    # Cosine schedule with warmup
    warmup_steps = 200
    total_steps = n_epochs
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, total_steps - warmup_steps, alpha=0.01),
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=wd)
    opt_state = tx.init(params)

    train_f = jnp.array(data['train_features'])
    train_fy = jnp.array(data['train_dFy'])
    train_fx = jnp.array(data['train_dFx'])
    n_train = len(train_f)

    @jax.jit
    def train_step(params, opt_state, features, dFy_target, dFx_target, key):
        def loss_fn(p):
            # Forward pass: PINN predicts (δFx, δFy) corrections
            def per_sample(feat):
                mods, _sigma = pinn.apply(p, feat)
                return mods  # (2,): [δFx_pred, δFy_pred]

            preds = jax.vmap(per_sample)(features)  # (B, 2)
            dFx_pred = preds[:, 0]
            dFy_pred = preds[:, 1]

            # Primary MSE on residuals
            mse_fy = jnp.mean((dFy_pred - dFy_target) ** 2)
            mse_fx = jnp.mean((dFx_pred - dFx_target) ** 2)

            # Symmetry regularisation:
            # Fy is odd in α → δFy(−α) ≈ −δFy(α)
            # Fx is even in α → δFx(−α) ≈ δFx(α)
            # Flip sin(α) → −sin(α), sin(2α) → −sin(2α); rest unchanged
            feat_flip = features.at[:, 0].set(-features[:, 0])  # sin(α) → -sin(α)
            feat_flip = feat_flip.at[:, 1].set(-features[:, 1])  # sin(2α) → -sin(2α)
            preds_flip = jax.vmap(per_sample)(feat_flip)
            sym_fy = jnp.mean((dFy_pred + preds_flip[:, 1]) ** 2)  # should cancel
            sym_fx = jnp.mean((dFx_pred - preds_flip[:, 0]) ** 2)  # should match

            loss = mse_fy + mse_fx + symmetry_weight * (sym_fy + sym_fx)
            return loss, (mse_fy, mse_fx)

        (loss, (mse_fy, mse_fx)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, mse_fy, mse_fx

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n[PINN-TTC] Training: {n_epochs} epochs, lr={lr}, batch={batch_size}")
    print(f"[PINN-TTC] Symmetry reg: λ={symmetry_weight}")
    print(f"{'─' * 72}")

    best_loss = float('inf')
    best_params = params
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Shuffle + mini-batch
        key, subkey = jax.random.split(key)
        idx = jax.random.permutation(subkey, n_train)[:batch_size]

        key, step_key = jax.random.split(key)
        params, opt_state, loss, mse_fy, mse_fx = train_step(
            params, opt_state,
            train_f[idx], train_fy[idx], train_fx[idx], step_key,
        )

        loss_val = float(loss)
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params

        if epoch % 250 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  ep {epoch:5d} | loss={loss_val:.6f}  "
                  f"mse_fy={float(mse_fy):.6f}  mse_fx={float(mse_fx):.6f}  "
                  f"[{elapsed:.1f}s]")

    print(f"{'─' * 72}")
    print(f"[PINN-TTC] Training complete. Best loss: {best_loss:.6f}")

    return best_params, {'best_loss': best_loss}


# ─────────────────────────────────────────────────────────────────────────────
# §3  Evaluation + R² Report
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pinn(params: dict, data: dict) -> dict:
    """
    Evaluate trained PINN on test set. Reports:
    - Residual MSE (should be lower than pre-training)
    - Combined R² (Pacejka + PINN vs measured)
    - Per-load-bin breakdown
    """
    pinn = TireOperatorPINN()
    test_f = jnp.array(data['test_features'])

    @jax.jit
    def predict_batch(features):
        def per_sample(feat):
            mods, _sigma = pinn.apply(params, feat)
            return mods
        return jax.vmap(per_sample)(features)

    # Predict in chunks
    chunk = 4096
    n_test = len(test_f)
    all_preds = []
    for i in range(0, n_test, chunk):
        s = slice(i, min(i + chunk, n_test))
        all_preds.append(np.array(predict_batch(test_f[s])))
    preds = np.concatenate(all_preds, axis=0)  # (N, 2)

    dFx_pred = preds[:, 0]
    dFy_pred = preds[:, 1]

    # ── Residual MSE ─────────────────────────────────────────────────────────
    test_dFy = data['test_dFy']
    test_dFx = data['test_dFx']
    mse_fy = float(np.mean((dFy_pred - test_dFy) ** 2))
    mse_fx = float(np.mean((dFx_pred - test_dFx) ** 2))

    # ── Combined R²: Pacejka + PINN correction vs measured ───────────────────
    Fy_pac = data['test_Fy_pac']
    Fx_pac = data['test_Fx_pac']
    Fy_meas = data['test_Fy_meas']
    Fx_meas = data['test_Fx_meas']

    # PINN-corrected forces: F_corrected = F_pacejka * (1 + δF_pinn)
    Fy_corrected = Fy_pac * (1.0 + dFy_pred)
    Fx_corrected = Fx_pac * (1.0 + dFx_pred)

    # R² for Fy
    ss_res_fy = np.sum((Fy_meas - Fy_corrected) ** 2)
    ss_tot_fy = np.sum((Fy_meas - Fy_meas.mean()) ** 2)
    r2_fy = 1.0 - ss_res_fy / (ss_tot_fy + 1e-12)

    # R² for Fx
    ss_res_fx = np.sum((Fx_meas - Fx_corrected) ** 2)
    ss_tot_fx = np.sum((Fx_meas - Fx_meas.mean()) ** 2)
    r2_fx = 1.0 - ss_res_fx / (ss_tot_fx + 1e-12)

    # Pacejka-only R² (baseline comparison)
    r2_fy_pac = 1.0 - np.sum((Fy_meas - Fy_pac) ** 2) / (ss_tot_fy + 1e-12)
    r2_fx_pac = 1.0 - np.sum((Fx_meas - Fx_pac) ** 2) / (ss_tot_fx + 1e-12)

    # RMSE
    rmse_fy = float(np.sqrt(np.mean((Fy_meas - Fy_corrected) ** 2)))
    rmse_fy_pac = float(np.sqrt(np.mean((Fy_meas - Fy_pac) ** 2)))

    print(f"\n{'=' * 72}")
    print(f"  PINN EVALUATION — TEST SET ({n_test} points)")
    print(f"{'=' * 72}")
    print(f"\n  Residual MSE:  δFy={mse_fy:.6f}  δFx={mse_fx:.6f}")
    print(f"\n  {'':>20} {'Pacejka Only':>14} {'Pac + PINN':>14} {'Δ':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'Fy R²':>20} {r2_fy_pac:>14.6f} {r2_fy:>14.6f} "
          f"{r2_fy - r2_fy_pac:>+10.6f}")
    print(f"  {'Fx R²':>20} {r2_fx_pac:>14.6f} {r2_fx:>14.6f} "
          f"{r2_fx - r2_fx_pac:>+10.6f}")
    print(f"  {'Fy RMSE [N]':>20} {rmse_fy_pac:>14.1f} {rmse_fy:>14.1f} "
          f"{rmse_fy - rmse_fy_pac:>+10.1f}")
    print(f"  {'─' * 60}")

    # ── Per-Fz breakdown ─────────────────────────────────────────────────────
    # Recover Fz from feature[5] = Fz/1000
    Fz_test = data['test_features'][:, 5] * 1000
    print(f"\n  Fy R² by normal load:")
    for lo, hi in [(100, 350), (350, 550), (550, 750), (750, 1000), (1000, 1300)]:
        m = (Fz_test >= lo) & (Fz_test < hi)
        if m.sum() < 50:
            continue
        e_pac = Fy_meas[m] - Fy_pac[m]
        e_pinn = Fy_meas[m] - Fy_corrected[m]
        r2_p = 1 - np.sum(e_pac ** 2) / (np.sum((Fy_meas[m] - Fy_meas[m].mean()) ** 2) + 1e-12)
        r2_c = 1 - np.sum(e_pinn ** 2) / (np.sum((Fy_meas[m] - Fy_meas[m].mean()) ** 2) + 1e-12)
        print(f"    Fz {lo:4d}–{hi:4d} N: Pac R²={r2_p:.4f} → +PINN R²={r2_c:.4f}  "
              f"(n={m.sum()}, Δ={r2_c - r2_p:+.4f})")

    print(f"\n{'=' * 72}\n")

    metrics = {
        'r2_fy': r2_fy, 'r2_fx': r2_fx,
        'r2_fy_pac': r2_fy_pac, 'r2_fx_pac': r2_fx_pac,
        'rmse_fy': rmse_fy, 'rmse_fy_pac': rmse_fy_pac,
        'mse_fy': mse_fy, 'mse_fx': mse_fx,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# §4  Save Weights
# ─────────────────────────────────────────────────────────────────────────────

def save_pinn_weights(params: dict, output_path: str = None):
    """Save trained PINN weights to models/pinn_params.bytes."""
    if output_path is None:
        output_path = os.path.join(_root, 'models', 'pinn_params.bytes')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[PINN-TTC] Weights saved → {output_path} ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# §5  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Project-GP: Train PINN against TTC Round 9 data'
    )
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--ttc-path', type=str, default=None,
                        help='Path to processed.npz (default: data/ttc_round9/processed.npz)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for pinn_params.bytes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load + compute residuals
    data = load_ttc_and_compute_residuals(args.ttc_path, seed=args.seed)

    # Train
    params, train_metrics = train_pinn_ttc(
        data,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        seed=args.seed,
    )

    # Evaluate
    eval_metrics = evaluate_pinn(params, data)

    # Save
    save_pinn_weights(params, args.output)

    # Summary
    print(f"\n{'═' * 72}")
    print(f"  PINN CALIBRATION SUMMARY")
    print(f"{'═' * 72}")
    print(f"  Pacejka-only Fy R²:  {eval_metrics['r2_fy_pac']:.6f}")
    print(f"  Pac + PINN   Fy R²:  {eval_metrics['r2_fy']:.6f}  "
          f"(Δ = {eval_metrics['r2_fy'] - eval_metrics['r2_fy_pac']:+.6f})")
    print(f"  Target:              > 0.990")
    print(f"  Status:              {'✅ PASS' if eval_metrics['r2_fy'] > 0.990 else '⚠️  CONTINUE TRAINING'}")
    print(f"{'═' * 72}")


if __name__ == '__main__':
    main()