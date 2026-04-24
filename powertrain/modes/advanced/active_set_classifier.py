# powertrain/modes/advanced/active_set_classifier.py
# Project-GP — Batch 2: Active-Set Classifier for mpQP Torque Allocator
# ═══════════════════════════════════════════════════════════════════════════════
#
# Predicts which of the 12 QP constraints are active at the optimum, given
# the normalised parameter vector θ ∈ ℝ¹⁵. Combined with the KKT linear
# solver (Batch 3), this replaces the 12-iteration projected-gradient SOCP.
#
# ARCHITECTURE:
#   θ (15) → Dense(128) → swish → Dense(128) → swish → Dense(64) → swish
#          → Dense(12) → sigmoid → active_set ∈ [0,1]¹²
#
#   Multi-label binary classification via BCE loss.
#   Platt calibration applied post-training (adjusts threshold per constraint).
#
# INFERENCE PROTOCOL (Batch 3 will use this):
#   1. active_set_probs = classifier.apply(params, theta_norm)          # 12 floats
#   2. active_set = (active_set_probs > threshold_per_constraint)       # 12 bools
#   3. Build reduced KKT system from active_set                         # (4+|A|)×(4+|A|)
#   4. T_opt = solve KKT system                                         # exact optimal
#   5. If KKT residual > tol: 3-step projected-gradient polish          # fallback
#
# TRAINING:
#   python -m powertrain.modes.advanced.active_set_classifier
#   Requires: models/qp_training_data.npz  (from generate_qp_training_data.py)
#   Outputs:  models/active_set_classifier.bytes
#             models/active_set_thresholds.npy
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
import optax
import numpy as np

# Matches generate_qp_training_data.py exactly
THETA_DIM     = 15
N_CONSTRAINTS = 12

# Normalisation scales — must match generate_qp_training_data.py exactly
MZ_SCALE    = 2000.0
FX_SCALE    = 8000.0
T_SCALE     = 320.0
DELTA_SCALE = 0.35

_THETA_SCALES = None   # lazy init to avoid module-load JAX overhead

def normalise_theta(theta_raw):
    global _THETA_SCALES
    if _THETA_SCALES is None:
        _THETA_SCALES = jnp.array([
            MZ_SCALE, FX_SCALE,
            T_SCALE, T_SCALE, T_SCALE, T_SCALE,   # T_min (4)
            T_SCALE, T_SCALE, T_SCALE, T_SCALE,   # T_max (4)
            T_SCALE, T_SCALE, T_SCALE, T_SCALE,   # T_fric (4)
            DELTA_SCALE,
        ])
    return theta_raw / _THETA_SCALES

# ─────────────────────────────────────────────────────────────────────────────
# §1  Model
# ─────────────────────────────────────────────────────────────────────────────

class ActiveSetClassifier(nn.Module):
    """
    Multi-label binary classifier: θ → P(constraint_i active) ∈ [0,1]¹².

    Sigmoid output — each constraint is independently predicted.
    During training: BCE loss on all 12 outputs simultaneously.
    During inference: threshold at calibrated per-constraint value.

    Width 128-128-64 is deliberately compact: the active-set function is
    piecewise-constant (it only changes at QP parameter boundaries), so
    a moderate-capacity network generalises well. Larger networks tend to
    overfit the boundary noise from finite-iteration training data.
    """
    hidden: tuple[int, ...] = (128, 128, 64)

    @nn.compact
    def __call__(self, theta: jax.Array) -> jax.Array:
        x = theta
        for i, h in enumerate(self.hidden):
            x = nn.Dense(h, name=f"dense_{i}")(x)
            x = nn.swish(x)
        # Raw logits → sigmoid probabilities
        logits = nn.Dense(N_CONSTRAINTS, name="out")(x)
        return nn.sigmoid(logits)


def init_classifier(rng: jax.Array) -> tuple[ActiveSetClassifier, dict]:
    model  = ActiveSetClassifier()
    params = model.init(rng, jnp.zeros(THETA_DIM))["params"]
    return model, params


# ─────────────────────────────────────────────────────────────────────────────
# §2  Loss — weighted BCE
# ─────────────────────────────────────────────────────────────────────────────

def make_loss_fn(model: ActiveSetClassifier, class_weights: jax.Array):
    """
    Weighted binary cross-entropy. class_weights[i] upweights constraint i
    when it is rarely active (handles class imbalance from interior solutions).

    Numerically stable: uses log-sum-exp form, never calls log(0).
    """
    @jax.jit
    def loss_fn(params: dict, theta_b: jax.Array, active_b: jax.Array) -> jaxp.Array:
        probs = jax.vmap(model.apply, in_axes=(None, 0))({"params": params}, theta_b)
        eps   = 1e-7
        bce   = -(
            active_b     * jnp.log(probs   + eps)
            + (1 - active_b) * jnp.log(1 - probs + eps)
        )  # (batch, 12)
        # Weight each constraint by inverse activation frequency
        weighted = bce * class_weights[None, :]
        return jnp.mean(weighted)

    return loss_fn

# typo fix — jax not jaxp
import jax.numpy as jnp  # noqa: F811


def make_loss_fn(model: ActiveSetClassifier, class_weights: jax.Array):  # noqa: F811
    @jax.jit
    def loss_fn(params: dict, theta_b: jax.Array, active_b: jax.Array) -> jax.Array:
        probs = jax.vmap(model.apply, in_axes=(None, 0))({"params": params}, theta_b)
        eps   = 1e-7
        bce   = -(
            active_b     * jnp.log(probs   + eps)
            + (1 - active_b) * jnp.log(1 - probs + eps)
        )
        return jnp.mean(bce * class_weights[None, :])

    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# §3  Platt calibration — adjusts per-constraint threshold post-training
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_thresholds(
    model: ActiveSetClassifier,
    params: dict,
    theta_val: np.ndarray,
    active_val: np.ndarray,
    target_recall: float = 0.97,
) -> np.ndarray:
    """
    Find per-constraint probability threshold that achieves target_recall.
    Recall is prioritised over precision: a false-positive active constraint
    costs one wasted row/column in the KKT solve (~microseconds); a false-
    negative misses an active constraint → KKT solution is infeasible and
    the polish fallback is invoked (~3 projected-gradient steps, still fast).

    Returns thresholds ∈ ℝ¹² — save alongside classifier weights.
    """
    probs = np.array(jax.vmap(
        model.apply, in_axes=(None, 0)
    )({"params": params}, jnp.array(theta_val)))   # (N_val, 12)

    thresholds = np.full(N_CONSTRAINTS, 0.5, dtype=np.float32)

    for c in range(N_CONSTRAINTS):
        y_true = active_val[:, c]
        p_c    = probs[:, c]
        if y_true.mean() < 0.01:
            # Near-never active — use 0.5, won't matter
            thresholds[c] = 0.5
            continue

        # Binary search for threshold giving ≥ target_recall
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            recall = ((p_c >= mid) & (y_true == 1)).sum() / (y_true.sum() + 1e-8)
            if recall >= target_recall:
                lo = mid
            else:
                hi = mid
        thresholds[c] = float(lo)

    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# §4  Training driver
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_path:  str = "models/qp_training_data.npz",
    out_params: str = "models/active_set_classifier.bytes",
    out_thresh: str = "models/active_set_thresholds.npy",
    epochs:     int = 3000,
    batch_size: int = 2048,
    lr_init:    float = 3e-4,
    val_frac:   float = 0.10,
    seed:       int = 0,
):
    print("[ActiveSetClassifier] Loading training data...")
    data = np.load(data_path)

    if 'geometry' in data:
        geo_train = data['geometry']
        from scripts.generate_qp_training_data import LF, LR, TF2, TR2, R_W
        geo_now   = np.array([LF, LR, TF2, TR2, R_W])
        if not np.allclose(geo_train, geo_now, atol=1e-3):
            raise ValueError(
                f"Geometry mismatch! Trained on {geo_train}, "
                f"current is {geo_now}. Retrain the classifier."
            )

    theta_norm  = data["theta_norm"]    # (N, 15)
    active_sets = data["active_sets"]   # (N, 12)
    N = theta_norm.shape[0]

    # Train / val split
    rng_np = np.random.default_rng(seed)
    idx    = rng_np.permutation(N)
    n_val  = int(N * val_frac)
    val_idx, trn_idx = idx[:n_val], idx[n_val:]

    theta_trn,  theta_val  = theta_norm[trn_idx],  theta_norm[val_idx]
    active_trn, active_val = active_sets[trn_idx], active_sets[val_idx]

    # Class weights: inverse frequency, capped at 10×
    freq   = active_trn.mean(axis=0) + 1e-3
    w      = np.clip(1.0 / freq, 1.0, 10.0)
    w      = w / w.mean()   # normalise so mean weight = 1
    class_weights = jnp.array(w.astype(np.float32))
    print(f"  Train: {len(trn_idx):,}  Val: {len(val_idx):,}")
    print(f"  Class weights: {w.round(2)}")
    print(f"  Activation rates: {active_trn.mean(axis=0).round(3)}")

    # Init
    rng      = jax.random.PRNGKey(seed)
    model, params = init_classifier(rng)
    loss_fn  = make_loss_fn(model, class_weights)

    schedule = optax.cosine_decay_schedule(lr_init, epochs, alpha=0.01)
    tx       = optax.adamw(schedule, weight_decay=1e-4)
    opt_state = tx.init(params)

    @jax.jit
    def step(params_, opt_state_, theta_b_, active_b_):
        loss_val, grads = jax.value_and_grad(loss_fn)(params_, theta_b_, active_b_)
        updates, new_state = tx.update(grads, opt_state_, params_)
        return optax.apply_updates(params_, updates), new_state, loss_val

    # Training loop
    print(f"\n[ActiveSetClassifier] Training {epochs} epochs  |  batch={batch_size}")
    best_val_loss = float("inf")
    best_params   = params

    for epoch in range(1, epochs + 1):
        # Minibatch
        perm  = rng_np.permutation(len(trn_idx))[:batch_size]
        t_b   = jnp.array(theta_trn[perm])
        a_b   = jnp.array(active_trn[perm])

        params, opt_state, trn_loss = step(params, opt_state, t_b, a_b)

        if epoch % 200 == 0:
            val_loss = float(loss_fn(params,
                                     jnp.array(theta_val[:4096]),
                                     jnp.array(active_val[:4096])))
            lr_now = float(schedule(epoch))

            # Per-constraint accuracy on validation set
            probs_val = np.array(jax.vmap(
                model.apply, in_axes=(None, 0)
            )({"params": params}, jnp.array(theta_val[:4096])))
            acc = ((probs_val > 0.5) == active_val[:4096]).mean(axis=0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params   = params

            print(f"  Epoch {epoch:5d} | trn={float(trn_loss):.4f}  "
                  f"val={val_loss:.4f}  "
                  f"acc_mean={acc.mean():.3f}  "
                  f"acc_min={acc.min():.3f}  "
                  f"lr={lr_now:.1e}")

    params = best_params

    # Calibrate thresholds
    print("\n[ActiveSetClassifier] Calibrating per-constraint thresholds...")
    thresholds = calibrate_thresholds(model, params, theta_val, active_val)
    print(f"  Thresholds: {thresholds.round(3)}")

    # Final accuracy at calibrated thresholds
    probs_val = np.array(jax.vmap(
        model.apply, in_axes=(None, 0)
    )({"params": params}, jnp.array(theta_val)))
    for c in range(N_CONSTRAINTS):
        y   = active_val[:, c]
        p   = probs_val[:, c] >= thresholds[c]
        tp  = ((p == 1) & (y == 1)).sum()
        fp  = ((p == 1) & (y == 0)).sum()
        fn  = ((p == 0) & (y == 1)).sum()
        rec = tp / (tp + fn + 1e-8)
        pre = tp / (tp + fp + 1e-8)
        print(f"  C{c:2d}: recall={rec:.3f}  precision={pre:.3f}  "
              f"thresh={thresholds[c]:.3f}  freq={y.mean():.3f}")

    # Save
    Path(out_params).parent.mkdir(parents=True, exist_ok=True)
    with open(out_params, "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    np.save(out_thresh, thresholds)
    print(f"\n[ActiveSetClassifier] Saved → {out_params}")
    print(f"[ActiveSetClassifier] Saved → {out_thresh}")

    return model, params, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# §5  Inference API (used by Batch 3 KKT solver)
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierBundle(NamedTuple):
    """Loaded classifier ready for JIT-compiled inference."""
    model:      ActiveSetClassifier
    params:     dict
    thresholds: jax.Array   # (12,) calibrated per-constraint thresholds


def load_classifier(
    params_path: str = "models/active_set_classifier.bytes",
    thresh_path:  str = "models/active_set_thresholds.npy",
) -> ClassifierBundle:
    model, dummy_params = init_classifier(jax.random.PRNGKey(0))
    with open(params_path, "rb") as f:
        params = flax.serialization.from_bytes(dummy_params, f.read())
    thresholds = jnp.array(np.load(thresh_path))
    return ClassifierBundle(model=model, params=params, thresholds=thresholds)


def make_predict_fns(bundle: ClassifierBundle):
    """Factory — closes over bundle so only arrays are traced."""
    @jax.jit
    def predict_active_set(theta_norm: jax.Array) -> jax.Array:
        probs = bundle.model.apply({"params": bundle.params}, theta_norm)
        return (probs >= bundle.thresholds).astype(jnp.float32)

    @jax.jit
    def predict_active_set_probs(theta_norm: jax.Array) -> jax.Array:
        return bundle.model.apply({"params": bundle.params}, theta_norm)

    return predict_active_set, predict_active_set_probs


# Thin wrappers for callers that still use the old two-argument API
def predict_active_set(bundle: ClassifierBundle, theta_norm: jax.Array) -> jax.Array:
    fn, _ = make_predict_fns(bundle)
    return fn(theta_norm)


def predict_active_set_probs(bundle: ClassifierBundle, theta_norm: jax.Array) -> jax.Array:
    _, fn = make_predict_fns(bundle)
    return fn(theta_norm)

# ── Batch 8 V2 constants ──────────────────────────────────────────────────────
 
THETA_DIM_V2     = 19    # 15 base + 4 slip (κ*_f, κ*_r, σ_f, σ_r)
N_CONSTRAINTS_V2 = 20    # 12 base + 8 slip (κ upper + lower, 4 wheels)
 
# Normalisation scales for the 4 new dimensions
KAPPA_SCALE = 0.20       # typical κ* range  [0.05, 0.20]
SIGMA_SCALE = 0.05       # typical σ(κ*) range [0.005, 0.05]
 
# Full V2 scale vector — must stay in sync with generate_qp_training_data.py
_THETA_SCALES_V2 = None
 
 
def normalise_theta_v2(theta_raw_v2):
    """Normalise a 19-dim raw θ for the V2 classifier."""
    global _THETA_SCALES_V2
    if _THETA_SCALES_V2 is None:
        _THETA_SCALES_V2 = jnp.array([
            MZ_SCALE,                                  # [0]  Mz_ref
            FX_SCALE,                                  # [1]  Fx_d
            T_SCALE, T_SCALE, T_SCALE, T_SCALE,        # [2:6]  T_min
            T_SCALE, T_SCALE, T_SCALE, T_SCALE,        # [6:10] T_max
            T_SCALE, T_SCALE, T_SCALE, T_SCALE,        # [10:14] T_fric
            DELTA_SCALE,                               # [14] delta
            KAPPA_SCALE, KAPPA_SCALE,                  # [15:17] κ*_f, κ*_r
            SIGMA_SCALE, SIGMA_SCALE,                  # [17:19] σ_f, σ_r
        ])
    return theta_raw_v2 / _THETA_SCALES_V2
 
 
# ── V2 Model ──────────────────────────────────────────────────────────────────
 
class ActiveSetClassifierV2(nn.Module):
    """
    V2 multi-label binary classifier: θ_v2(19) → P(constraint_i active) ∈ [0,1]^{20}.
 
    Same architecture as V1 (128-128-64 with swish) — just extended I/O.
    Trained on data that includes braking-into-corner scenarios where the
    8 slip-barrier constraints actively bind.
    """
    @nn.compact
    def __call__(self, theta_norm: jax.Array) -> jax.Array:
        x = nn.Dense(128)(theta_norm)
        x = nn.swish(x)
        x = nn.Dense(128)(x)
        x = nn.swish(x)
        x = nn.Dense(64)(x)
        x = nn.swish(x)
        return nn.sigmoid(nn.Dense(N_CONSTRAINTS_V2)(x))   # (20,) ∈ [0,1]
 
 
def init_classifier_v2(rng: jax.Array) -> tuple:
    model  = ActiveSetClassifierV2()
    params = model.init(rng, jnp.zeros(THETA_DIM_V2))["params"]
    return model, params
 
 
# ── V2 Calibration ────────────────────────────────────────────────────────────
 
def calibrate_thresholds_v2(
    model:       ActiveSetClassifierV2,
    params:      dict,
    theta_val:   np.ndarray,    # (N_val, 19)
    active_val:  np.ndarray,    # (N_val, 20)
    target_recall: float = 0.97,
) -> np.ndarray:
    """
    Per-constraint threshold calibration for V2 (20 outputs).
    Identical logic to V1 calibrate_thresholds — factored as separate fn
    to avoid mutating the V1 calibrated thresholds.
    """
    probs = np.array(jax.vmap(
        model.apply, in_axes=(None, 0)
    )({"params": params}, jnp.array(theta_val)))    # (N_val, 20)
 
    thresholds = np.full(N_CONSTRAINTS_V2, 0.5, dtype=np.float32)
 
    for c in range(N_CONSTRAINTS_V2):
        y_true = active_val[:, c]
        p_c    = probs[:, c]
        if y_true.mean() < 0.01:
            thresholds[c] = 0.5
            continue
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid    = 0.5 * (lo + hi)
            recall = ((p_c >= mid) & (y_true == 1)).sum() / (y_true.sum() + 1e-8)
            if recall >= target_recall:
                lo = mid
            else:
                hi = mid
        thresholds[c] = float(lo)
 
    return thresholds
 
 
# ── V2 Training driver ────────────────────────────────────────────────────────
 
def train_v2(
    data_path:  str = "models/qp_training_data_v2.npz",
    out_params: str = "models/active_set_classifier_v2.bytes",
    out_thresh: str = "models/active_set_thresholds_v2.npy",
    epochs:     int = 3000,
    batch_size: int = 2048,
    lr_init:    float = 3e-4,
    val_frac:   float = 0.10,
    seed:       int = 0,
):
    """
    Train the V2 classifier on the extended dataset.
 
    Requires: models/qp_training_data_v2.npz  (from generate_qp_training_data.py --v2)
      Keys: theta_norm (N, 19), active_sets (N, 20), T_opt (N, 4)
 
    Outputs:
      models/active_set_classifier_v2.bytes
      models/active_set_thresholds_v2.npy
 
    Run: python -m powertrain.modes.advanced.active_set_classifier --v2
    """
    print("[ActiveSetClassifierV2] Loading training data...")
    data        = np.load(data_path)
    
    if 'geometry' in data:
        geo_train = data['geometry']
        from scripts.generate_qp_training_data import LF, LR, TF2, TR2, R_W
        geo_now   = np.array([LF, LR, TF2, TR2, R_W])
        if not np.allclose(geo_train, geo_now, atol=1e-3):
            raise ValueError(
                f"Geometry mismatch! Trained on {geo_train}, "
                f"current is {geo_now}. Retrain the classifier."
            )

    theta_norm  = data["theta_norm"]       # (N, 19)
    active_sets = data["active_sets"]      # (N, 20)
    N           = theta_norm.shape[0]
 
    assert theta_norm.shape[1]  == THETA_DIM_V2,     \
        f"Expected θ dim {THETA_DIM_V2}, got {theta_norm.shape[1]}"
    assert active_sets.shape[1] == N_CONSTRAINTS_V2,  \
        f"Expected {N_CONSTRAINTS_V2} constraints, got {active_sets.shape[1]}"
 
    rng_np  = np.random.default_rng(seed)
    idx     = rng_np.permutation(N)
    n_val   = int(N * val_frac)
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
 
    theta_trn,  theta_val  = theta_norm[trn_idx],  theta_norm[val_idx]
    active_trn, active_val = active_sets[trn_idx], active_sets[val_idx]
 
    # Inverse-frequency class weights (same logic as V1)
    freq          = active_trn.mean(axis=0) + 1e-3
    w             = np.clip(1.0 / freq, 1.0, 10.0)
    w             = w / w.mean()
    class_weights = jnp.array(w.astype(np.float32))
    print(f"  Train: {len(trn_idx):,}  Val: {len(val_idx):,}")
 
    rng           = jax.random.PRNGKey(seed)
    model, params = init_classifier_v2(rng)
 
    @jax.jit
    def loss_fn(params_, theta_b_, active_b_):
        probs = jax.vmap(model.apply, in_axes=(None, 0))({"params": params_}, theta_b_)
        eps   = 1e-7
        bce   = -(active_b_ * jnp.log(probs + eps)
                  + (1 - active_b_) * jnp.log(1 - probs + eps))
        return jnp.mean(bce * class_weights[None, :])
 
    schedule  = optax.cosine_decay_schedule(lr_init, epochs, alpha=0.01)
    tx        = optax.adamw(schedule, weight_decay=1e-4)
    opt_state = tx.init(params)
 
    @jax.jit
    def step(params_, opt_state_, theta_b_, active_b_):
        loss_val, grads = jax.value_and_grad(loss_fn)(params_, theta_b_, active_b_)
        updates, new_st = tx.update(grads, opt_state_, params_)
        return optax.apply_updates(params_, updates), new_st, loss_val
 
    best_val_loss = float("inf")
    best_params   = params
 
    print(f"\n[ActiveSetClassifierV2] Training {epochs} epochs  |  batch={batch_size}")
    for epoch in range(1, epochs + 1):
        perm  = rng_np.permutation(len(trn_idx))[:batch_size]
        t_b   = jnp.array(theta_trn[perm])
        a_b   = jnp.array(active_trn[perm])
        params, opt_state, trn_loss = step(params, opt_state, t_b, a_b)
 
        if epoch % 200 == 0:
            val_loss = float(loss_fn(params,
                                     jnp.array(theta_val[:4096]),
                                     jnp.array(active_val[:4096])))
            lr_now = float(schedule(epoch))
            probs_v = np.array(jax.vmap(model.apply, in_axes=(None, 0))(
                {"params": params}, jnp.array(theta_val[:4096])))
            acc = ((probs_v > 0.5) == active_val[:4096]).mean(axis=0)
            print(f"  Epoch {epoch:5d} | trn={float(trn_loss):.4f}  "
                  f"val={val_loss:.4f}  "
                  f"acc_mean={acc.mean():.3f}  acc_min={acc.min():.3f}  "
                  f"lr={lr_now:.1e}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params   = params
 
    params = best_params
 
    # Calibrate thresholds
    print("\n[ActiveSetClassifierV2] Calibrating per-constraint thresholds...")
    thresholds = calibrate_thresholds_v2(model, params, theta_val, active_val)
    print(f"  V2 thresholds (20): {thresholds.round(3)}")
 
    Path(out_params).parent.mkdir(parents=True, exist_ok=True)
    with open(out_params, "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    np.save(out_thresh, thresholds)
    print(f"\n[ActiveSetClassifierV2] Saved → {out_params}")
    print(f"[ActiveSetClassifierV2] Saved → {out_thresh}")
    return model, params, thresholds
 
 
# ── V2 Load ───────────────────────────────────────────────────────────────────
 
def load_classifier_v2(
    params_path: str = "models/active_set_classifier_v2.bytes",
    thresh_path: str = "models/active_set_thresholds_v2.npy",
) -> ClassifierBundle:
    """
    Load V2 classifier weights into a ClassifierBundle.
 
    Returns ClassifierBundle with model=ActiveSetClassifierV2 — compatible
    with predict_active_set_soft() in explicit_mpqp_allocator.py.
    """
    model, dummy_params = init_classifier_v2(jax.random.PRNGKey(0))
    with open(params_path, "rb") as f:
        params = flax.serialization.from_bytes(dummy_params, f.read())
    thresholds = jnp.array(np.load(thresh_path))
    return ClassifierBundle(model=model, params=params, thresholds=thresholds)
 
 
# ── CLI: allow `python -m powertrain.modes.advanced.active_set_classifier --v2` ──
 
if __name__ == "__main__":
    import sys
    if "--v2" in sys.argv:
        print("[CLI] Training V2 classifier (19-dim θ, 20 constraints)")
        train_v2()
    else:
        # Default: existing V1 training (keep existing __main__ block if present,
        # else call train() from V1)
        print("[CLI] Training V1 classifier (15-dim θ, 12 constraints)")
        train()

