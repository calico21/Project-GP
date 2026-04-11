# powertrain/modes/advanced/koopman_tv.py
# Project-GP — Dictionary-Switched Koopman TV Controller (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces the reactive PD yaw moment generator in tv_step with a
# Dictionary-Switched Koopman LQR that is predictive, physically grounded,
# and formally dual to the existing Diff-WMPC wavelet control parameterisation.
#
# Architecture:
#   Three local Koopman operators, each trained on one grip utilisation regime:
#     Regime 0  ρ < 0.70  — linear grip      m₀ = 32
#     Regime 1  0.70–0.92 — transition       m₁ = 64
#     Regime 2  ρ > 0.92  — saturation       m₂ = 32
#
#   Online pipeline at 200 Hz (≈ 0.18 ms total):
#     1. ρ  = grip_utilisation(Fx, Fy, Fz, μ)       ← one-step-delayed Fx
#     2. w  = blend_weights(ρ)                        ← soft Gaussians
#     3. e  = normalise_error(ψ̇_err, vy, vx, δ)
#     4. z_k = φ_k.apply(params_k, e)   ∀ k ∈ {0,1,2}
#     5. Mz_k = -L_k @ z_k              ← precomputed Riccati gains
#     6. Mz  = Σ w_k · Mz_k
#     7. Mz  = blend · Mz + (1-blend) · Mz_PD        ← smooth training ramp
#
#   The trained_blend field in KoopmanTVConfig controls the ramp:
#     0.0 → pure PD fallback (identical to pre-Phase-2 behaviour)
#     1.0 → full Koopman LQR (post-training)
#
#   All ops are C∞, purely JAX — safe inside jit/grad/vmap/scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanTVConfig(NamedTuple):
    """
    Static configuration for the Dictionary-Switched Koopman TV controller.

    All floats here are Python scalars — they act as compile-time constants
    inside @jax.jit (shape-static), except trained_blend which is converted
    to a JAX array inside the traced function to allow value-updates without
    recompilation.
    """
    # ── Regime partition-of-unity ──────────────────────────────────────
    rho_c0: float = 0.35          # Gaussian centre for regime 0
    rho_c1: float = 0.81          # Gaussian centre for regime 1
    rho_c2: float = 0.96          # Gaussian centre for regime 2
    sigma_blend: float = 0.12     # Gaussian bandwidth (all regimes)

    # ── MLP lifting dimensions ─────────────────────────────────────────
    m0: int = 32
    m1: int = 64
    m2: int = 32
    hidden: int = 64              # hidden layer width (all three MLPs)

    # ── Input normalisation (maps e → [-1, 1] for MLP health) ─────────
    wz_scale:    float = 2.0      # rad/s  (clip ±wz_scale before normalising)
    vy_scale:    float = 5.0      # m/s
    vx_scale:    float = 30.0     # m/s
    delta_scale: float = 0.3      # rad

    # ── LQR cost weights (offline Riccati solve, not used at runtime) ──
    Q_wz: float = 200.0
    R_Mz: float = 0.01

    # ── Trained blend ramp ─────────────────────────────────────────────
    # 0.0 = pure PD fallback  →  1.0 = full Koopman LQR
    trained_blend: float = 0.0

    # ── PD fallback (active when trained_blend < 1.0) ─────────────────
    Kp_fallback: float = 80.0    # Nm / (rad/s)
    Kd_fallback: float = 5.0     # Nm / (rad/s²)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Koopman Operators (trained, stored as JAX arrays)
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanOperators(NamedTuple):
    """
    Per-regime Koopman matrices.  Produced by train_koopman_tv.py and stored
    as .npy files; loaded via load_koopman_bundle() at runtime.

    K_k ∈ ℝ^{m_k × m_k}  autonomous dynamics in lifted space
    b_k ∈ ℝ^{m_k}         control input coupling (Mz is scalar)
    L_k ∈ ℝ^{m_k}         Riccati LQR gain  (Mz* = −L_k @ z_k)
    """
    # Regime 0
    K0: jax.Array   # (m0, m0)
    b0: jax.Array   # (m0,)
    L0: jax.Array   # (m0,)
    # Regime 1
    K1: jax.Array   # (m1, m1)
    b1: jax.Array   # (m1,)
    L1: jax.Array   # (m1,)
    # Regime 2
    K2: jax.Array   # (m2, m2)
    b2: jax.Array   # (m2,)
    L2: jax.Array   # (m2,)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Bundle (operators + phi params + config, passed as a single pytree)
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanTVBundle(NamedTuple):
    """
    Everything the online Koopman controller needs at runtime.
    Passed as a single pytree argument to koopman_mz_reference().
    """
    ops:    KoopmanOperators
    phi0:   Any     # Flax param pytree for LiftingMap regime 0
    phi1:   Any     # Flax param pytree for LiftingMap regime 1
    phi2:   Any     # Flax param pytree for LiftingMap regime 2
    cfg:    KoopmanTVConfig


# ─────────────────────────────────────────────────────────────────────────────
# §4  Lifting Maps (Flax MLPs)
# ─────────────────────────────────────────────────────────────────────────────

class LiftingMap(nn.Module):
    """
    3-layer tanh MLP: ℝ^4 → ℝ^m_k.

    Input: normalised error state e = [ψ̇_err_n, vy_n, vx_n, δ_n]
    Output: lifted coordinates z ∈ ℝ^{m_k}

    tanh activations are C∞ and bounded — essential for JAX grad stability
    through the Koopman generator loss during training.
    """
    hidden:  int
    out_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.out_dim)(x)
        return x


# Module singletons — architecture is compile-time constant; only params vary.
# Instantiated once at module import; apply() is called with dynamic params.
_PHI0 = LiftingMap(hidden=64, out_dim=32)
_PHI1 = LiftingMap(hidden=64, out_dim=64)
_PHI2 = LiftingMap(hidden=64, out_dim=32)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Grip Utilisation Index ρ
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def grip_utilisation(
    Fx: jax.Array,   # (4,) longitudinal forces [N]
    Fy: jax.Array,   # (4,) lateral forces [N]
    Fz: jax.Array,   # (4,) vertical loads [N]
    mu: jax.Array,   # scalar estimated friction coefficient
) -> jax.Array:
    """
    Scalar grip utilisation index ρ ∈ [0, 1].

    ρ = max_i sqrt(Fx_i² + Fy_i²) / (μ · Fz_i)

    Smooth max via softmax-weighted sum (avoids non-differentiable hard max).
    """
    friction_limit = jnp.maximum(mu * Fz, 1.0)           # (4,) [N]
    utilisation    = jnp.sqrt(Fx**2 + Fy**2) / friction_limit  # (4,) ∈ [0, ∞)

    # Soft max: temperature τ = 20 → close to hard max, still C∞
    tau    = 20.0
    w_soft = jax.nn.softmax(tau * utilisation)
    rho    = jnp.sum(w_soft * utilisation)

    return jnp.clip(rho, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Soft Partition-of-Unity Blending Weights
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def blend_weights(
    rho: jax.Array,
    cfg: KoopmanTVConfig = KoopmanTVConfig(),
) -> jax.Array:
    """
    Soft blending weights w ∈ Δ² (3-simplex) via normalised Gaussians.

        w_k(ρ) = exp(−(ρ − ρ_k*)² / (2σ²)) / Z

    Returns (3,) weight vector summing to 1.
    Differentiable everywhere; w_k → 1 near ρ_k*, 0 far away.
    """
    centres = jnp.array([cfg.rho_c0, cfg.rho_c1, cfg.rho_c2])
    log_w   = -0.5 * ((rho - centres) / cfg.sigma_blend) ** 2
    return jax.nn.softmax(log_w)          # (3,) ∈ Δ²


# ─────────────────────────────────────────────────────────────────────────────
# §7  Error State Normalisation
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def normalise_error(
    wz_err: jax.Array,
    vy:     jax.Array,
    vx:     jax.Array,
    delta:  jax.Array,
    cfg:    KoopmanTVConfig = KoopmanTVConfig(),
) -> jax.Array:
    """
    Normalise the 4-component error state to approximately [-1, 1].

    Clipping before division prevents gradient explosion from out-of-distribution
    states; tanh provides an additional smooth bound inside the MLP.

    Returns e_norm ∈ ℝ^4.
    """
    wz_n    = jnp.clip(wz_err, -cfg.wz_scale,    cfg.wz_scale)    / cfg.wz_scale
    vy_n    = jnp.clip(vy,     -cfg.vy_scale,    cfg.vy_scale)    / cfg.vy_scale
    vx_n    = jnp.clip(vx,      0.0,              cfg.vx_scale)   / cfg.vx_scale
    delta_n = jnp.clip(delta,  -cfg.delta_scale,  cfg.delta_scale) / cfg.delta_scale
    return jnp.array([wz_n, vy_n, vx_n, delta_n])


# ─────────────────────────────────────────────────────────────────────────────
# §8  Main Online Function — koopman_mz_reference
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def koopman_mz_reference(
    wz_err:  jax.Array,     # scalar ψ̇_ref − ψ̇ [rad/s]
    dwz_err: jax.Array,     # scalar d/dt(ψ̇_err) [rad/s²]  for PD fallback D-term
    vy:      jax.Array,     # lateral velocity [m/s]
    vx:      jax.Array,     # longitudinal velocity [m/s]
    delta:   jax.Array,     # steering angle [rad]
    Fx:      jax.Array,     # (4,) longitudinal wheel forces [N]  (T_prev / r_w)
    Fy:      jax.Array,     # (4,) lateral tire forces [N]
    Fz:      jax.Array,     # (4,) vertical loads [N]
    mu_est:  jax.Array,     # scalar friction estimate
    bundle:  KoopmanTVBundle,
) -> tuple[jax.Array, jax.Array]:
    """
    Dictionary-Switched Koopman LQR yaw moment reference.

    Replaces the PD block:
        Mz = Kp * wz_err + Kd * dwz_err

    With:
        Mz = Σ_k w_k(ρ) · (−L_k @ φ_k(e))   [Koopman LQR]

    Blended with a PD fallback via cfg.trained_blend for safe deployment.

    Returns:
        Mz_target : scalar [Nm]
        rho_util  : scalar [0, 1]  — grip utilisation (diagnostics)
    """
    cfg = bundle.cfg
    ops = bundle.ops

    # ── Grip utilisation (one-step-delayed Fx from T_prev) ───────────────
    rho = grip_utilisation(Fx, Fy, Fz, mu_est)

    # ── Soft blending weights ─────────────────────────────────────────────
    w = blend_weights(rho, cfg)               # (3,)

    # ── Normalised error state ────────────────────────────────────────────
    e_norm = normalise_error(wz_err, vy, vx, delta, cfg)   # (4,)

    # ── Lift into Koopman space (three small MLP forward passes) ─────────
    z0 = _PHI0.apply(bundle.phi0, e_norm)     # (m0,)
    z1 = _PHI1.apply(bundle.phi1, e_norm)     # (m1,)
    z2 = _PHI2.apply(bundle.phi2, e_norm)     # (m2,)

    # ── LQR control law in lifted space: Mz_k = −L_k @ z_k ──────────────
    Mz0 = -jnp.dot(ops.L0, z0)               # scalar
    Mz1 = -jnp.dot(ops.L1, z1)               # scalar
    Mz2 = -jnp.dot(ops.L2, z2)               # scalar

    # ── Blend across regimes ──────────────────────────────────────────────
    Mz_koopman = w[0] * Mz0 + w[1] * Mz1 + w[2] * Mz2   # scalar

    # ── PD fallback (Kp + Kd, identical to legacy tv_step behaviour) ─────
    Mz_pd = cfg.Kp_fallback * wz_err + cfg.Kd_fallback * dwz_err

    # ── Smooth blend: 0.0 = pure PD  →  1.0 = full Koopman ──────────────
    # trained_blend as jnp array so value updates don't trigger recompilation
    blend = jnp.clip(jnp.array(cfg.trained_blend, dtype=jnp.float32), 0.0, 1.0)
    Mz_target = blend * Mz_koopman + (1.0 - blend) * Mz_pd

    return Mz_target, rho


# ─────────────────────────────────────────────────────────────────────────────
# §9  Koopman Generator (for training — not used at runtime)
# ─────────────────────────────────────────────────────────────────────────────

def koopman_generator(
    phi_apply_fn,          # callable: (e,) → z  (single sample)
    e:       jax.Array,    # (4,) error state
    f_e:     jax.Array,    # (4,) time derivative of e  (from bicycle model)
) -> jax.Array:
    """
    Exact Koopman generator applied to the lifting map φ:

        (ℒφ)(e) = ⟨∇_e φ(e), f(e, u)⟩ = JVP of φ in direction f_e

    This is zero-bias — no finite-difference approximation, no trajectory rollout.
    Used exclusively during offline training (train_koopman_tv.py).

    Args:
        phi_apply_fn : lambda e → z  (Module.apply with params already bound)
        e            : single (4,) state sample
        f_e          : (4,) vector field at (e, u)

    Returns: Lφ ∈ ℝ^m_k  (Lie derivative)
    """
    _, Lphi = jax.jvp(phi_apply_fn, (e,), (f_e,))
    return Lphi


# ─────────────────────────────────────────────────────────────────────────────
# §10  Factory — Inert Default Bundle (PD-equivalent, no training required)
# ─────────────────────────────────────────────────────────────────────────────

def _inert_operators(cfg: KoopmanTVConfig = KoopmanTVConfig()) -> KoopmanOperators:
    """
    Returns PD-equivalent Koopman operators.

    K_k = 0.95 · I  (stable, slightly decaying autonomous dynamics)
    b_k = 0          (control decoupled — PD fallback handles it)
    L_k = 0          (zero gain — full weight on PD fallback while blend = 0.0)

    With cfg.trained_blend = 0.0, the Koopman branch is inactive and the
    controller is identical to the pre-Phase-2 PD controller. Ramp
    trained_blend → 1.0 after validating trained operators.
    """
    def _inert_regime(m: int):
        K = 0.95 * jnp.eye(m)
        b = jnp.zeros(m)
        L = jnp.zeros(m)
        return K, b, L

    K0, b0, L0 = _inert_regime(cfg.m0)
    K1, b1, L1 = _inert_regime(cfg.m1)
    K2, b2, L2 = _inert_regime(cfg.m2)

    return KoopmanOperators(
        K0=K0, b0=b0, L0=L0,
        K1=K1, b1=b1, L1=L1,
        K2=K2, b2=b2, L2=L2,
    )


def make_default_koopman_bundle(cfg: KoopmanTVConfig = KoopmanTVConfig()) -> KoopmanTVBundle:
    """
    Construct a fully inert KoopmanTVBundle usable before training.

    The bundle is valid for immediate deployment; trained_blend=0.0 ensures
    the PD fallback is active. After running train_koopman_tv.py, call
    load_koopman_bundle() to hot-swap the operators.
    """
    key = jax.random.PRNGKey(42)
    dummy_e = jnp.zeros(4)

    # Initialise MLP params with zero-mean Glorot weights.
    # Output is ~0 everywhere with these params; PD fallback is fully active.
    phi0_params = _PHI0.init(key, dummy_e)
    phi1_params = _PHI1.init(key, dummy_e)
    phi2_params = _PHI2.init(key, dummy_e)

    return KoopmanTVBundle(
        ops  = _inert_operators(cfg),
        phi0 = phi0_params,
        phi1 = phi1_params,
        phi2 = phi2_params,
        cfg  = cfg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §11  I/O — Load Trained Bundle from Disk
# ─────────────────────────────────────────────────────────────────────────────

def load_koopman_bundle(
    path:          str  = "trained/koopman_tv",
    trained_blend: float = 1.0,
    cfg:           KoopmanTVConfig = KoopmanTVConfig(),
) -> KoopmanTVBundle:
    """
    Load trained operators from .npy files produced by train_koopman_tv.py.

    Expected files under `path/`:
        K0.npy, b0.npy, L0.npy, phi0_params.pkl
        K1.npy, b1.npy, L1.npy, phi1_params.pkl
        K2.npy, b2.npy, L2.npy, phi2_params.pkl

    Args:
        path          : directory containing saved operator files
        trained_blend : override for cfg.trained_blend (ramp to 1.0 gradually)
        cfg           : KoopmanTVConfig to embed in the bundle

    Returns: KoopmanTVBundle ready for deployment.
    """
    import os
    import pickle
    import numpy as np

    def load(name: str) -> jax.Array:
        return jnp.array(np.load(os.path.join(path, name)))

    def load_params(name: str) -> Any:
        with open(os.path.join(path, name), 'rb') as f:
            return pickle.load(f)

    ops = KoopmanOperators(
        K0=load("K0.npy"), b0=load("b0.npy"), L0=load("L0.npy"),
        K1=load("K1.npy"), b1=load("b1.npy"), L1=load("L1.npy"),
        K2=load("K2.npy"), b2=load("b2.npy"), L2=load("L2.npy"),
    )

    # Embed trained_blend override into config
    cfg_deployed = KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": trained_blend})

    return KoopmanTVBundle(
        ops  = ops,
        phi0 = load_params("phi0_params.pkl"),
        phi1 = load_params("phi1_params.pkl"),
        phi2 = load_params("phi2_params.pkl"),
        cfg  = cfg_deployed,
    )


def save_koopman_bundle(bundle: KoopmanTVBundle, path: str = "trained/koopman_tv") -> None:
    """Save a trained bundle to disk (called from train_koopman_tv.py)."""
    import os
    import pickle
    import numpy as np

    os.makedirs(path, exist_ok=True)
    ops = bundle.ops

    for name, arr in [
        ("K0", ops.K0), ("b0", ops.b0), ("L0", ops.L0),
        ("K1", ops.K1), ("b1", ops.b1), ("L1", ops.L1),
        ("K2", ops.K2), ("b2", ops.b2), ("L2", ops.L2),
    ]:
        np.save(os.path.join(path, f"{name}.npy"), np.array(arr))

    for name, params in [
        ("phi0_params", bundle.phi0),
        ("phi1_params", bundle.phi1),
        ("phi2_params", bundle.phi2),
    ]:
        with open(os.path.join(path, f"{name}.pkl"), 'wb') as f:
            pickle.dump(params, f)

    print(f"[KoopmanTV] Bundle saved to {path}/")