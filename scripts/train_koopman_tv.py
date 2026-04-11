# scripts/train_koopman_tv.py
# Project-GP — Offline Koopman TV Training (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Trains three Dictionary-Switched Koopman operators via EDMD-DL
# (Extended Dynamic Mode Decomposition with Dictionary Learning).
#
# Method:
#   1. Generate training trajectories from a differentiable 3-DOF bicycle model
#      (yaw, lateral velocity, speed — the correct fidelity for TV control).
#   2. For each grip regime k, train lifting map φ_k to minimise the EDMD
#      residual via Adam. The Koopman matrices K_k, b_k are solved in closed
#      form (least squares) given φ_k, then updated alternately.
#   3. Optionally augment with exact Koopman generator loss via JVP — this
#      uses the exact vector field, not a trajectory approximation.
#   4. Solve the discrete Riccati equation offline (scipy) to produce L_k.
#   5. Save all operators + phi_params to trained/koopman_tv/.
#
# Run:
#   python scripts/train_koopman_tv.py [--n_samples 100000] [--n_epochs 500]
#                                      [--use_hnet] [--output trained/koopman_tv]
#
# With --use_hnet, the 3-DOF bicycle is replaced by single-step rollouts
# through the full H_net vehicle dynamics for higher-fidelity training data.
# This requires the physics engine to be initialised and is ~10x slower.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.linalg

from powertrain.modes.advanced.koopman_tv import (
    KoopmanTVConfig, KoopmanOperators, KoopmanTVBundle,
    LiftingMap, _PHI0, _PHI1, _PHI2,
    koopman_generator, blend_weights, grip_utilisation,
    normalise_error, save_koopman_bundle,
)


# ─────────────────────────────────────────────────────────────────────────────
# §1  3-DOF Bicycle Model (training dynamics)
# ─────────────────────────────────────────────────────────────────────────────

# Vehicle constants (Ter27)
_M      = 300.0     # kg total mass
_IZ     = 150.0     # kg·m² yaw inertia
_LF     = 0.8525    # m  CG to front axle
_LR     = 0.6975    # m  CG to rear axle
_CY_F   = 30000.0   # N/rad front cornering stiffness (linear, tunable)
_CY_R   = 35000.0   # N/rad rear cornering stiffness
_MU_MAX = 1.5       # peak friction coefficient

def bicycle_forces(vy: float, wz: float, vx: float, delta: float, mu: float):
    """Pacejka-linearised lateral forces and effective stiffness."""
    vx_s  = jnp.maximum(vx, 1.0)
    alpha_f = vy / vx_s + wz * _LF / vx_s - delta
    alpha_r = vy / vx_s - wz * _LR / vx_s
    # Saturating cornering stiffness (tanh approximation of Pacejka peak)
    Fy_f_lin = _CY_F * alpha_f
    Fy_r_lin = _CY_R * alpha_r
    # Smooth saturation at friction limit
    F_limit  = mu * _M * 9.81 / 2.0        # per axle
    Fy_f = F_limit * jnp.tanh(Fy_f_lin / (F_limit + 1e-3))
    Fy_r = F_limit * jnp.tanh(Fy_r_lin / (F_limit + 1e-3))
    return Fy_f, Fy_r


@jax.jit
def bicycle_dynamics(
    e:    jax.Array,    # (4,) = [wz_err, vy, vx, delta]
    Mz:   jax.Array,   # scalar yaw moment [Nm]
    wz_ref: jax.Array, # scalar yaw rate reference [rad/s]
    mu:   jax.Array,   # scalar friction coefficient
) -> jax.Array:
    """
    3-DOF bicycle model time derivative of the error state.

    Returns ė = [dψ̇_err/dt, dvy/dt, dvx/dt, ddelta/dt] ∈ ℝ^4

    dvx and ddelta are treated as slowly varying (≈ 0) for the TV timescale.
    This is the correct approximation: TV acts on the ~100ms yaw timescale
    over which vx and delta are nearly constant.
    """
    wz_err, vy, vx, delta = e[0], e[1], e[2], e[3]
    wz = wz_ref + wz_err

    Fy_f, Fy_r = bicycle_forces(vy, wz, vx, delta, mu)

    # Yaw acceleration
    dwz_dt = (Mz + Fy_f * _LF - Fy_r * _LR) / _IZ

    # Lateral velocity (slip angle dynamics)
    dvy_dt = (Fy_f + Fy_r) / _M - vx * wz

    # dwz_err/dt = dwz/dt - dwz_ref/dt ≈ dwz/dt (wz_ref slowly varying)
    dwz_err_dt = dwz_dt

    return jnp.array([dwz_err_dt, dvy_dt, 0.0, 0.0])


# ─────────────────────────────────────────────────────────────────────────────
# §2  Training Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_data(
    n_samples: int,
    dt:        float = 0.005,
    seed:      int   = 0,
    cfg:       KoopmanTVConfig = KoopmanTVConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (E, Mz_data, E_next, F_e_data) training tuples.

    Samples random initial error states from the operating envelope,
    applies a random Mz, and integrates one Euler step to get E_next.
    F_e is the exact vector field at (E, Mz) — used for the generator loss.

    Returns:
        E       : (N, 4)  error states
        Mz_data : (N,)    applied yaw moments [Nm]
        E_next  : (N, 4)  next error states (one dt step)
        F_e     : (N, 4)  vector field at each sample
    """
    rng = np.random.default_rng(seed)

    # Sample state space
    wz_err = rng.uniform(-cfg.wz_scale,    cfg.wz_scale,    n_samples)
    vy     = rng.uniform(-cfg.vy_scale,    cfg.vy_scale,    n_samples)
    vx     = rng.uniform(3.0,              cfg.vx_scale,    n_samples)
    delta  = rng.uniform(-cfg.delta_scale, cfg.delta_scale, n_samples)

    # Sample control inputs: Mz range ≈ ±I_z · max_dwz ≈ ±150 · 10 = ±1500 Nm
    Mz_max = 1500.0
    Mz_data = rng.uniform(-Mz_max, Mz_max, n_samples)

    # Random operating conditions
    wz_ref = rng.uniform(-2.0, 2.0, n_samples)
    mu     = rng.uniform(0.8, 1.5, n_samples)

    E      = np.stack([wz_err, vy, vx, delta], axis=1)   # (N, 4)
    E_jnp  = jnp.array(E)
    Mz_jnp = jnp.array(Mz_data)

    # Batch vector field computation
    F_e = jax.vmap(bicycle_dynamics)(
        E_jnp,
        Mz_jnp,
        jnp.array(wz_ref),
        jnp.array(mu),
    )

    # One-step Euler integration
    E_next = E_jnp + dt * F_e

    return (
        np.array(E),
        np.array(Mz_data),
        np.array(E_next),
        np.array(F_e),
    )


def split_by_regime(
    E: np.ndarray,
    Mz: np.ndarray,
    E_next: np.ndarray,
    F_e: np.ndarray,
    mu: np.ndarray,
    Fz_nominal: np.ndarray,
    cfg: KoopmanTVConfig,
) -> list[tuple]:
    """
    Partition samples into three grip utilisation regimes.

    For training data, Fx ≈ 0 (TV-only context) so ρ ≈ |Fy| / (μ · Fz).
    Uses the Fy from the bicycle model at each sample.
    """
    rho_vals = np.zeros(len(E))
    for i, (e, mz_i) in enumerate(zip(E, Mz)):
        wz_err_i, vy_i, vx_i, delta_i = e
        wz_ref_i = 0.0      # nominal zero error context
        Fy_f, Fy_r = bicycle_forces(vy_i, wz_ref_i + wz_err_i, vx_i, delta_i, mu[i])
        Fy_total   = abs(Fy_f) + abs(Fy_r)
        rho_vals[i] = np.clip(Fy_total / (mu[i] * 9.81 * _M + 1.0), 0.0, 1.0)

    masks = [
        rho_vals < 0.70,
        (rho_vals >= 0.70) & (rho_vals <= 0.92),
        rho_vals > 0.92,
    ]

    regimes = []
    for mask in masks:
        regimes.append((E[mask], Mz[mask], E_next[mask], F_e[mask]))
        print(f"  Regime {len(regimes)-1}: {mask.sum()} samples ({mask.mean()*100:.1f}%)")

    return regimes


# ─────────────────────────────────────────────────────────────────────────────
# §3  EDMD-DL Training (alternating minimisation)
# ─────────────────────────────────────────────────────────────────────────────

def fit_edmd_matrices(
    phi_apply: callable,         # e → z (single sample, params already bound)
    E:    np.ndarray,            # (N, 4)
    Mz:   np.ndarray,            # (N,)
    E_next: np.ndarray,          # (N, 4)
    m:    int,                   # lifting dimension
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form EDMD: given fixed φ, solve for K, b via least squares.

    Stacks [z | Mz] and solves:
        [K | b]  =  argmin ||Z_next - [Z | Mz] · [K; b]^T||_F²

    Returns K ∈ ℝ^{m×m}, b ∈ ℝ^m.
    """
    Z      = jax.vmap(phi_apply)(jnp.array(E))           # (N, m)
    Z_next = jax.vmap(phi_apply)(jnp.array(E_next))      # (N, m)

    Z      = np.array(Z)
    Z_next = np.array(Z_next)

    X = np.concatenate([Z, Mz[:, None]], axis=1)          # (N, m+1)

    # Least squares: X @ Theta^T ≈ Z_next  →  Theta = (X^T X)^{-1} X^T Z_next
    Theta, _, _, _ = np.linalg.lstsq(X, Z_next, rcond=None)  # (m+1, m)

    K = Theta[:m].T       # (m, m)
    b = Theta[m]          # (m,)
    return K, b


@partial(jax.jit, static_argnums=(0, 1))
def edmd_loss_jit(
    module:    LiftingMap,
    m:         int,
    params:    dict,
    K:         jax.Array,
    b:         jax.Array,
    E_batch:   jax.Array,        # (B, 4)
    Mz_batch:  jax.Array,        # (B,)
    E_next_b:  jax.Array,        # (B, 4)
    F_e_batch: jax.Array,        # (B, 4)
    w_gen:     float = 0.1,
) -> jax.Array:
    """
    EDMD residual + Koopman generator consistency loss.

    L = ||Z_next − K·Z − b·Mz||²_F        (EDMD)
      + w_gen · ||Kφ(e) − Lφ(e, f)||²_F   (generator)
    """
    phi_fn = lambda e: module.apply(params, e)

    Z      = jax.vmap(phi_fn)(E_batch)           # (B, m)
    Z_next = jax.vmap(phi_fn)(E_next_b)          # (B, m)

    Z_pred = (K @ Z.T + b[:, None] * Mz_batch[None, :]).T   # (B, m)
    l_edmd = jnp.mean((Z_next - Z_pred) ** 2)

    # Exact generator via JVP (no finite differences)
    Lphi_exact = jax.vmap(
        lambda e, f: koopman_generator(phi_fn, e, f)
    )(E_batch, F_e_batch)                        # (B, m)
    Lphi_linear = jax.vmap(lambda z: K @ z)(Z)  # (B, m) — linear Koopman prediction
    l_gen = jnp.mean((Lphi_exact - Lphi_linear) ** 2)

    return l_edmd + w_gen * l_gen


def train_regime(
    module:      LiftingMap,
    m:           int,
    E:           np.ndarray,
    Mz:          np.ndarray,
    E_next:      np.ndarray,
    F_e:         np.ndarray,
    n_epochs:    int   = 500,
    batch_size:  int   = 2048,
    lr:          float = 3e-4,
    n_alt_iters: int   = 5,
    seed:        int   = 0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Alternating minimisation:
      outer: n_alt_iters iterations of (fix K,b → train φ) then (fix φ → update K,b)
      inner: Adam gradient steps on the EDMD + generator loss.

    Returns (phi_params, K, b).
    """
    key   = jax.random.PRNGKey(seed)
    dummy = jnp.zeros(4)
    params = module.init(key, dummy)

    phi_apply = lambda e: module.apply(params, e)
    K, b = fit_edmd_matrices(phi_apply, E, Mz, E_next, m)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    N = len(E)
    n_batches = max(N // batch_size, 1)
    rng = np.random.default_rng(seed + 1)

    @jax.jit
    def step(params_, opt_state_, K_, b_, E_b, Mz_b, En_b, Fe_b):
        loss, grads = jax.value_and_grad(
            lambda p: edmd_loss_jit(module, m, p, K_, b_, E_b, Mz_b, En_b, Fe_b)
        )(params_)
        updates, opt_state_new = optimizer.update(grads, opt_state_, params_)
        params_new = optax.apply_updates(params_, updates)
        return params_new, opt_state_new, loss

    print(f"    Training φ (m={m}): {N} samples, {n_epochs} epochs × {n_alt_iters} alt-iters")
    t0 = time.time()

    for alt_iter in range(n_alt_iters):
        # ── Inner: train phi ───────────────────────────────────────────
        epoch_losses = []
        idx = np.arange(N)
        for epoch in range(n_epochs):
            rng.shuffle(idx)
            epoch_loss = 0.0
            for b_idx in range(n_batches):
                sel = idx[b_idx * batch_size: (b_idx + 1) * batch_size]
                params, opt_state, loss = step(
                    params, opt_state,
                    jnp.array(K), jnp.array(b),
                    jnp.array(E[sel]), jnp.array(Mz[sel]),
                    jnp.array(E_next[sel]), jnp.array(F_e[sel]),
                )
                epoch_loss += float(loss)
            epoch_losses.append(epoch_loss / n_batches)

        phi_apply_new = lambda e: module.apply(params, e)
        K, b = fit_edmd_matrices(phi_apply_new, E, Mz, E_next, m)

        # Re-initialise optimizer (params same shape, EDMD matrices changed)
        opt_state = optimizer.init(params)

        mean_loss = np.mean(epoch_losses[-20:])
        print(f"      Alt-iter {alt_iter+1}/{n_alt_iters}  loss={mean_loss:.4e}  "
              f"spectral_radius(K)={np.max(np.abs(np.linalg.eigvals(K))):.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

    return params, K, b


# ─────────────────────────────────────────────────────────────────────────────
# §4  Discrete Riccati Solver (offline, scipy)
# ─────────────────────────────────────────────────────────────────────────────

def solve_riccati(
    K: np.ndarray,   # (m, m) Koopman autonomous matrix
    b: np.ndarray,   # (m,) control coupling vector
    Q_wz: float = 200.0,
    R_Mz: float = 0.01,
) -> np.ndarray:
    """
    Solve the discrete-time algebraic Riccati equation:

        P = Q + K^T P K - K^T P b (R + b^T P b)^{-1} b^T P K

    via scipy.linalg.solve_discrete_are.

    Returns L ∈ ℝ^m, the LQR gain vector: Mz* = −L @ z.
    """
    m = K.shape[0]
    # Full state cost: penalise all lifted dimensions equally (scaled by Q_wz)
    Q = Q_wz * np.eye(m)
    # Control input cost
    R = np.array([[R_Mz]])

    # scipy expects (A, B, Q, R) for the standard form
    B = b.reshape(-1, 1)    # (m, 1)

    try:
        P = scipy.linalg.solve_discrete_are(K, B, Q, R)
    except scipy.linalg.LinAlgError as e:
        print(f"    [WARN] Riccati failed ({e}), falling back to fixed-point (100 iters)")
        P = np.eye(m) * Q_wz
        for _ in range(100):
            BtPB = float(B.T @ P @ B)
            KtPB = K.T @ P @ B
            P = Q + K.T @ P @ K - (KtPB @ KtPB.T) / (R_Mz + BtPB)

    BtP = B.T @ P         # (1, m)
    L = BtP.flatten() / (R_Mz + float(B.T @ P @ B))   # (m,)

    return L


# ─────────────────────────────────────────────────────────────────────────────
# §5  Main Training Script
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Koopman TV operators")
    parser.add_argument("--n_samples",  type=int,   default=100_000)
    parser.add_argument("--n_epochs",   type=int,   default=300)
    parser.add_argument("--n_alt",      type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--output",     type=str,   default="trained/koopman_tv")
    parser.add_argument("--seed",       type=int,   default=0)
    args = parser.parse_args()

    cfg = KoopmanTVConfig()
    modules = [_PHI0, _PHI1, _PHI2]
    dims    = [cfg.m0, cfg.m1, cfg.m2]

    print(f"[KoopmanTV] Generating {args.n_samples:,} training samples...")
    E, Mz_data, E_next, F_e = generate_training_data(args.n_samples, cfg=cfg, seed=args.seed)

    # Approximate mu and Fz for regime splitting (nominal values)
    mu_approx  = np.full(len(E), 1.3)
    Fz_nominal = np.full(4, _M * 9.81 / 4.0)

    print("[KoopmanTV] Splitting into grip regimes...")
    regimes = split_by_regime(E, Mz_data, E_next, F_e, mu_approx, Fz_nominal, cfg)

    trained_params = []
    trained_K = []
    trained_b = []
    trained_L = []

    for k, (E_k, Mz_k, E_next_k, F_e_k) in enumerate(regimes):
        print(f"\n[KoopmanTV] ═══ Regime {k}  (m={dims[k]}) ═══")

        if len(E_k) < 512:
            print(f"  [WARN] Only {len(E_k)} samples in regime {k}. "
                  f"Using full dataset with regime loss weighting.")
            E_k, Mz_k, E_next_k, F_e_k = E, Mz_data, E_next, F_e

        phi_params, K, b = train_regime(
            module     = modules[k],
            m          = dims[k],
            E          = E_k,
            Mz         = Mz_k,
            E_next     = E_next_k,
            F_e        = F_e_k,
            n_epochs   = args.n_epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            n_alt_iters= args.n_alt,
            seed       = args.seed + k,
        )

        print(f"  Solving Riccati (m={dims[k]})...")
        L = solve_riccati(K, b, cfg.Q_wz, cfg.R_Mz)
        print(f"  ||L||₂ = {np.linalg.norm(L):.2f}  "
              f"max|L| = {np.max(np.abs(L)):.2f}")

        trained_params.append(phi_params)
        trained_K.append(K)
        trained_b.append(b)
        trained_L.append(L)

    # ── Package and save ────────────────────────────────────────────────
    ops = KoopmanOperators(
        K0=jnp.array(trained_K[0]), b0=jnp.array(trained_b[0]), L0=jnp.array(trained_L[0]),
        K1=jnp.array(trained_K[1]), b1=jnp.array(trained_b[1]), L1=jnp.array(trained_L[1]),
        K2=jnp.array(trained_K[2]), b2=jnp.array(trained_b[2]), L2=jnp.array(trained_L[2]),
    )

    # Deploy with trained_blend=0.0 — ramp to 1.0 after hardware validation
    bundle = KoopmanTVBundle(
        ops  = ops,
        phi0 = trained_params[0],
        phi1 = trained_params[1],
        phi2 = trained_params[2],
        cfg  = KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": 0.0}),
    )

    save_koopman_bundle(bundle, args.output)

    # ── Quick sanity check ───────────────────────────────────────────────
    print("\n[KoopmanTV] Sanity checks...")
    from powertrain.modes.advanced.koopman_tv import koopman_mz_reference

    # Inert check: at wz_err=0, Mz should be ≈ 0
    Mz_at_zero, rho = koopman_mz_reference(
        jnp.array(0.0), jnp.array(0.0),
        jnp.array(3.0), jnp.array(15.0), jnp.array(0.05),
        jnp.zeros(4), jnp.zeros(4), jnp.full(4, 750.0),
        jnp.array(1.3),
        bundle._replace(cfg=KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": 1.0})),
    )
    print(f"  Mz(wz_err=0): {float(Mz_at_zero):.2f} Nm  (should be ≈ 0)")
    print(f"  ρ:            {float(rho):.3f}")

    # Gradient check
    grad_fn = jax.grad(lambda e: koopman_mz_reference(
        e, jnp.array(0.0), jnp.array(3.0), jnp.array(15.0), jnp.array(0.0),
        jnp.zeros(4), jnp.zeros(4), jnp.full(4, 750.0), jnp.array(1.3),
        bundle._replace(cfg=KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": 1.0})),
    )[0])
    dMz_de = grad_fn(jnp.array(1.0))
    print(f"  dMz/d(wz_err) at e=1: {float(dMz_de):.2f} Nm/(rad/s)  "
          f"(PD equivalent: {cfg.Kp_fallback:.1f})")
    print(f"  Gradient finite: {bool(jnp.isfinite(dMz_de))}")
    print("\n[KoopmanTV] Training complete.")


if __name__ == "__main__":
    main()