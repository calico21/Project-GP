# scripts/train_koopman_tv.py
# Project-GP — Offline Koopman TV Training (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
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
    koopman_generator, save_koopman_bundle,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global constants — declared ONCE
# ─────────────────────────────────────────────────────────────────────────────
MZ_NORM = 1500.0   # [Nm]

# ─────────────────────────────────────────────────────────────────────────────
# §1  3-DOF Bicycle Model
# ─────────────────────────────────────────────────────────────────────────────
_M    = 300.0;  _IZ = 150.0;  _LF = 0.8525;  _LR = 0.6975
_CY_F = 30000.0; _CY_R = 35000.0


def bicycle_forces(vy, wz, vx, delta, mu):
    vx_s    = jnp.maximum(vx, 1.0)
    F_limit = mu * _M * 9.81 / 2.0
    Fy_f    = F_limit * jnp.tanh(_CY_F * (vy/vx_s + wz*_LF/vx_s - delta) / (F_limit + 1e-3))
    Fy_r    = F_limit * jnp.tanh(_CY_R * (vy/vx_s - wz*_LR/vx_s)           / (F_limit + 1e-3))
    return Fy_f, Fy_r


@jax.jit
def bicycle_dynamics(e, Mz, wz_ref, mu):
    """Physical units in, physical units out."""
    wz_err, vy, vx, delta = e[0], e[1], e[2], e[3]
    Fy_f, Fy_r = bicycle_forces(vy, wz_ref + wz_err, vx, delta, mu)
    return jnp.array([
        (Mz + Fy_f*_LF - Fy_r*_LR) / _IZ,
        (Fy_f + Fy_r) / _M - vx*(wz_ref + wz_err),
        0.0, 0.0,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# §2  Training Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def _norm_E(E: np.ndarray, cfg: KoopmanTVConfig) -> np.ndarray:
    """Raw physical -> normalised in [-1,1]^4. Mirrors runtime normalise_error()."""
    return np.stack([
        np.clip(E[:,0], -cfg.wz_scale,    cfg.wz_scale)    / cfg.wz_scale,
        np.clip(E[:,1], -cfg.vy_scale,    cfg.vy_scale)    / cfg.vy_scale,
        np.clip(E[:,2],  0.0,              cfg.vx_scale)   / cfg.vx_scale,
        np.clip(E[:,3], -cfg.delta_scale,  cfg.delta_scale) / cfg.delta_scale,
    ], axis=1)


def generate_training_data(n_samples, dt=0.005, seed=0, cfg=KoopmanTVConfig()):
    """
    Returns (E_raw, E_norm, Mz_norm, E_next_norm, F_e_norm).
    Everything in normalised coords except E_raw (used for regime splitting).
    """
    rng     = np.random.default_rng(seed)
    wz_err  = rng.uniform(-cfg.wz_scale,    cfg.wz_scale,    n_samples)
    vy      = rng.uniform(-cfg.vy_scale,    cfg.vy_scale,    n_samples)
    vx      = rng.uniform( 3.0,             cfg.vx_scale,    n_samples)
    delta   = rng.uniform(-cfg.delta_scale, cfg.delta_scale, n_samples)
    Mz_phys = rng.uniform(-MZ_NORM,          MZ_NORM,         n_samples)
    wz_ref  = rng.uniform(-2.0,  2.0,  n_samples)
    mu      = rng.uniform( 0.8,  1.5,  n_samples)

    E_raw  = np.stack([wz_err, vy, vx, delta], axis=1)
    E_norm = _norm_E(E_raw, cfg)

    F_phys = np.array(jax.vmap(bicycle_dynamics)(
        jnp.array(E_raw), jnp.array(Mz_phys), jnp.array(wz_ref), jnp.array(mu),
    ))

    E_next_norm = _norm_E(np.array(jnp.array(E_raw) + dt * jnp.array(F_phys)), cfg)

    f_scale  = np.array([cfg.wz_scale, cfg.vy_scale, cfg.vx_scale, cfg.delta_scale])
    F_e_norm = F_phys / f_scale[np.newaxis, :]   # normalised for generator JVP

    return E_raw, E_norm, Mz_phys / MZ_NORM, E_next_norm, F_e_norm


def split_by_regime(E_raw, E_norm, Mz_norm, E_next_norm, F_e_norm, mu, cfg):
    rho_vals = np.zeros(len(E_raw))
    for i, e in enumerate(E_raw):
        Fy_f, Fy_r = bicycle_forces(e[1], e[0], e[2], e[3], mu[i])
        rho_vals[i] = np.clip(
            (abs(float(Fy_f)) + abs(float(Fy_r))) / (mu[i]*9.81*_M + 1.0), 0, 1
        )
    masks = [rho_vals < 0.70, (rho_vals >= 0.70) & (rho_vals <= 0.92), rho_vals > 0.92]
    regimes = []
    for mask in masks:
        regimes.append((E_norm[mask], Mz_norm[mask], E_next_norm[mask], F_e_norm[mask]))
        print(f"  Regime {len(regimes)-1}: {mask.sum()} samples ({mask.mean()*100:.1f}%)")
    return regimes


# ─────────────────────────────────────────────────────────────────────────────
# §3  EDMD-DL Training
# ─────────────────────────────────────────────────────────────────────────────

def fit_edmd_matrices(phi_apply, E_norm, Mz_norm, E_next_norm, m):
    Z      = np.array(jax.vmap(phi_apply)(jnp.array(E_norm)))
    Z_next = np.array(jax.vmap(phi_apply)(jnp.array(E_next_norm)))
    X      = np.concatenate([Z, Mz_norm[:, None]], axis=1)
    Theta, _, _, _ = np.linalg.lstsq(X, Z_next, rcond=None)
    return Theta[:m].T, Theta[m]   # K, b


@partial(jax.jit, static_argnums=(0, 1))
def edmd_loss_jit(module, m, params, K, b,
                  E_norm_b, Mz_norm_b, E_next_b, F_e_norm_b,
                  w_gen: float = 0.1, w_obs: float = 5.0):
    """
    EDMD residual  +  Koopman generator  +  observability regulariser.

    l_obs trains phi to linearly encode wz_err_norm in z-space by minimising
    the batch decoder MSE. This replaces the blind variance penalty, which
    caused phi to find high-variance but wz_err-unrelated representations.
    """
    phi_fn = lambda e: module.apply(params, e)

    Z      = jax.vmap(phi_fn)(E_norm_b)
    Z_next = jax.vmap(phi_fn)(E_next_b)

    Z_pred = (K @ Z.T + b[:, None] * Mz_norm_b[None, :]).T
    l_edmd = jnp.mean((Z_next - Z_pred) ** 2)

    Lphi_exact  = jax.vmap(lambda e, f: koopman_generator(phi_fn, e, f))(E_norm_b, F_e_norm_b)
    Lphi_linear = jax.vmap(lambda z: K @ z)(Z)
    l_gen = jnp.mean((Lphi_exact - Lphi_linear) ** 2)

    # Observability: phi must linearly encode wz_err_norm (first component of e_norm)
    wz_n  = E_norm_b[:, 0]
    ZtZ   = Z.T @ Z + 1e-4 * jnp.eye(m)
    c_hat = jnp.linalg.solve(ZtZ, Z.T @ wz_n)
    l_obs = jnp.mean((Z @ c_hat - wz_n) ** 2)

    return l_edmd + w_gen * l_gen + w_obs * l_obs


def train_regime(module, m, E_norm, Mz_norm, E_next_norm, F_e_norm,
                 n_epochs=500, batch_size=2048, lr=3e-4, n_alt_iters=5, seed=0):
    key    = jax.random.PRNGKey(seed)
    params = module.init(key, jnp.zeros(4))

    phi_apply = lambda e: module.apply(params, e)
    K, b = fit_edmd_matrices(phi_apply, E_norm, Mz_norm, E_next_norm, m)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    N         = len(E_norm)
    n_batches = max(N // batch_size, 1)
    rng       = np.random.default_rng(seed + 1)

    @jax.jit
    def step(params_, opt_state_, K_, b_, E_b, Mz_b, En_b, Fe_b):
        loss, grads = jax.value_and_grad(
            lambda p: edmd_loss_jit(module, m, p, K_, b_, E_b, Mz_b, En_b, Fe_b)
        )(params_)
        updates, opt_state_new = optimizer.update(grads, opt_state_, params_)
        return optax.apply_updates(params_, updates), opt_state_new, loss

    print(f"    Training φ (m={m}): {N} samples, {n_epochs} epochs x {n_alt_iters} alt-iters")
    t0 = time.time()
    for alt_iter in range(n_alt_iters):
        losses = []
        idx = np.arange(N)
        for _ in range(n_epochs):
            rng.shuffle(idx)
            epoch_loss = 0.0
            for bi in range(n_batches):
                sel = idx[bi*batch_size:(bi+1)*batch_size]
                params, opt_state, loss = step(
                    params, opt_state,
                    jnp.array(K), jnp.array(b),
                    jnp.array(E_norm[sel]),      jnp.array(Mz_norm[sel]),
                    jnp.array(E_next_norm[sel]), jnp.array(F_e_norm[sel]),
                )
                epoch_loss += float(loss)
            losses.append(epoch_loss / n_batches)
        phi_new = lambda e: module.apply(params, e)
        K, b    = fit_edmd_matrices(phi_new, E_norm, Mz_norm, E_next_norm, m)
        opt_state = optimizer.init(params)

        Z_probe    = np.array(jax.vmap(phi_new)(jnp.array(E_norm[:2048])))
        z_var_mean = float(np.mean(np.var(Z_probe, axis=0)))
        print(f"      Alt-iter {alt_iter+1}/{n_alt_iters}  "
              f"loss={np.mean(losses[-20:]):.4e}  "
              f"ρ(K)={np.max(np.abs(np.linalg.eigvals(K))):.4f}  "
              f"z_var={z_var_mean:.2e}  "
              f"elapsed={time.time()-t0:.0f}s")
    return params, K, b


# ─────────────────────────────────────────────────────────────────────────────
# §3.5  Spectral Stabilisation + Gain Computation
# ─────────────────────────────────────────────────────────────────────────────

def stabilise_koopman(K, target_radius=0.98):
    T, Z = scipy.linalg.schur(K, output='complex')
    mags  = np.abs(np.diag(T))
    scale = np.where(mags > target_radius, target_radius / mags, 1.0)
    return (Z @ (T * scale[np.newaxis, :] * scale[:, np.newaxis]) @ Z.conj().T).real


def compute_koopman_gain(K, b, phi_apply, m, cfg, Kp_scale=1.0, target_radius=0.98):
    rho = np.max(np.abs(np.linalg.eigvals(K)))
    K_s = stabilise_koopman(K, target_radius)
    if rho > target_radius:
        print(f"    [stabilise] ρ: {rho:.6f} -> "
              f"{np.max(np.abs(np.linalg.eigvals(K_s))):.6f}")

    rng     = np.random.default_rng(42)
    n_probe = 4000
    E_probe = np.stack([
        rng.uniform(-1.0, 1.0, n_probe),
        rng.uniform(-1.0, 1.0, n_probe),
        rng.uniform( 0.0, 1.0, n_probe),
        rng.uniform(-1.0, 1.0, n_probe),
    ], axis=1)
    Z_probe   = np.array(jax.vmap(phi_apply)(jnp.array(E_probe)))
    wz_target = E_probe[:, 0]

    z_std_vec = np.std(Z_probe, axis=0)
    print(f"    [phi stats] z_std: min={z_std_vec.min():.3e}  "
          f"max={z_std_vec.max():.3e}  mean={z_std_vec.mean():.3e}")

    U, s, Vt = np.linalg.svd(Z_probe, full_matrices=False)
    threshold = 1e-3 * s[0]
    s_inv     = np.where(s > threshold, 1.0/s, 0.0)
    c         = Vt.T @ (s_inv * (U.T @ wz_target))

    # Sign correction: ensure c @ z is positively correlated with wz_err
    alignment = float(np.dot(wz_target, Z_probe @ c))
    if alignment < 0:
        c = -c
        print(f"    [decoder] Sign flipped (alignment was {alignment:.3f})")

    effective_rank = int(np.sum(s > threshold))
    wz_pred = Z_probe @ c
    r2 = 1.0 - np.var(wz_target - wz_pred) / (np.var(wz_target) + 1e-12)
    print(f"    [decoder] R2={r2:.4f}  ||c||={np.linalg.norm(c):.4f}  "
          f"rank={effective_rank}/{m}")
    if r2 < 0.5:
        print(f"    [decoder] WARNING: R2={r2:.3f} < 0.5")

    Kp_eff = cfg.Kp_fallback * Kp_scale
    L      = - Kp_eff * cfg.wz_scale * c

    for wz_n_test, label in [(0.25, "wz=0.5 rad/s"), (0.50, "wz=1.0 rad/s")]:
        z_t   = np.array(phi_apply(jnp.array([wz_n_test, 0.0, 0.5, 0.0])))
        Mz_t  = float(np.dot(L, z_t))
        Mz_pd = Kp_eff * wz_n_test * cfg.wz_scale
        print(f"    [{label}] Mz={Mz_t:.1f} Nm  PD={Mz_pd:.1f} Nm  "
              f"ratio={Mz_t/(Mz_pd+1e-9):.2f}x")

    return L


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int,   default=100_000)
    parser.add_argument("--n_epochs",   type=int,   default=300)
    parser.add_argument("--n_alt",      type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--output",     type=str,   default="trained/koopman_tv")
    parser.add_argument("--seed",       type=int,   default=0)
    args = parser.parse_args()

    cfg     = KoopmanTVConfig()
    modules = [_PHI0, _PHI1, _PHI2]
    dims    = [cfg.m0, cfg.m1, cfg.m2]

    print(f"[KoopmanTV] Generating {args.n_samples:,} training samples...")
    E_raw, E_norm, Mz_norm, E_next_norm, F_e_norm = generate_training_data(
        args.n_samples, cfg=cfg, seed=args.seed,
    )
    mu_approx = np.full(len(E_raw), 1.3)

    print("[KoopmanTV] Splitting into grip regimes...")
    regimes = split_by_regime(
        E_raw, E_norm, Mz_norm, E_next_norm, F_e_norm, mu_approx, cfg,
    )

    trained_params, trained_K, trained_b, trained_L = [], [], [], []

    for k, (E_k, Mz_k, E_next_k, F_e_k) in enumerate(regimes):
        print(f"\n[KoopmanTV] === Regime {k}  (m={dims[k]}) ===")
        if len(E_k) < 512:
            print(f"  [WARN] Only {len(E_k)} samples -- using full dataset.")
            E_k, Mz_k, E_next_k, F_e_k = E_norm, Mz_norm, E_next_norm, F_e_norm

        phi_params, K, b = train_regime(
            modules[k], dims[k], E_k, Mz_k, E_next_k, F_e_k,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, n_alt_iters=args.n_alt, seed=args.seed + k,
        )

        # Default-arg binding prevents loop-variable capture
        phi_k = lambda e, _p=phi_params, _k=k: modules[_k].apply(_p, e)

        print(f"  Computing gain (m={dims[k]})...")
        L = compute_koopman_gain(K, b, phi_k, dims[k], cfg)

        trained_params.append(phi_params)
        trained_K.append(K);  trained_b.append(b);  trained_L.append(L)

    ops = KoopmanOperators(
        K0=jnp.array(trained_K[0]), b0=jnp.array(trained_b[0]), L0=jnp.array(trained_L[0]),
        K1=jnp.array(trained_K[1]), b1=jnp.array(trained_b[1]), L1=jnp.array(trained_L[1]),
        K2=jnp.array(trained_K[2]), b2=jnp.array(trained_b[2]), L2=jnp.array(trained_L[2]),
    )
    bundle = KoopmanTVBundle(
        ops=ops,
        phi0=trained_params[0], phi1=trained_params[1], phi2=trained_params[2],
        cfg=KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": 0.0}),
    )
    save_koopman_bundle(bundle, args.output)

    # Sanity checks on Koopman path (blend=1.0)
    print("\n[KoopmanTV] Sanity checks...")
    from powertrain.modes.advanced.koopman_tv import koopman_mz_reference

    bundle_test = bundle._replace(
        cfg=KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": 1.0})
    )

    Mz_at_zero, rho = koopman_mz_reference(
        jnp.array(0.0), jnp.array(0.0),
        jnp.array(3.0), jnp.array(15.0), jnp.array(0.05),
        jnp.zeros(4), jnp.zeros(4), jnp.full(4, 750.0),
        jnp.array(1.3), bundle_test,
    )
    print(f"  Mz(wz_err=0): {float(Mz_at_zero):.2f} Nm  (should be ~0)")

    Mz_at_one, _ = koopman_mz_reference(
        jnp.array(1.0), jnp.array(0.0),
        jnp.array(15.0), jnp.array(15.0), jnp.array(0.05),
        jnp.zeros(4), jnp.zeros(4), jnp.full(4, 750.0),
        jnp.array(1.3), bundle_test,
    )
    print(f"  Mz(wz_err=1): {float(Mz_at_one):.2f} Nm  "
          f"(PD equivalent: {cfg.Kp_fallback:.1f} Nm)")
    print(f"  Gain ratio vs PD: {abs(float(Mz_at_one))/cfg.Kp_fallback:.2f}x")

    grad_fn = jax.grad(lambda e: koopman_mz_reference(
        e, jnp.array(0.0), jnp.array(15.0), jnp.array(15.0), jnp.array(0.0),
        jnp.zeros(4), jnp.zeros(4), jnp.full(4, 750.0), jnp.array(1.3), bundle_test,
    )[0])
    dMz_de = grad_fn(jnp.array(1.0))
    print(f"  dMz/d(wz_err): {float(dMz_de):.2f} Nm/(rad/s)  "
          f"(finite: {bool(jnp.isfinite(dMz_de))})")
    print("\n[KoopmanTV] Training complete.")


if __name__ == "__main__":
    main()