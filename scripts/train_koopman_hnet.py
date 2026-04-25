# scripts/train_koopman_hnet.py
# Project-GP — Koopman TV Retraining on H_net 46-DOF Trajectories
# ═══════════════════════════════════════════════════════════════════════════════
#
# Fully self-contained — no dependency on train_koopman_tv.py.
# All EDMD-DL helpers are inlined here verbatim.
#
# Usage:
#   python scripts/train_koopman_hnet.py --quick                        # ~5 min
#   python scripts/train_koopman_hnet.py --n_samples 500000 --n_epochs 500
#   python scripts/train_koopman_hnet.py --setup trained/pareto_best_setup.npy
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.linalg

# ── Project root ─────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter27 import vehicle_params_ter27 as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from simulator.sim_config import S
from powertrain.modes.advanced.koopman_tv import (
    KoopmanTVConfig, KoopmanOperators, KoopmanTVBundle,
    _PHI0, _PHI1, _PHI2,
    koopman_generator, save_koopman_bundle,
)

# ─────────────────────────────────────────────────────────────────────────────
# §1  Global constants
# ─────────────────────────────────────────────────────────────────────────────

MZ_NORM = 1500.0   # [Nm] — normalisation constant for Mz

_LF   = VP_DICT.get('lf',          0.8525)
_LR   = VP_DICT.get('lr',          0.6975)
_L    = _LF + _LR
_IZ   = VP_DICT.get('Iz',          150.0)
_K_US = 0.006

_DEFAULT_SETUP_28 = jnp.array([
    42000., 40000.,   # k_f, k_r  [N/m]
     3200.,  3000.,   # c_f, c_r  [Ns/m]
     8000.,  7000.,   # arb_f, arb_r [Nm/rad]
    -0.04,  -0.03,    # camber_f, camber_r [rad]
     0.002, -0.002,   # toe_f, toe_r [rad]
     0.030,  0.032,   # ride_height_f, ride_height_r [m]
     0.5,    0.5,     # aero_bias, ballast_frac
     0., 0., 0., 0., 0., 0., 0.,   # 14 extended params (zeros = nominal)
     0., 0., 0., 0., 0., 0., 0.,
], dtype=jnp.float32)

_WARM_TIRE_TEMP = 82.0   # °C


# ─────────────────────────────────────────────────────────────────────────────
# §2  EDMD-DL helpers (inlined from train_koopman_tv.py verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_E(E: np.ndarray, cfg: KoopmanTVConfig) -> np.ndarray:
    return np.stack([
        np.clip(E[:, 0], -cfg.wz_scale,    cfg.wz_scale)    / cfg.wz_scale,
        np.clip(E[:, 1], -cfg.vy_scale,    cfg.vy_scale)    / cfg.vy_scale,
        np.clip(E[:, 2],  0.0,             cfg.vx_scale)    / cfg.vx_scale,
        np.clip(E[:, 3], -cfg.delta_scale, cfg.delta_scale) / cfg.delta_scale,
    ], axis=1)


def fit_edmd_matrices(phi_apply, E_norm, Mz_norm, E_next_norm, m):
    # 1. Generate features
    Z      = np.array(jax.vmap(phi_apply)(jnp.array(E_norm)))
    Z_next = np.array(jax.vmap(phi_apply)(jnp.array(E_next_norm)))
    X      = np.concatenate([Z, Mz_norm[:, None]], axis=1)
    
    # 2. Crucial: Purge any NaN/Inf data points that might have 
    # slipped through the XLA compiled physics engine.
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(Z_next).all(axis=1)
    X = X[valid_mask]
    Z_next = Z_next[valid_mask]

    # 3. Dynamic Regularization (scales with data size)
    n_samples, n_features = X.shape
    lambda_reg = 1e-4 * n_samples  # Stronger penalty for larger matrices
    
    X_T = X.T
    A = X_T @ X + lambda_reg * np.eye(n_features)
    B = X_T @ Z_next
    
    # 4. Use a robust pseudo-inverse fallback instead of strict Cholesky
    try:
        # Try the fast, symmetric solver first
        import scipy.linalg
        Theta = scipy.linalg.solve(A, B, assume_a='pos')
    except (np.linalg.LinAlgError, ValueError):
        # If the matrix is still technically singular (or nearly), 
        # fall back to the SVD-based pinv which NEVER crashes.
        print("  [WARN] Matrix singular in EDMD, falling back to regularized PINV.")
        Theta = np.linalg.pinv(A) @ B
        
    return Theta[:m].T, Theta[m]   # K, b


@partial(jax.jit, static_argnums=(0, 1))
def edmd_loss_jit(module, m, params, K, b,
                  E_norm_b, Mz_norm_b, E_next_b, F_e_norm_b,
                  w_gen: float = 0.1, w_obs: float = 5.0):
    phi_fn = lambda e: module.apply(params, e)
    Z      = jax.vmap(phi_fn)(E_norm_b)
    Z_next = jax.vmap(phi_fn)(E_next_b)
    Z_pred = (K @ Z.T + b[:, None] * Mz_norm_b[None, :]).T
    l_edmd = jnp.mean((Z_next - Z_pred) ** 2)

    Lphi_exact  = jax.vmap(
        lambda e, f: koopman_generator(phi_fn, e, f)
    )(E_norm_b, F_e_norm_b)
    Lphi_linear = jax.vmap(lambda z: K @ z)(Z)
    l_gen = jnp.mean((Lphi_exact - Lphi_linear) ** 2)

    wz_n  = E_norm_b[:, 0]
    ZtZ   = Z.T @ Z + 1e-4 * jnp.eye(m)
    c_hat = jnp.linalg.solve(ZtZ, Z.T @ wz_n)
    l_obs = jnp.mean((Z @ c_hat - wz_n) ** 2)

    return l_edmd + w_gen * l_gen + w_obs * l_obs


def stabilise_koopman(K, target_radius=0.98):
    T, Z = scipy.linalg.schur(K, output='complex')
    mags  = np.abs(np.diag(T))
    scale = np.where(mags > target_radius, target_radius / mags, 1.0)
    return (Z @ (T * scale[np.newaxis, :] * scale[:, np.newaxis]) @ Z.conj().T).real


def compute_koopman_gain(K, b, phi_apply, m, cfg,
                         Q_scale: float = 10.0, R_scale: float = 0.01):
    K_stab = stabilise_koopman(K)
    Q = Q_scale * np.eye(m)
    R = np.array([[R_scale]])
    B = b[:, np.newaxis]
    try:
        P = scipy.linalg.solve_discrete_are(K_stab, B, Q, R)
        L = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ K_stab).flatten()
    except Exception:
        L = Q_scale * np.ones(m)

    wz_n_test = 1.0
    Kp_eff = cfg.Kp_fallback
    for label, wz_n_test in [("low", 0.3), ("mid", 0.7), ("hi", 1.0)]:
        e_test = np.array([wz_n_test, 0.5, 0.5, 0.0])
        z_t    = np.array(phi_apply(jnp.array(e_test)))
        Mz_t   = float(np.dot(L, z_t))
        Mz_pd  = Kp_eff * wz_n_test * cfg.wz_scale
        print(f"    [{label}] Mz={Mz_t:.1f} Nm  PD={Mz_pd:.1f} Nm  "
              f"ratio={Mz_t / (Mz_pd + 1e-9):.2f}×")
    return L


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
        updates, new_opt = optimizer.update(grads, opt_state_, params_)
        return optax.apply_updates(params_, updates), new_opt, loss

    print(f"    Training φ (m={m}): {N:,} samples, "
          f"{n_epochs} epochs × {n_alt_iters} alt-iters")
    t0 = time.time()

    for alt_iter in range(n_alt_iters):
        losses = []
        idx    = np.arange(N)
        for _ in range(n_epochs):
            rng.shuffle(idx)
            ep_loss = 0.0
            for bi in range(n_batches):
                sel = idx[bi * batch_size:(bi + 1) * batch_size]
                params, opt_state, loss = step(
                    params, opt_state,
                    jnp.array(K), jnp.array(b),
                    jnp.array(E_norm[sel]),       jnp.array(Mz_norm[sel]),
                    jnp.array(E_next_norm[sel]),  jnp.array(F_e_norm[sel]),
                )
                ep_loss += float(loss)
            losses.append(ep_loss / n_batches)

        phi_new   = lambda e: module.apply(params, e)
        K, b      = fit_edmd_matrices(phi_new, E_norm, Mz_norm, E_next_norm, m)
        opt_state = optimizer.init(params)

        Z_probe   = np.array(jax.vmap(phi_new)(jnp.array(E_norm[:2048])))
        z_var     = float(np.mean(np.var(Z_probe, axis=0)))
        rho_K     = float(np.max(np.abs(np.linalg.eigvals(K))))
        print(f"      alt {alt_iter+1}/{n_alt_iters}  "
              f"loss={np.mean(losses[-20:]):.4e}  ρ(K)={rho_K:.4f}  "
              f"z_var={z_var:.2e}  t={time.time()-t0:.0f}s")

    return params, K, b


# ─────────────────────────────────────────────────────────────────────────────
# §3  H_net state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_initial_state(vx: float, vy: float, wz: float,
                         t_tire: float = _WARM_TIRE_TEMP) -> jax.Array:
    r_w   = VP_DICT.get('wheel_radius', 0.2032)
    
    # Use the 108-DOF factory method instead of zeros(46)
    state = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx)
    
    state = state.at[2].set(-0.015)
    state = state.at[S.VY].set(vy)
    state = state.at[S.WYAW].set(wz)
    
    omega0 = float(np.clip(vx / (r_w + 1e-6), 0.0, 200.0))
    state = state.at[S.WSPIN_FL].set(omega0)
    state = state.at[S.WSPIN_FR].set(omega0)
    state = state.at[S.WSPIN_RL].set(omega0)
    state = state.at[S.WSPIN_RR].set(omega0)
    
    # 108-DOF model has 28 thermal states (4 wheels * 7 nodes), not 10
    state = state.at[28:56].set(jnp.full(28, t_tire))
    return state


@jax.jit
def _extract_error_state(state: jax.Array, delta: jax.Array) -> jax.Array:
    vx     = state[S.VX]
    vy     = state[S.VY]
    wz     = state[S.WYAW]
    wz_ref = vx * delta / (_L * (1.0 + _K_US * vx ** 2) + 1e-6)
    return jnp.array([wz - wz_ref, vy, vx, delta])


@jax.jit
def _grip_util_hnet(state: jax.Array, state_next: jax.Array,
                    dt: float, mu: float) -> jax.Array:
    dvy_dt = (state_next[S.VY] - state[S.VY]) / (dt + 1e-6)
    ay_c   = dvy_dt + state[S.VX] * state[S.WYAW]
    return jnp.clip(jnp.abs(ay_c) / (mu * 9.81 + 1e-3), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Vmappable transition generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_transition_fn(vehicle: DifferentiableMultiBodyVehicle,
                        setup: jax.Array, dt: float):
    @jax.jit
    def _single(state, delta, Mz, mu):
        controls   = jnp.stack([delta, jnp.zeros_like(delta)])
        state_next = vehicle.simulate_step(state, controls, setup, dt)
        E_t        = _extract_error_state(state,      delta)
        E_next_nat = _extract_error_state(state_next, delta)
        # Exact Mz injection: I_z·ψ̈ += Mz  →  ψ̇_next += Mz·dt/I_z
        E_next     = E_next_nat.at[0].add(Mz * dt / _IZ)
        F_e        = (E_next - E_t) / (dt + 1e-6)
        rho        = _grip_util_hnet(state, state_next, dt, mu)  # <--- FIX: removed float()
        return E_t, Mz / MZ_NORM, E_next, F_e, rho

    return jax.vmap(_single)


# ─────────────────────────────────────────────────────────────────────────────
# §5  H_net data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_hnet_training_data(n_samples, vehicle, setup,
                                dt=0.005, seed=0, vmap_batch=512,
                                cfg=KoopmanTVConfig()):
    rng       = np.random.default_rng(seed)
    n_uniform = int(n_samples * 0.70)
    n_biased  = n_samples - n_uniform
    print(f"  Mixture: {n_uniform:,} uniform + {n_biased:,} saturation-biased")

    transition_fn = _make_transition_fn(vehicle, setup, dt)

    all_E, all_Mz, all_En, all_Fe, all_rho = [], [], [], [], []

    def _run_batch(vx_b, vy_b, wz_b, delta_b, Mz_b, mu_b):
        states = jnp.stack([
            _build_initial_state(float(vx_b[i]), float(vy_b[i]), float(wz_b[i]))
            for i in range(len(vx_b))
        ])
        E, Mzn, En, Fe, rho = transition_fn(
            states,
            jnp.array(delta_b, dtype=jnp.float32),
            jnp.array(Mz_b,    dtype=jnp.float32),
            jnp.array(mu_b,    dtype=jnp.float32),
        )
        return map(np.array, (E, Mzn, En, Fe, rho))

    # ── Uniform ───────────────────────────────────────────────────────────
    print("  Uniform sampling...")
    for start in range(0, n_uniform, vmap_batch):
        n_b    = min(vmap_batch, n_uniform - start)
        vx_b   = rng.uniform(3.0,  cfg.vx_scale, n_b)
        vy_b   = rng.uniform(-cfg.vy_scale, cfg.vy_scale, n_b)
        del_b  = rng.uniform(-cfg.delta_scale, cfg.delta_scale, n_b)
        wz_ref = vx_b * del_b / (_L * (1.0 + _K_US * vx_b ** 2) + 1e-6)
        wz_b   = wz_ref + rng.uniform(-cfg.wz_scale, cfg.wz_scale, n_b)
        Mz_b   = rng.uniform(-MZ_NORM, MZ_NORM, n_b)
        mu_b   = rng.uniform(0.8, 1.6, n_b)

        E, Mzn, En, Fe, rho = _run_batch(vx_b, vy_b, wz_b, del_b, Mz_b, mu_b)
        all_E.append(E); all_Mz.append(Mzn)
        all_En.append(En); all_Fe.append(Fe); all_rho.append(rho)

        if (start // vmap_batch) % 20 == 0:
            pct = (start + n_b) / n_samples * 100
            print(f"    {pct:4.0f}%  ρ_sat={(rho > 0.92).mean() * 100:.1f}%")

    # ── Saturation-biased ─────────────────────────────────────────────────
    print("  Saturation-biased sampling...")
    for start in range(0, n_biased, vmap_batch):
        n_b    = min(vmap_batch, n_biased - start)
        vx_b   = rng.uniform(12.0, cfg.vx_scale, n_b)
        vy_b   = rng.uniform(-cfg.vy_scale, cfg.vy_scale, n_b)
        sign   = rng.choice([-1., 1.], n_b)
        del_b  = sign * rng.uniform(0.12, cfg.delta_scale, n_b)
        wz_ref = vx_b * del_b / (_L * (1.0 + _K_US * vx_b ** 2) + 1e-6)
        wz_b   = wz_ref + rng.uniform(-2.0, 2.0, n_b)
        Mz_b   = rng.uniform(-MZ_NORM, MZ_NORM, n_b)
        mu_b   = rng.uniform(0.9, 1.6, n_b)

        E, Mzn, En, Fe, rho = _run_batch(vx_b, vy_b, wz_b, del_b, Mz_b, mu_b)
        all_E.append(E); all_Mz.append(Mzn)
        all_En.append(En); all_Fe.append(Fe); all_rho.append(rho)

    E_raw  = np.concatenate(all_E,   axis=0)
    Mz_arr = np.concatenate(all_Mz,  axis=0)
    E_next = np.concatenate(all_En,  axis=0)
    F_e    = np.concatenate(all_Fe,  axis=0)
    rho    = np.concatenate(all_rho, axis=0)

    idx    = rng.permutation(len(E_raw))
    E_raw  = E_raw[idx]; Mz_arr = Mz_arr[idx]
    E_next = E_next[idx]; F_e = F_e[idx]; rho = rho[idx]

    E_norm      = _norm_E(E_raw,  cfg)
    E_next_norm = _norm_E(E_next, cfg)
    f_scale     = np.array([cfg.wz_scale, cfg.vy_scale,
                             cfg.vx_scale, cfg.delta_scale])
    F_e_norm    = F_e / (f_scale[np.newaxis, :] + 1e-9)

    return E_raw, E_norm, Mz_arr, E_next_norm, F_e_norm, rho


# ─────────────────────────────────────────────────────────────────────────────
# §6  Regime split on H_net ρ
# ─────────────────────────────────────────────────────────────────────────────

def split_by_regime_hnet(E_norm, Mz_norm, E_next_norm, F_e_norm, rho_vals):
    masks  = [rho_vals < 0.70,
              (rho_vals >= 0.70) & (rho_vals <= 0.92),
              rho_vals > 0.92]
    labels = ["linear  (<0.70)", "transition (0.70–0.92)", "saturation (>0.92)"]
    regimes = []
    for k, (mask, lbl) in enumerate(zip(masks, labels)):
        pct = mask.mean() * 100
        tag = "[OK]" if (k < 2 or pct >= 20.0) else "[WARN]"
        print(f"  Regime {k} {lbl}: {mask.sum():,} ({pct:.1f}%) {tag}")
        regimes.append((E_norm[mask], Mz_norm[mask],
                        E_next_norm[mask], F_e_norm[mask]))
    if (rho_vals > 0.92).mean() < 0.15:
        print("  [WARN] Saturation < 15% — try --n_samples 750000")
    return regimes


# ─────────────────────────────────────────────────────────────────────────────
# §7  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",  type=int,   default=500_000)
    parser.add_argument("--n_epochs",   type=int,   default=500)
    parser.add_argument("--n_alt",      type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--vmap_batch", type=int,   default=512,
                        help="H_net vmap batch (reduce to 128 if OOM)")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--output",     type=str,   default="trained/koopman_hnet")
    parser.add_argument("--setup",      type=str,   default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--quick",      action="store_true",
                        help="20k samples / 50 epochs — pipeline smoke test")
    args = parser.parse_args()

    if args.quick:
        args.n_samples = 20_000; args.n_epochs = 50; args.vmap_batch = 256
        print("[KoopmanHnet] QUICK MODE")

    t_start = time.perf_counter()
    cfg     = KoopmanTVConfig()
    modules = [_PHI0, _PHI1, _PHI2]
    dims    = [cfg.m0, cfg.m1, cfg.m2]
    dt      = 0.005

    # ── Vehicle ───────────────────────────────────────────────────────────
    print("[KoopmanHnet] Initialising H_net vehicle...")
    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)

    setup = _DEFAULT_SETUP_28
    if args.setup:
        loaded = jnp.array(np.load(args.setup), dtype=jnp.float32)
        setup  = loaded[:28] if loaded.shape[0] >= 28 else setup
        print(f"  Setup: {args.setup}")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\n[KoopmanHnet] Generating {args.n_samples:,} H_net samples...")
    t0 = time.perf_counter()
    E_raw, E_norm, Mz_norm, E_next_norm, F_e_norm, rho_vals = (
        generate_hnet_training_data(
            args.n_samples, vehicle, setup,
            dt=dt, seed=args.seed, vmap_batch=args.vmap_batch, cfg=cfg,
        )
    )
    t_gen = time.perf_counter() - t0
    print(f"  Done in {t_gen:.1f}s  ({args.n_samples / t_gen:,.0f} samples/s)")
    print(f"  ρ_sat (>0.92): {(rho_vals > 0.92).mean() * 100:.1f}%  (target ≥ 20%)")

    # ── Regime split ──────────────────────────────────────────────────────
    print("\n[KoopmanHnet] Regime split...")
    regimes = split_by_regime_hnet(E_norm, Mz_norm, E_next_norm, F_e_norm, rho_vals)

    # ── EDMD-DL ───────────────────────────────────────────────────────────
    trained_params, trained_K, trained_b, trained_L = [], [], [], []

    for k, (E_k, Mz_k, En_k, Fe_k) in enumerate(regimes):
        print(f"\n[KoopmanHnet] === Regime {k}  (m={dims[k]}, n={len(E_k):,}) ===")
        if len(E_k) < 512:
            print("  [WARN] Too few — using full dataset")
            E_k, Mz_k, En_k, Fe_k = E_norm, Mz_norm, E_next_norm, F_e_norm

        phi_params, K, b = train_regime(
            modules[k], dims[k], E_k, Mz_k, En_k, Fe_k,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, n_alt_iters=args.n_alt, seed=args.seed + k,
        )
        phi_k = lambda e, _p=phi_params, _k=k: modules[_k].apply(_p, e)
        print(f"  Computing LQR gain...")
        L = compute_koopman_gain(K, b, phi_k, dims[k], cfg)
        trained_params.append(phi_params)
        trained_K.append(K); trained_b.append(b); trained_L.append(L)

    # ── Save ─────────────────────────────────────────────────────────────
    ops = KoopmanOperators(
        K0=jnp.array(trained_K[0]), b0=jnp.array(trained_b[0]), L0=jnp.array(trained_L[0]),
        K1=jnp.array(trained_K[1]), b1=jnp.array(trained_b[1]), L1=jnp.array(trained_L[1]),
        K2=jnp.array(trained_K[2]), b2=jnp.array(trained_b[2]), L2=jnp.array(trained_L[2]),
    )
    bundle = KoopmanTVBundle(
        ops=ops, phi0=trained_params[0], phi1=trained_params[1], phi2=trained_params[2],
        cfg=KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": 0.0}),
    )
    save_koopman_bundle(bundle, args.output)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print("\n[KoopmanHnet] Sanity checks...")
    from powertrain.modes.advanced.koopman_tv import koopman_mz_reference
    bt   = bundle._replace(cfg=KoopmanTVConfig(**{**cfg._asdict(), "trained_blend": 1.0}))
    _fz  = jnp.full(4, 750.0); _z4 = jnp.zeros(4)

    Mz0, _ = koopman_mz_reference(
        jnp.array(0.0), jnp.array(0.0), jnp.array(3.0), jnp.array(15.0),
        jnp.array(0.05), _z4, _z4, _fz, jnp.array(1.3), bt)
    print(f"  Mz(ψ̇_err=0): {float(Mz0):.2f} Nm  "
          f"{'[PASS]' if abs(float(Mz0)) < 50 else '[WARN]'}")

    Mz1, _ = koopman_mz_reference(
        jnp.array(1.0), jnp.array(0.0), jnp.array(15.0), jnp.array(15.0),
        jnp.array(0.05), _z4, _z4, _fz, jnp.array(1.3), bt)
    dir_ok = float(Mz1) < 0
    print(f"  Mz(ψ̇_err=1): {float(Mz1):.2f} Nm  "
          f"{'[PASS]' if dir_ok else '[FAIL — wrong sign]'}  "
          f"(ratio vs PD: {abs(float(Mz1)) / (cfg.Kp_fallback + 1e-6):.2f}×)")

    dMz = jax.grad(lambda e: koopman_mz_reference(
        e, jnp.array(0.0), jnp.array(15.0), jnp.array(15.0), jnp.array(0.0),
        _z4, _z4, _fz, jnp.array(1.3), bt)[0])(jnp.array(1.0))
    print(f"  Gradient: {float(dMz):.2f}  "
          f"{'[PASS]' if bool(jnp.isfinite(dMz)) else '[FAIL — NaN]'}")

    sat_pct = float((rho_vals > 0.92).mean() * 100)
    print(f"  Saturation coverage: {sat_pct:.1f}%  "
          f"{'[PASS]' if sat_pct >= 20 else '[WARN]'}")

    mins = (time.perf_counter() - t_start) / 60
    print(f"\n[KoopmanHnet] Done in {mins:.1f} min → {args.output}/")
    print("  Next: load_koopman_bundle('{}', trained_blend=0.3) → shakedown".format(args.output))


if __name__ == "__main__":
    main()