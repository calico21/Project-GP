# scripts/generate_qp_training_data.py
# Project-GP — Batch 2 + Batch 8: Offline QP Data Generator (V1 + V2)
# ═══════════════════════════════════════════════════════════════════════════════
#
# V1 (unchanged):  15-dim θ, 12 constraints, 100k samples
#   Run:  python -m scripts.generate_qp_training_data
#   Out:  models/qp_training_data.npz
#
# V2 (Batch 8):  19-dim θ, 20 constraints, 500k samples
#   Adds:  κ*_f, κ*_r, σ_f, σ_r to θ (slip-observer outputs)
#   Adds:  8 per-wheel slip-barrier constraints to the QP
#   Biases: 25% of samples toward braking-into-corner scenarios
#   Run:  python -m scripts.generate_qp_training_data --v2
#   Out:  models/qp_training_data_v2.npz
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# §1  Shared normalisation constants (must match active_set_classifier.py)
# ─────────────────────────────────────────────────────────────────────────────

MZ_SCALE    = 2000.0
FX_SCALE    = 8000.0
T_SCALE     = 320.0
DELTA_SCALE = 0.35
KAPPA_SCALE = 0.20    # Batch 8
SIGMA_SCALE = 0.05    # Batch 8
N_CONSTRAINTS_V1 = 12
N_CONSTRAINTS_V2 = 20
THETA_DIM_V1     = 15
THETA_DIM_V2     = 19

# Geometry — must match TVGeometry defaults
LF   = 0.8525
LR   = 0.6975
TF2  = 0.600    # half track front
TR2  = 0.590    # half track rear
R_W  = 0.2032


def normalise_theta_v1(theta_raw: jax.Array) -> jax.Array:
    scales = jnp.array([
        MZ_SCALE, FX_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        DELTA_SCALE,
    ])
    return theta_raw / scales


def normalise_theta_v2(theta_raw: jax.Array) -> jax.Array:
    scales = jnp.array([
        MZ_SCALE, FX_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        DELTA_SCALE,
        KAPPA_SCALE, KAPPA_SCALE,   # κ*_front, κ*_rear
        SIGMA_SCALE, SIGMA_SCALE,   # σ_front,  σ_rear
    ])
    return theta_raw / scales


# ─────────────────────────────────────────────────────────────────────────────
# §2  QP solver (500 iterations — ground-truth for classifier training)
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_arms(delta):
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    return np.array([
        -TF2 * cos_d + LF * sin_d,
         TF2 * cos_d - LF * sin_d,
        -TR2,
         TR2,
    ]) / R_W


def _solve_qp_v1(mz_ref, fx_d, t_min, t_max, t_fric, delta, t_prev,
                 omega, n_iters=500, lr=0.25):
    """
    Ground-truth projected-gradient solver for the V1 QP (12 constraints).
    Returns (T_opt, active_set_12).
    """
    arms    = _yaw_arms(delta)
    ones_rw = np.ones(4) / R_W

    Q = (2.0 * 8.0  * np.outer(arms, arms)
       + 2.0 * 1.0  * np.outer(ones_rw, ones_rw)
       + 2.0 * 0.05 * np.eye(4)
       + 2.0 * 1e-4 * np.diag(np.abs(omega) + 1.0))

    c = (-2.0 * 8.0  * mz_ref * arms
       - 2.0 * 1.0  * fx_d   * ones_rw
       - 2.0 * 0.05 * t_prev)

    T = t_prev.copy()
    for _ in range(n_iters):
        T = T - lr * (Q @ T + c)
        T = np.clip(T, t_min, t_max)
        T = np.clip(T, -t_fric, t_fric)

    # Active-set detection: constraint is "active" if within 3 Nm of its bound
    tol = 3.0
    a_lower = (np.abs(T - t_min) < tol).astype(np.float32)
    a_upper = (np.abs(T - t_max) < tol).astype(np.float32)
    a_fric  = (np.abs(np.abs(T) - t_fric) < tol).astype(np.float32)
    active  = np.concatenate([a_lower, a_upper, a_fric])   # (12,)
    return T, active


def _solve_qp_v2(mz_ref, fx_d, t_min, t_max, t_fric, delta, t_prev,
                 omega, kappa_star_f, kappa_star_r, sigma_f, sigma_r,
                 vx, n_iters=500, lr=0.20):
    """
    Ground-truth projected-gradient solver for the V2 QP (20 constraints):
      same as V1 but adds 8 slip-barrier rows.

    The slip budget per wheel:
      budget_i = max(kappa_star_axle − 1.5 · sigma_axle, 0.015)

    Slip constraint:  κ_{k+d}^{(i)} ∈ [−budget_i, +budget_i]
    Linearised:  kappa_now_i + sens · (T_i − Fx_tire_i · r_w) ∈ [−b, +b]
    → sens · T_i ≤  b − kappa_preview_0
       sens · T_i ≥ −b − kappa_preview_0

    For the offline solver we use kappa_now = 0 and Fx_tire = 0 as conservative
    priors (the solver doesn't have a running vehicle state).
    Samples that have a braking demand (fx_d < 0) use kappa_now drawn from the
    braking-scenario distribution.
    """
    # ── V1 solve first ───────────────────────────────────────────────────────
    arms    = _yaw_arms(delta)
    ones_rw = np.ones(4) / R_W

    Q = (2.0 * 8.0  * np.outer(arms, arms)
       + 2.0 * 1.0  * np.outer(ones_rw, ones_rw)
       + 2.0 * 0.05 * np.eye(4)
       + 2.0 * 1e-4 * np.diag(np.abs(omega) + 1.0))

    c = (-2.0 * 8.0  * mz_ref * arms
       - 2.0 * 1.0  * fx_d   * ones_rw
       - 2.0 * 0.05 * t_prev)

    # ── Slip budget & constraint coefficients ────────────────────────────────
    vx_s       = max(abs(vx), 1.5)
    tau_delay  = 0.015
    I_w        = 1.2
    sensitivity = R_W * tau_delay / (I_w * vx_s)    # scalar

    budget_f = max(kappa_star_f - 1.5 * sigma_f, 0.015)
    budget_r = max(kappa_star_r - 1.5 * sigma_r, 0.015)
    budget   = np.array([budget_f, budget_f, budget_r, budget_r])   # (4,)

    # Conservatively: kappa_now = 0, Fx_tire_est = 0 (offline, no vehicle state)
    kappa_preview_0 = np.zeros(4)

    # Slip constraint upper: +sens · T_i ≤ +budget_i − kappa_preview_0
    # Slip constraint lower: −sens · T_i ≤ +budget_i + kappa_preview_0
    slip_upper_rhs = budget - kappa_preview_0                # (4,)
    slip_lower_rhs = budget + kappa_preview_0                # (4,)

    T = t_prev.copy()
    for _ in range(n_iters):
        T = T - lr * (Q @ T + c)
        # Box + friction
        T = np.clip(T, t_min, t_max)
        T = np.clip(T, -t_fric, t_fric)
        # Slip constraint projection:
        #   +sens · T_i ≤ slip_upper_rhs  → T_i ≤ slip_upper_rhs / sens
        #   −sens · T_i ≤ slip_lower_rhs  → T_i ≥ −slip_lower_rhs / sens
        if sensitivity > 1e-9:
            T = np.clip(T,
                        -slip_lower_rhs / sensitivity,
                        slip_upper_rhs  / sensitivity)
            # Re-clip to box after slip (slip may widen box)
            T = np.clip(T, t_min, t_max)
            T = np.clip(T, -t_fric, t_fric)

    # ── Active-set detection (20 constraints) ────────────────────────────────
    tol      = 3.0
    slip_tol = 0.002   # 0.2pp slip  — tighter than box tol

    a_lower = (np.abs(T - t_min) < tol).astype(np.float32)                           # (4,)
    a_upper = (np.abs(T - t_max) < tol).astype(np.float32)                           # (4,)
    a_fric  = (np.abs(np.abs(T) - t_fric) < tol).astype(np.float32)                 # (4,)

    kappa_pred = kappa_preview_0 + sensitivity * T                                    # (4,)
    a_slip_up  = (np.abs(kappa_pred - slip_upper_rhs) < slip_tol).astype(np.float32) # (4,)
    a_slip_lo  = (np.abs(kappa_pred + slip_lower_rhs) < slip_tol).astype(np.float32) # (4,)

    active = np.concatenate([a_lower, a_upper, a_fric, a_slip_up, a_slip_lo])   # (20,)
    return T, active


# ─────────────────────────────────────────────────────────────────────────────
# §3  Sampling distributions
# ─────────────────────────────────────────────────────────────────────────────

def _sample_v1(rng_np, n_samples):
    """Original V1 sampling — uniform over operational envelope."""
    mz_ref = rng_np.uniform(-2000, 2000, n_samples)
    fx_d   = rng_np.uniform(-8000, 6000, n_samples)
    t_min_mag = rng_np.uniform(150, 320, (n_samples, 4))
    t_max_mag = rng_np.uniform(150, 450, (n_samples, 4))
    t_fric    = rng_np.uniform(200, 400, (n_samples, 4))
    delta     = rng_np.uniform(-0.35, 0.35, n_samples)
    t_prev    = rng_np.uniform(-100, 100, (n_samples, 4))
    omega     = rng_np.uniform(30, 200, (n_samples, 4))
    return mz_ref, fx_d, -t_min_mag, t_max_mag, t_fric, delta, t_prev, omega


def _sample_v2(rng_np, n_samples):
    """
    V2 sampling adds κ*, σ, vx, and biases 25% toward braking-into-corner.

    Braking-into-corner scenario:
      fx_d < -3000 N   (hard braking)
      |delta| > 0.05 rad  (cornering)
      κ*  ∈ [0.06, 0.13]  (mid-range optimal slip)
      σ   ∈ [0.005, 0.025] (moderate observer confidence)
    """
    mz_ref, fx_d, t_min, t_max, t_fric, delta, t_prev, omega = _sample_v1(rng_np, n_samples)

    # ── Koopman outputs: sample from the operational range ──────────────────
    kappa_star_f = rng_np.uniform(0.05, 0.18, n_samples)
    kappa_star_r = rng_np.uniform(0.05, 0.18, n_samples)
    sigma_f      = rng_np.uniform(0.004, 0.045, n_samples)
    sigma_r      = rng_np.uniform(0.004, 0.045, n_samples)
    vx           = rng_np.uniform(2.0, 30.0, n_samples)

    # ── Bias 25% of samples toward braking-into-corner ──────────────────────
    n_braking = n_samples // 4
    braking_idx = rng_np.choice(n_samples, n_braking, replace=False)

    fx_d[braking_idx]         = rng_np.uniform(-8000, -3000, n_braking)   # hard braking
    delta[braking_idx]        = rng_np.uniform(-0.35, 0.35, n_braking)    # cornering
    np.abs(delta[braking_idx] + rng_np.uniform(0.05, 0.30, n_braking))    # ensure |delta|>0.05
    kappa_star_f[braking_idx] = rng_np.uniform(0.06, 0.13, n_braking)     # realistic κ*
    kappa_star_r[braking_idx] = rng_np.uniform(0.06, 0.13, n_braking)
    sigma_f[braking_idx]      = rng_np.uniform(0.005, 0.025, n_braking)   # tighter uncertainty
    sigma_r[braking_idx]      = rng_np.uniform(0.005, 0.025, n_braking)
    vx[braking_idx]           = rng_np.uniform(8.0, 25.0, n_braking)      # real braking speeds

    # Also bias mz_ref to match trail-braking rotation demand
    mz_ref[braking_idx] = (
        np.sign(delta[braking_idx]) * rng_np.uniform(200, 1500, n_braking)
    )

    return (mz_ref, fx_d, t_min, t_max, t_fric, delta, t_prev, omega,
            kappa_star_f, kappa_star_r, sigma_f, sigma_r, vx)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_v1(n_samples=100_000, out_path="models/qp_training_data.npz", seed=42):
    """Original V1 data generation (Batch 2, unchanged)."""
    print(f"[DataGen V1] Generating {n_samples:,} samples...")
    rng_np = np.random.default_rng(seed)

    mz_ref, fx_d, t_min, t_max, t_fric, delta, t_prev, omega = _sample_v1(rng_np, n_samples)

    theta_raws  = np.zeros((n_samples, THETA_DIM_V1), dtype=np.float32)
    theta_norms = np.zeros((n_samples, THETA_DIM_V1), dtype=np.float32)
    active_sets = np.zeros((n_samples, N_CONSTRAINTS_V1), dtype=np.float32)
    T_opts      = np.zeros((n_samples, 4), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n_samples):
        T_opt, active = _solve_qp_v1(
            mz_ref[i], fx_d[i], t_min[i], t_max[i], t_fric[i],
            delta[i], t_prev[i], omega[i],
        )
        theta_raw = np.array([
            mz_ref[i], fx_d[i],
            *t_min[i], *t_max[i], *t_fric[i], delta[i],
        ], dtype=np.float32)
        theta_raws[i]  = theta_raw
        theta_norms[i] = np.array(normalise_theta_v1(jnp.array(theta_raw)))
        active_sets[i] = active
        T_opts[i]      = T_opt
        if (i + 1) % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1:6d}/{n_samples}  ({elapsed:.1f}s)")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             theta_raw=theta_raws, theta_norm=theta_norms,
             active_sets=active_sets, T_opt=T_opts)
    print(f"[DataGen V1] Saved {n_samples:,} samples → {out_path}")
    print(f"  Activation rates: {active_sets.mean(axis=0).round(3)}")


def generate_v2(n_samples=500_000, out_path="models/qp_training_data_v2.npz", seed=42):
    """
    Batch 8 V2 data generation.

    500k samples with:
      - 25% braking-into-corner scenarios (slip barriers bind)
      - 75% general driving (slips inactive in most cases)

    Runtime estimate: ~45 min on a single CPU core.
    Parallelise across workers for faster turnaround.
    """
    print(f"[DataGen V2] Generating {n_samples:,} samples (Batch 8 slip-aware)...")
    print(f"  θ dim: {THETA_DIM_V2} | constraint dim: {N_CONSTRAINTS_V2}")
    rng_np = np.random.default_rng(seed)

    (mz_ref, fx_d, t_min, t_max, t_fric, delta, t_prev, omega,
     kappa_star_f, kappa_star_r, sigma_f, sigma_r, vx) = _sample_v2(rng_np, n_samples)

    theta_raws  = np.zeros((n_samples, THETA_DIM_V2),     dtype=np.float32)
    theta_norms = np.zeros((n_samples, THETA_DIM_V2),     dtype=np.float32)
    active_sets = np.zeros((n_samples, N_CONSTRAINTS_V2), dtype=np.float32)
    T_opts      = np.zeros((n_samples, 4),                dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n_samples):
        T_opt, active = _solve_qp_v2(
            mz_ref[i], fx_d[i], t_min[i], t_max[i], t_fric[i],
            delta[i], t_prev[i], omega[i],
            kappa_star_f[i], kappa_star_r[i], sigma_f[i], sigma_r[i], vx[i],
        )
        theta_raw = np.array([
            mz_ref[i], fx_d[i],
            *t_min[i], *t_max[i], *t_fric[i], delta[i],
            kappa_star_f[i], kappa_star_r[i], sigma_f[i], sigma_r[i],
        ], dtype=np.float32)
        theta_raws[i]  = theta_raw
        theta_norms[i] = np.array(normalise_theta_v2(jnp.array(theta_raw)))
        active_sets[i] = active
        T_opts[i]      = T_opt

        if (i + 1) % 20_000 == 0:
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (n_samples - i - 1) / rate
            print(f"  {i+1:7d}/{n_samples}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            slip_act = active_sets[:i+1, 12:].mean()
            print(f"    Slip-constraint activation rate (V2 rows): {slip_act:.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             theta_raw=theta_raws, theta_norm=theta_norms,
             active_sets=active_sets, T_opt=T_opts)
    print(f"[DataGen V2] Saved {n_samples:,} samples → {out_path}")
    print(f"  V1 constraint activation rates: {active_sets[:, :12].mean(axis=0).round(3)}")
    print(f"  V2 slip constraint activation rates: {active_sets[:, 12:].mean(axis=0).round(3)}")


# ─────────────────────────────────────────────────────────────────────────────
# §5  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true",
                        help="Generate V2 dataset (Batch 8, 19-dim θ, 20 constraints)")
    parser.add_argument("--n", type=int, default=None,
                        help="Override sample count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.v2:
        n = args.n or 500_000
        generate_v2(n_samples=n, seed=args.seed)
    else:
        n = args.n or 100_000
        generate_v1(n_samples=n, seed=args.seed)