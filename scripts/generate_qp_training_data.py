# scripts/generate_qp_training_data.py
# Project-GP — Batch 2: Offline QP Data Generator for Active-Set Classifier
# ═══════════════════════════════════════════════════════════════════════════════
#
# Solves the torque-vectoring QP offline for 100k parameter samples drawn from
# the operational envelope, records the active constraint set at each optimum,
# and saves (θ_norm, active_set, T_opt) to disk for classifier training.
#
# The QP solved here is the EXACT same problem as the 12-iteration projected-
# gradient SOCP in torque_vectoring.py, but with 500 iterations to guarantee
# convergence and active-set stability. This is a one-time offline cost.
#
# CONSTRAINT NUMBERING (active_set vector, 12 bits):
#   [0:4]   T ≥ T_min   (lower motor box)   — active when T_i = T_min_i
#   [4:8]   T ≤ T_max   (upper motor box)   — active when T_i = T_max_i
#   [8:12]  T ≤ T_fric  (friction circle)   — active when |T_i|/r_w = μ_i·Fz_i
#
# PARAMETER VECTOR θ (15-dim, normalised to [-1, 1] or [0, 1]):
#   [0]     M_z_ref / MZ_SCALE             yaw moment reference
#   [1]     F_x_d / FX_SCALE               longitudinal force demand
#   [2:6]   T_min / T_SCALE                lower motor limits per wheel
#   [6:10]  T_max / T_SCALE                upper motor limits per wheel
#   [10:14] T_fric / T_SCALE               friction capacity per wheel
#   [14]    delta / DELTA_SCALE             steering angle
#
# Run:
#   python -m scripts.generate_qp_training_data
#   → saves  models/qp_training_data.npz
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# §1  Normalisation constants (match classifier input exactly)
# ─────────────────────────────────────────────────────────────────────────────

MZ_SCALE    = 2000.0    # Nm    max yaw moment reference
FX_SCALE    = 8000.0    # N     max longitudinal force demand
T_SCALE     = 320.0     # Nm    max wheel torque (motor peak per-wheel)
DELTA_SCALE = 0.35      # rad   max steering angle
N_CONSTRAINTS = 12
THETA_DIM     = 15

# Geometry — must match torque_vectoring.py TVGeometry defaults
LF    = 0.8525   # m
LR    = 0.6975   # m
TF2   = 0.600    # m  half track front
TR2   = 0.590    # m  half track rear
R_W   = 0.2032   # m  loaded radius


def normalise_theta(theta_raw: jax.Array) -> jax.Array:
    """Map raw θ to [-1,1] / [0,1] for MLP input."""
    scales = jnp.array([
        MZ_SCALE, FX_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,   # T_min (negative → /T_SCALE gives [-1,0])
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,   # T_max
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,   # T_fric (always positive)
        DELTA_SCALE,
    ])
    return theta_raw / scales


# ─────────────────────────────────────────────────────────────────────────────
# §2  QP solver (high-iteration, accurate, used only offline)
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_arms(delta: float) -> np.ndarray:
    """Yaw moment arm per wheel [m], matching torque_vectoring.yaw_moment_arms."""
    cd, sd = np.cos(delta), np.sin(delta)
    return np.array([
        -TF2 * cd + LF * sd,
        +TF2 * cd - LF * sd,
        -TR2,
        +TR2,
    ]) / R_W


def solve_qp_accurate(
    mz_ref: float, fx_d: float,
    t_min: np.ndarray, t_max: np.ndarray, t_fric: np.ndarray,
    delta: float, t_prev: np.ndarray,
    n_iters: int = 500,
    lr: float = 0.80,
    w_mz: float = 8.0, w_fx: float = 1.0,
    w_rate: float = 0.05, w_loss: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    High-accuracy projected-gradient solve. Returns (T_opt, active_set).

    active_set[i] = 1 if constraint i is active at optimum (within 1 Nm tol).
    Constraint ordering: [lower_box(4), upper_box(4), friction(4)].
    """
    arms = _yaw_arms(delta)
    T = np.clip(t_prev.copy(), t_min, t_max)

    for _ in range(n_iters):
        mz  = np.dot(T, arms)
        fx  = np.sum(T) / R_W

        g_mz   = 2.0 * w_mz   * (mz - mz_ref) * arms
        g_fx   = 2.0 * w_fx   * (fx - fx_d)   * np.ones(4) / R_W
        g_rate = 2.0 * w_rate  * (T - t_prev)
        # Copper-loss proxy: T²ω — ω approximated as |T|/R_W/0.5 (slip estimate)
        omega_approx = np.abs(T) / (R_W * 0.5) + 1.0
        g_loss = 2.0 * w_loss  * T * omega_approx

        grad = g_mz + g_fx + g_rate + g_loss

        # Softplus friction barrier gradient
        fric_ratio = (np.abs(T) / R_W) / (t_fric / R_W + 1e-6)
        barrier = 200.0 * np.maximum(fric_ratio - 0.95, 0.0)
        g_barrier = barrier * np.sign(T) / R_W / (t_fric / R_W + 1e-6) * 50.0
        grad = grad + g_barrier

        T = T - lr * grad
        # Project onto box
        T = np.clip(T, t_min, t_max)
        # Project onto friction circle (soft — just clip magnitude)
        for i in range(4):
            if np.abs(T[i]) > t_fric[i]:
                T[i] = t_fric[i] * np.sign(T[i])

    # Determine active constraints (within 1.5 Nm tolerance)
    TOL = 1.5
    active = np.zeros(N_CONSTRAINTS, dtype=np.float32)
    active[0:4]  = (T <= t_min + TOL).astype(np.float32)   # lower box
    active[4:8]  = (T >= t_max - TOL).astype(np.float32)   # upper box
    active[8:12] = (np.abs(T) >= t_fric - TOL).astype(np.float32)  # friction

    return T, active


# ─────────────────────────────────────────────────────────────────────────────
# §3  Operational envelope sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_operational_envelope(n: int, seed: int = 42) -> np.ndarray:
    """
    Sample n parameter vectors from the Ter27 operational envelope.
    Returns raw θ array of shape (n, THETA_DIM).
    """
    rng = np.random.default_rng(seed)

    # Yaw moment reference: covers full bidirectional range
    mz_ref = rng.uniform(-MZ_SCALE, MZ_SCALE, n)

    # Longitudinal force: braking and acceleration
    fx_d = rng.uniform(-FX_SCALE * 0.5, FX_SCALE, n)

    # Motor limits: asymmetric (traction vs regen), thermal derating
    t_max_base = rng.uniform(150.0, T_SCALE, (n, 4))   # positive: drive
    t_min_base = rng.uniform(-120.0, -20.0, (n, 4))    # negative: regen
    # Thermal derating: randomly derate 1–2 corners
    derate = rng.uniform(0.6, 1.0, (n, 4))
    t_max = t_max_base * derate
    t_min = t_min_base * derate

    # Friction capacity: depends on Fz and mu
    fz = rng.uniform(500.0, 2500.0, (n, 4))            # [N] per corner
    mu = rng.uniform(1.0, 1.6, (n, 4))                 # combined mu
    t_fric = mu * fz * R_W                              # [Nm]
    t_fric = np.clip(t_fric, 20.0, T_SCALE)

    # Steering angle
    delta = rng.uniform(-DELTA_SCALE, DELTA_SCALE, n)

    theta = np.column_stack([mz_ref, fx_d, t_min, t_max, t_fric, delta])
    assert theta.shape == (n, THETA_DIM), theta.shape
    return theta


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main data generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate(n_samples: int = 100_000, out_path: str = "models/qp_training_data.npz"):
    print(f"[QP DataGen] Generating {n_samples:,} samples...")
    print(f"[QP DataGen] θ dim={THETA_DIM}  |  active-set dim={N_CONSTRAINTS}")

    theta_raw = sample_operational_envelope(n_samples)
    theta_norm = theta_raw / np.array([
        MZ_SCALE, FX_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        T_SCALE, T_SCALE, T_SCALE, T_SCALE,
        DELTA_SCALE,
    ])

    active_sets = np.zeros((n_samples, N_CONSTRAINTS), dtype=np.float32)
    t_opts      = np.zeros((n_samples, 4), dtype=np.float32)
    t_prev      = np.zeros(4)

    t0 = time.perf_counter()
    log_every = n_samples // 20

    for i in range(n_samples):
        θ = theta_raw[i]
        mz_ref, fx_d = float(θ[0]), float(θ[1])
        t_min  = θ[2:6];   t_max  = θ[6:10]
        t_fric = θ[10:14]; delta  = float(θ[14])

        T_opt, active = solve_qp_accurate(
            mz_ref, fx_d, t_min, t_max, t_fric, delta, t_prev)

        active_sets[i] = active
        t_opts[i]      = T_opt
        t_prev         = T_opt   # warm-start chain

        if (i + 1) % log_every == 0:
            elapsed  = time.perf_counter() - t0
            eta      = elapsed / (i + 1) * (n_samples - i - 1)
            n_active = active.sum()
            print(f"  [{i+1:6d}/{n_samples}]  "
                  f"active constraints avg: {active_sets[:i+1].mean(axis=0).sum():.2f}  "
                  f"ETA: {eta:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"\n[QP DataGen] Done in {elapsed:.1f}s  ({elapsed/n_samples*1000:.2f}ms/sample)")

    # Statistics
    mean_active = active_sets.mean(axis=0)
    print("\n[QP DataGen] Constraint activation rates:")
    labels = ([f"T_min_{i}" for i in range(4)]
              + [f"T_max_{i}" for i in range(4)]
              + [f"T_fric_{i}" for i in range(4)])
    for lbl, rate in zip(labels, mean_active):
        bar = "█" * int(rate * 20)
        print(f"  {lbl:12s}  {rate:.3f}  {bar}")

    n_zero_active = (active_sets.sum(axis=1) == 0).sum()
    print(f"\n  Interior (no active): {n_zero_active:,} ({100*n_zero_active/n_samples:.1f}%)")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             theta_norm=theta_norm.astype(np.float32),
             active_sets=active_sets,
             t_opts=t_opts.astype(np.float32),
             theta_raw=theta_raw.astype(np.float32))
    print(f"[QP DataGen] Saved → {out_path}")
    print(f"  theta_norm: {theta_norm.shape}  active_sets: {active_sets.shape}  t_opts: {t_opts.shape}")
    return out_path


if __name__ == "__main__":
    generate()