# optimization/quality_diversity.py
# Project-GP — JAX-Native MAP-Elites with Gradient-Guided Mutation
# ═══════════════════════════════════════════════════════════════════════════
#
# Behavior Descriptors (analytical, no simulation cost):
#   BD₁ = K_roll_f / (K_roll_f + K_roll_r)  ∈ [0.30, 0.70]   (balance axis)
#   BD₂ = sigmoid(log(k_f·k_r) / log(k_ref²))  ∈ [0.0,  1.0]  (stiffness axis)
#
# These are computed from setup params only — trivially differentiable,
# allowing gradient-guided mutation: mutate while staying in-cell via:
#   ∂/∂s[-quality(s) + λ·||BD(s) - BD_cell_center||²]
#
# Archive is a (N_B1, N_B2) grid of (quality, setup_vec) pairs.
# All operations are JIT-compatible via dynamic scatter (.at[i,j].set).
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from optimization.objectives import compute_skidpad_objective
from config.vehicles.ter26 import vehicle_params as VP

N_B1 = 15    # BD₁ grid resolution (roll balance axis)
N_B2 = 15    # BD₂ grid resolution (stiffness axis)
N_DIM = 28   # setup dimension

# Physical BD bounds (tuned for FS car dynamics range)
BD1_LO, BD1_HI = 0.30, 0.70   # front roll stiffness fraction
BD2_LO, BD2_HI = 0.15, 0.85   # normalized stiffness level


class MAPElitesArchive(NamedTuple):
    quality: jax.Array    # (N_B1, N_B2)  −∞ for empty cells
    setups:  jax.Array    # (N_B1, N_B2, N_DIM)
    bd_map:  jax.Array    # (N_B1, N_B2, 2)  cell center coordinates
    n_evals: jax.Array    # scalar int32


def init_archive() -> MAPElitesArchive:
    """Initialize empty archive with −∞ quality sentinel."""
    # Pre-compute cell center BD values for the gradient constraint
    b1 = jnp.linspace(BD1_LO, BD1_HI, N_B1)
    b2 = jnp.linspace(BD2_LO, BD2_HI, N_B2)
    b1g, b2g = jnp.meshgrid(b1, b2, indexing='ij')   # (N_B1, N_B2)
    bd_map = jnp.stack([b1g, b2g], axis=-1)            # (N_B1, N_B2, 2)

    from models.vehicle_dynamics import DEFAULT_SETUP
    return MAPElitesArchive(
        quality = jnp.full((N_B1, N_B2), -jnp.inf),
        setups  = jnp.zeros((N_B1, N_B2, N_DIM)),
        bd_map  = bd_map,
        n_evals = jnp.array(0, dtype=jnp.int32),
    )


@jax.jit
def compute_behavior_descriptors(setup: jax.Array) -> jax.Array:
    """
    Analytical BD computation — no simulation, trivially differentiable.
    Derived from objectives.py BUGFIX-B roll stiffness formula.
    """
    # Reuse Ter27 geometry constants from objectives.py
    mr_f = jnp.array(VP.get('motion_ratio_f_poly', [1.14, 2.5, 0.0]))[0]
    mr_r = jnp.array(VP.get('motion_ratio_r_poly', [1.16, 2.0, 0.0]))[0]
    t_w  = VP.get('track_front', 1.20)
    t_r  = VP.get('track_rear',  1.18)

    k_f, k_r     = setup[0], setup[1]
    arb_f, arb_r = setup[2], setup[3]

    Kroll_f = (k_f / mr_f**2) * t_w**2 * 0.5 + arb_f * t_w
    Kroll_r = (k_r / mr_r**2) * t_r**2 * 0.5 + arb_r * t_r

    # BD₁: front roll stiffness fraction (understeer/oversteer axis)
    bd1 = Kroll_f / (Kroll_f + Kroll_r + 1.0)

    # BD₂: total stiffness normalized to [0,1] via sigmoid of log-ratio
    k_ref = 40000.0
    bd2 = jax.nn.sigmoid(jnp.log(jnp.maximum(k_f * k_r, 1.0)) / jnp.log(k_ref**2) * 4.0 - 2.0)

    return jnp.array([
        jnp.clip(bd1, BD1_LO, BD1_HI),
        jnp.clip(bd2, BD2_LO, BD2_HI),
    ])


@jax.jit
def bd_to_cell_index(bd: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Map continuous BD → (i, j) grid index. Clipped, never out-of-bounds."""
    i = jnp.clip(
        jnp.floor((bd[0] - BD1_LO) / (BD1_HI - BD1_LO) * N_B1).astype(jnp.int32),
        0, N_B1 - 1,
    )
    j = jnp.clip(
        jnp.floor((bd[1] - BD2_LO) / (BD2_HI - BD2_LO) * N_B2).astype(jnp.int32),
        0, N_B2 - 1,
    )
    return i, j


@jax.jit
def update_archive(
    archive:  MAPElitesArchive,
    setup:    jax.Array,          # (N_DIM,) candidate
    quality:  jax.Array,          # scalar
) -> MAPElitesArchive:
    """
    Vectorized archive update. Inserts candidate iff quality exceeds cell incumbent.
    JAX dynamic scatter (.at[i,j].set) — compatible with jit and lax.scan.
    """
    bd   = compute_behavior_descriptors(setup)
    i, j = bd_to_cell_index(bd)

    improves = quality > archive.quality[i, j]

    new_quality = jnp.where(
        improves,
        archive.quality.at[i, j].set(quality),
        archive.quality,
    )
    new_setups = jnp.where(
        improves,
        archive.setups.at[i, j].set(setup),
        archive.setups,
    )

    return MAPElitesArchive(
        quality = new_quality,
        setups  = new_setups,
        bd_map  = archive.bd_map,
        n_evals = archive.n_evals + 1,
    )


@partial(jax.jit, static_argnums=(1,))
def gradient_guided_mutation(
    setup:       jax.Array,          # (N_DIM,) parent setup from archive cell
    quality_fn,                      # setup → scalar quality (e.g. skidpad grip)
    bd_target:   jax.Array,          # (2,) center of target cell for mutation
    key:         jax.Array,
    n_steps:     int = 8,
    lr:          float = 0.005,
    lbd_bd:      float = 8.0,        # BD constraint Lagrangian weight
    sigma_init:  float = 0.02,       # initial random perturbation (normalized)
) -> jax.Array:
    """
    Local gradient refinement within a behavioral niche.

    Minimizes: -quality(s) + λ·||BD(s) - BD_target||²
    Gradient steps stay on the quality gradient while being pushed
    toward the target cell's BD center — in-cell optimization.
    """
    from models.vehicle_dynamics import SETUP_LB, SETUP_UB
    lb = jnp.array(SETUP_LB)
    ub = jnp.array(SETUP_UB)

    # Random perturbation to escape parent's basin
    noise  = jax.random.normal(key, setup.shape) * sigma_init * (ub - lb)
    s_init = jnp.clip(setup + noise, lb, ub)

    def constrained_loss(s: jax.Array) -> jax.Array:
        q   = quality_fn(s)
        bd  = compute_behavior_descriptors(s)
        pen = lbd_bd * jnp.sum((bd - bd_target) ** 2)
        return -q + pen     # minimize negative quality + BD deviation

    def opt_step(s, _):
        g  = jax.grad(constrained_loss)(s)
        sn = s - lr * g
        return jnp.clip(sn, lb, ub), None

    s_final, _ = jax.lax.scan(opt_step, s_init, None, length=n_steps)
    return s_final


@partial(jax.jit, static_argnums=(1,))
def map_elites_step(
    archive:      MAPElitesArchive,
    quality_fn,                       # setup → scalar
    key:          jax.Array,
) -> tuple[MAPElitesArchive, jax.Array]:
    """
    Single MAP-Elites iteration — JIT-compilable, scannable.

    1. Sample an occupied cell proportional to quality (softmax weighting)
    2. Gradient-guided mutation toward a random target cell
    3. Evaluate mutated setup
    4. Update archive if improvement or new cell
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # ── 1. Sample parent cell ─────────────────────────────────────────────
    # Soft-max weighted sampling: better cells explored more often
    q_flat      = archive.quality.ravel()                  # (N_B1·N_B2,)
    is_occupied = jnp.isfinite(q_flat)
    weights     = jnp.where(is_occupied, jax.nn.softmax(q_flat * 0.5), 0.0)
    parent_flat = jax.random.choice(k1, q_flat.shape[0], p=weights)
    parent_i    = parent_flat // N_B2
    parent_j    = parent_flat  % N_B2
    parent_setup = archive.setups[parent_i, parent_j]

    # ── 2. Sample random target cell for mutation direction ───────────────
    target_flat  = jax.random.randint(k2, (), 0, N_B1 * N_B2)
    target_bd    = archive.bd_map.reshape(-1, 2)[target_flat]

    # ── 3. Gradient-guided mutation ───────────────────────────────────────
    child = gradient_guided_mutation(parent_setup, quality_fn, target_bd, k3)

    # ── 4. Evaluate and update ────────────────────────────────────────────
    quality  = quality_fn(child)
    archive  = update_archive(archive, child, quality)

    return archive, quality


def run_map_elites(
    quality_fn,
    init_setups:  np.ndarray,         # (N_init, N_DIM) — seed from MORL Pareto front
    init_key:     jax.Array,
    n_iterations: int = 5000,
    verbose:      bool = True,
) -> MAPElitesArchive:
    """
    Full MAP-Elites run with gradient-guided mutation.

    Typical FS usage:
        1. Run MORL-SB-TRPO → get Pareto front setups
        2. Seed archive with Pareto setups
        3. Run MAP-Elites to fill the behavioral map
        4. Give race engineer the 225-cell grid with quality score + setup per cell
    """
    archive = init_archive()

    # Seed archive with provided setups (e.g., MORL Pareto front)
    for s in init_setups:
        s_jax   = jnp.array(s, dtype=jnp.float32)
        q       = float(quality_fn(s_jax))
        archive = update_archive(archive, s_jax, jnp.array(q))

    keys = jax.random.split(init_key, n_iterations)

    for i, k in enumerate(keys):
        archive, q_child = map_elites_step(archive, quality_fn, k)

        if verbose and i % 500 == 0:
            n_occupied = int(jnp.sum(jnp.isfinite(archive.quality)))
            q_max      = float(jnp.max(jnp.where(jnp.isfinite(archive.quality),
                                                   archive.quality, -jnp.inf)))
            print(f"[MAP-Elites] iter={i:5d} | occupied={n_occupied:3d}/{N_B1*N_B2} "
                  f"| q_max={q_max:.4f} G | q_child={float(q_child):.4f}")

    return archive


def archive_to_engineer_report(archive: MAPElitesArchive) -> dict:
    """
    Convert archive to race engineer deliverable.
    Returns a menu of setup philosophies with behavioral labels.
    """
    from models.vehicle_dynamics import SuspensionSetup, SETUP_NAMES

    occupied_mask = jnp.isfinite(archive.quality)
    i_coords, j_coords = jnp.where(occupied_mask)

    report = []
    for ii, jj in zip(i_coords.tolist(), j_coords.tolist()):
        bd1_val = float(archive.bd_map[ii, jj, 0])
        bd2_val = float(archive.bd_map[ii, jj, 1])

        balance_label = ("UNDERSTEER"  if bd1_val > 0.55 else
                         "OVERSTEER"   if bd1_val < 0.45 else "NEUTRAL")
        stiffness_label = ("STIFF" if bd2_val > 0.65 else
                            "SOFT"  if bd2_val < 0.35 else "MEDIUM")

        setup_vec = archive.setups[ii, jj]
        report.append({
            "paradigm":     f"{stiffness_label} / {balance_label}",
            "quality_G":    float(archive.quality[ii, jj]),
            "bd1_roll_frac": bd1_val,
            "bd2_stiffness": bd2_val,
            "k_f_Nm":       float(setup_vec[0]),
            "k_r_Nm":       float(setup_vec[1]),
            "arb_f":        float(setup_vec[2]),
            "arb_r":        float(setup_vec[3]),
            "setup_vector": setup_vec,
        })

    report.sort(key=lambda x: x["quality_G"], reverse=True)
    return {"n_paradigms": len(report), "setups": report}