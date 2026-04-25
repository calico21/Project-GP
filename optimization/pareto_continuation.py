# optimization/pareto_continuation.py
# Project-GP — Batch 5: Homotopy Pareto Continuation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Traces the Pareto front (grip vs stability) via warm-started homotopy:
#   1. Corner A: Adam 120 steps at w=1.0 (max grip)
#   2. Corner B: Adam 120 steps at w=0.0 (max stability)
#   3. Interior:  Adam 30 steps each, warm-started from previous point
#
# KEY DESIGN: grad_fn is compiled ONCE with w as a JAX scalar argument.
# Previous version created a new Python closure per call → 5× recompilation
# of the full physics gradient (each ~800s). Now: one compilation, reused.
#
# SPEED vs MORL: ~5-8 min vs 45 min (same Pareto coverage, deterministic)
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.vehicle_dynamics import (
    SETUP_LB, SETUP_UB, DEFAULT_SETUP,
    compute_equilibrium_suspension,
)
from optimization.objectives import (
    compute_skidpad_objective,
    compute_step_steer_objective,
)
from config.vehicles.ter26 import vehicle_params as VP
from config.tire_coeffs import tire_coeffs as TC


# ─────────────────────────────────────────────────────────────────────────────
# §1  Setup space — logit parameterisation
# ─────────────────────────────────────────────────────────────────────────────

_LB   = jnp.array(SETUP_LB,   dtype=jnp.float32)
_UB   = jnp.array(SETUP_UB,   dtype=jnp.float32)
_MID  = 0.5 * (_LB + _UB)
_HALF = 0.5 * (_UB - _LB)


def _logit_to_setup(z: jax.Array) -> jax.Array:
    """z ∈ ℝ^28 → s ∈ [LB, UB] via tanh projection."""
    return _MID + _HALF * jnp.tanh(z / (_HALF + 1e-6))


def _setup_to_logit(s: jax.Array) -> jax.Array:
    """s ∈ [LB, UB] → z ∈ ℝ^28."""
    x = (s - _MID) / (_HALF + 1e-6)
    return jnp.arctanh(jnp.clip(x, -0.999, 0.999))


# ─────────────────────────────────────────────────────────────────────────────
# §2  Compiled objective + gradient — ONE compilation, w as JAX argument
# ─────────────────────────────────────────────────────────────────────────────

def _make_compiled_fns(vehicle):
    """
    Returns (loss_fn, grad_fn, eval_fn) — all compiled once.

    Critically: w is passed as a JAX scalar, not captured in a closure.
    This means jax.jit traces with w as an abstract value and the same
    compiled kernel is reused for all weights. Previous version captured
    w as a Python float in a closure → new compilation per call (~800s each).
    """
    @jax.jit
    def eval_fn(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        s    = _logit_to_setup(z)
        z_eq = compute_equilibrium_suspension(s, VP)
        from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
        x0_base = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=15.0)
        x0   = (x0_base
                .at[6:10].set(z_eq)
                .at[28:38].set(jnp.array([85., 85., 85., 85., 80.,
                                           85., 85., 85., 85., 80.])))
        grip, _ = compute_skidpad_objective(vehicle.simulate_step, s, x0)
        stab    = compute_step_steer_objective(vehicle.simulate_step, s, x0)
        return grip, stab

    @jax.jit
    def loss_fn(z: jax.Array, w: jax.Array) -> jax.Array:
        """Scalarised loss: minimise −[w·grip + (1−w)·stab]."""
        grip, stab = eval_fn(z)
        return -(w * grip + (1.0 - w) * stab)

    # Compile gradient once — same kernel for all w values
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    return eval_fn, loss_fn, grad_fn


# ─────────────────────────────────────────────────────────────────────────────
# §3  Single-objective solver — takes pre-compiled fns
# ─────────────────────────────────────────────────────────────────────────────

def _solve(
    loss_fn, grad_fn, eval_fn,
    w:       float,
    z_init:  jax.Array,
    n_steps: int   = 120,
    lr:      float = 0.05,
    verbose: bool  = False,
) -> tuple[jax.Array, float, float]:
    """Adam descent in logit space using pre-compiled grad_fn."""
    w_jax     = jnp.array(w, dtype=jnp.float32)
    schedule  = optax.cosine_decay_schedule(lr, max(n_steps, 1), alpha=0.05)
    tx        = optax.adam(schedule)
    opt_state = tx.init(z_init)
    z         = z_init
    best_z, best_val = z, float('inf')

    for i in range(n_steps):
        g = grad_fn(z, w_jax)
        updates, opt_state = tx.update(g, opt_state, z)
        z   = optax.apply_updates(z, updates)
        val = float(loss_fn(z, w_jax))
        if val < best_val:
            best_val = val
            best_z   = z
        if verbose and i % 20 == 0:
            grip, stab = eval_fn(best_z)
            print(f"    step {i:4d}  w={w:.2f}  grip={float(grip):.4f}G"
                  f"  stab={float(stab):.4f}  F={best_val:.4f}")

    grip_f, stab_f = eval_fn(best_z)
    return best_z, float(grip_f), float(stab_f)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Full Pareto sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_pareto_front(
    vehicle,
    n_points:     int   = 20,
    corner_steps: int   = 120,
    interior_steps: int = 30,
    verbose:      bool  = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warm-started homotopy sweep from w=1 (grip) to w=0 (stability).

    Returns:
        setups  (n_points, 28) float32
        grips   (n_points,)    float64  — skidpad grip [G]
        stabs   (n_points,)    float64  — step-steer score (higher = more stable)
    """
    eval_fn, loss_fn, grad_fn = _make_compiled_fns(vehicle)

    if verbose:
        print("\n[Pareto-IFT] Compiling physics gradient (one-time, ~60s) ...")

    # Trigger compilation by evaluating once
    z_default = _setup_to_logit(DEFAULT_SETUP)
    _ = grad_fn(z_default, jnp.array(0.5, dtype=jnp.float32))

    if verbose:
        print("[Pareto-IFT] Compilation complete. Starting sweep.")
        print(f"  n_points={n_points}  corner_steps={corner_steps}"
              f"  interior_steps={interior_steps}")

    # ── Corner A: w=1.0 ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if verbose:
        print("\n[Pareto-IFT] Corner A: maximising grip (w=1.0) ...")
    z_A, grip_A, stab_A = _solve(
        loss_fn, grad_fn, eval_fn, w=1.0, z_init=z_default,
        n_steps=corner_steps, lr=0.05, verbose=verbose,
    )
    if verbose:
        print(f"  Corner A: grip={grip_A:.4f}G  stab={stab_A:.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

    # ── Corner B: w=0.0 ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if verbose:
        print("\n[Pareto-IFT] Corner B: maximising stability (w=0.0) ...")
    z_B, grip_B, stab_B = _solve(
        loss_fn, grad_fn, eval_fn, w=0.0, z_init=z_default,
        n_steps=corner_steps, lr=0.05, verbose=verbose,
    )
    if verbose:
        print(f"  Corner B: grip={grip_B:.4f}G  stab={stab_B:.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

    # ── Interior — warm-started homotopy ──────────────────────────────────────
    weights = np.linspace(1.0, 0.0, n_points)
    setups  = np.zeros((n_points, len(DEFAULT_SETUP)), dtype=np.float32)
    grips   = np.zeros(n_points)
    stabs   = np.zeros(n_points)

    setups[0]  = np.array(_logit_to_setup(z_A))
    grips[0]   = grip_A
    stabs[0]   = stab_A

    setups[-1] = np.array(_logit_to_setup(z_B))
    grips[-1]  = grip_B
    stabs[-1]  = stab_B

    t_sweep = time.perf_counter()
    z_curr  = z_A

    for i in range(1, n_points - 1):
        w_next = float(weights[i])
        z_corr, g_corr, s_corr = _solve(
            loss_fn, grad_fn, eval_fn, w=w_next, z_init=z_curr,
            n_steps=interior_steps, lr=0.05, verbose=False,
        )
        setups[i] = np.array(_logit_to_setup(z_corr))
        grips[i]  = g_corr
        stabs[i]  = s_corr
        z_curr    = z_corr
        if verbose:
            print(f"  [w={w_next:.3f}]  grip={g_corr:.4f}G  stab={s_corr:.4f}")

    if verbose:
        print(f"\n[Pareto-IFT] Sweep complete in {time.perf_counter()-t_sweep:.1f}s")
        print(f"  Grip range:  [{grips.min():.4f}, {grips.max():.4f}] G")
        print(f"  Stab range:  [{stabs.min():.4f}, {stabs.max():.4f}]")

    return setups, grips, stabs


# ─────────────────────────────────────────────────────────────────────────────
# §5  Non-domination filter
# ─────────────────────────────────────────────────────────────────────────────

def _get_non_dominated(grips: np.ndarray, stabs: np.ndarray) -> np.ndarray:
    """
    Return indices of non-dominated points.
    Maximise both grips (higher = better) and stabs (higher = more stable).

    Note: stabs from compute_step_steer_objective are negative overshoot
    penalties — higher (less negative) = better stability. We maximise
    directly without negation.
    """
    n = len(grips)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i: at least as good on both, strictly better on one
            if grips[j] >= grips[i] and stabs[j] >= stabs[i]:
                if grips[j] > grips[i] or stabs[j] > stabs[i]:
                    dominated[i] = True
                    break
    return np.where(~dominated)[0]


# ─────────────────────────────────────────────────────────────────────────────
# §6  Sensitivity report
# ─────────────────────────────────────────────────────────────────────────────

def _print_sensitivity(vehicle, p_setups, p_grips, p_stabs):
    from models.vehicle_dynamics import SETUP_NAMES
    eval_fn, _, grad_fn = _make_compiled_fns(vehicle)
    w_jax = jnp.array(1.0, dtype=jnp.float32)
    print(f"\n[Pareto-IFT] Setup sensitivity (∂grip/∂param) at Pareto front:")
    header = f"  {'Setup':>5}  {'Grip':>5}  {'Stab':>5}"
    for name in SETUP_NAMES[:8]:
        header += f"  {name[:7]:>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, (s, g, st) in enumerate(zip(p_setups, p_grips, p_stabs)):
        z = _setup_to_logit(jnp.array(s))
        try:
            g_vec = np.array(grad_fn(z, w_jax))
            row = f"  {i+1:>5}  {g:>5.3f}  {st:>5.2f}"
            for gi in g_vec[:8]:
                row += f"  {-gi:>+7.2f}"   # negate: grad of loss → grad of grip
            print(row)
        except Exception:
            print(f"  {i+1:>5}  {g:>5.3f}  {st:>5.2f}  (grad unavailable)")


# ─────────────────────────────────────────────────────────────────────────────
# §7  Drop-in class
# ─────────────────────────────────────────────────────────────────────────────

class ParetoOptimizer:
    """
    Drop-in replacement for MORL_SB_TRPO_Optimizer.
    Output tuple matches: (setups, grips, stabs_neg, gens)
    where stabs_neg = -stabs to match MORL archive convention.
    """

    def __init__(self, n_points: int = 20, verbose: bool = True,
                corner_steps: int = 120, interior_steps: int = 30):
        self.n_points       = n_points
        self.corner_steps   = corner_steps
        self.interior_steps = interior_steps
        self.verbose        = verbose
        self._vehicle       = None

    def _get_vehicle(self):
        if self._vehicle is None:
            from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
            self._vehicle = DifferentiableMultiBodyVehicle(VP, TC)
        return self._vehicle

    def run(
        self,
        n_points:   int | None = None,
        iterations: int        = 0,    # ignored, kept for API compat
        wandb_run              = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_points is None:
            n_points = self.n_points

        veh = self._get_vehicle()
        t0  = time.perf_counter()

        setups, grips, stabs = sweep_pareto_front(
            veh, n_points=n_points,
            corner_steps=self.corner_steps,
            interior_steps=self.interior_steps,
            verbose=self.verbose,
        )

        elapsed = time.perf_counter() - t0

        # Non-domination: maximize both grip and stab (higher stab = more stable)
        pareto_idx = _get_non_dominated(grips, stabs)
        p_setups   = setups[pareto_idx]
        p_grips    = grips[pareto_idx]
        p_stabs    = stabs[pareto_idx]
        p_gens     = pareto_idx.astype(np.int64)

        # Negate stabs for MORL convention (MORL stores negative stab as cost)
        p_stabs_neg = -p_stabs

        print(f"\n[Pareto-IFT] Done in {elapsed:.1f}s  "
              f"({n_points} candidates → {len(p_grips)} non-dominated)")
        print(f"  Best grip:       {p_grips.max():.4f} G")
        print(f"  Stability range: {p_stabs.min():.3f}–{p_stabs.max():.3f}"
              f"  (higher = more stable)")

        _print_sensitivity(veh, p_setups, p_grips, p_stabs)

        if wandb_run is not None:
            wandb_run.summary["Max_Grip_IFT"]     = float(p_grips.max())
            wandb_run.summary["Pareto_Count_IFT"] = int(len(p_grips))
            wandb_run.summary["IFT_Wall_Time_s"]  = elapsed

        return p_setups, p_grips, p_stabs_neg, p_gens


# ─────────────────────────────────────────────────────────────────────────────
# §8  Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points",      type=int, default=20)
    parser.add_argument("--corner-steps",  type=int, default=120)
    parser.add_argument("--interior-steps",type=int, default=30)
    parser.add_argument("--out", type=str, default="models/pareto_ift.npz")
    args = parser.parse_args()

    opt = ParetoOptimizer(n_points=args.n_points, verbose=True)
    setups, grips, stabs_neg, gens = opt.run()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, setups=setups, grips=grips, stabs=-stabs_neg)
    print(f"\n[Pareto-IFT] Saved → {args.out}")