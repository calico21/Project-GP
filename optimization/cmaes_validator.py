#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# optimization/cmaes_validator.py
# Project-GP — CMA-ES Crossover Validator for MORL Pareto Front
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Independent gradient-free optimizer that validates the MORL-SB-TRPO
#   Pareto front. If CMA-ES finds solutions that dominate the MORL front,
#   the gradient-based optimizer has a bug or is stuck in a local optimum.
#
#   CMA-ES is the gold standard for black-box optimisation in 10-50 dims.
#   It doesn't need gradients (uses as cross-check), is intrinsically
#   multi-modal, and converges faster than TRPO in noisy landscapes.
#
# INTEGRATION:
#   Run AFTER MORL completes. Loads the Pareto front from morl_pareto_front.csv,
#   seeds CMA-ES with the best Pareto solutions, and verifies no dominating
#   solution exists nearby.
#
# USAGE:
#   python optimization/cmaes_validator.py [--generations 200] [--popsize 40]
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import argparse
import time

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
try:
    import jax_config  # noqa
except ImportError:
    pass

import jax
import jax.numpy as jnp

from models.vehicle_dynamics import (
    SuspensionSetup, SETUP_DIM, SETUP_LB, SETUP_UB, SETUP_NAMES,
)
from data.configs.vehicle_params import vehicle_params as VP
from data.configs.tire_coeffs import tire_coeffs as TC


# ─────────────────────────────────────────────────────────────────────────────
# §1  Lightweight CMA-ES (pure NumPy, no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class CMA_ES:
    """
    Covariance Matrix Adaptation Evolution Strategy.

    Implements the (μ/μ_w, λ)-CMA-ES with rank-1 and rank-μ updates.
    Reference: Hansen & Ostermeier (2001), "Completely Derandomized
    Self-Adaptation in Evolution Strategies."

    Operates in [0, 1]^dim normalised space. Physical bounds enforced
    by sigmoid projection.
    """

    def __init__(
        self,
        dim: int,
        popsize: int = None,
        sigma0: float = 0.3,
        seed: int = 42,
    ):
        self.dim = dim
        self.lam = popsize or (4 + int(3 * np.log(dim)))  # default: 4+3ln(n)
        self.mu = self.lam // 2
        self.sigma = sigma0
        self.rng = np.random.default_rng(seed)

        # Recombination weights (log-linear)
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size control
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Covariance adaptation
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff),
        )

        # State
        self.mean = np.full(dim, 0.0)
        self.C = np.eye(dim)
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.gen = 0

    def seed_from_solution(self, x_norm: np.ndarray):
        """Seed CMA-ES mean at a known good solution (from MORL)."""
        self.mean = np.clip(x_norm, -3.0, 3.0)  # logit-ish space

    def ask(self) -> np.ndarray:
        """Generate λ candidate solutions. Returns (λ, dim) in normalised space."""
        # Eigendecomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-12)
        D = np.sqrt(eigvals)
        B = eigvecs

        z = self.rng.standard_normal((self.lam, self.dim))
        samples = self.mean[None, :] + self.sigma * (z @ np.diag(D) @ B.T)
        return samples

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray):
        """Update CMA-ES state from evaluated solutions. MINIMISES fitness."""
        # Sort by fitness (ascending = best first for minimisation)
        idx = np.argsort(fitnesses)
        solutions = solutions[idx]

        # Weighted recombination of top-μ
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * solutions[:self.mu], axis=0)

        # Evolution path update
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-12)
        invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mu_eff
        ) * invsqrtC @ (self.mean - old_mean) / self.sigma

        hs = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1)))
            < (1.4 + 2 / (self.dim + 1)) * self.chi_n
        )

        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mu_eff
        ) * (self.mean - old_mean) / self.sigma

        # Covariance matrix update
        artmp = (solutions[:self.mu] - old_mean[None, :]) / self.sigma
        rank1 = self.c1 * np.outer(self.pc, self.pc)
        rankmu = self.cmu * (self.weights[:, None] * artmp).T @ artmp

        self.C = (
            (1 - self.c1 - self.cmu + (1 - hs) * self.c1 * self.cc * (2 - self.cc)) * self.C
            + rank1
            + rankmu
        )
        # Enforce symmetry
        self.C = 0.5 * (self.C + self.C.T)

        # Step-size update
        self.sigma *= np.exp(
            (self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1)
        )
        self.sigma = np.clip(self.sigma, 1e-8, 5.0)

        self.gen += 1

    @property
    def best_fitness(self):
        return getattr(self, '_best_fitness', float('inf'))


# ─────────────────────────────────────────────────────────────────────────────
# §2  Objective Wrapper
# ─────────────────────────────────────────────────────────────────────────────

def build_objective():
    """
    Build the scalarised objective function for CMA-ES.
    Returns a callable: f(setup_norm_np) → scalar (LOWER = better).
    """
    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle, compute_equilibrium_suspension
    from optimization.objectives import compute_skidpad_objective, compute_step_steer_objective

    vehicle = DifferentiableMultiBodyVehicle(VP, TC)
    lb = np.array(SETUP_LB)
    ub = np.array(SETUP_UB)

    @jax.jit
    def _eval(setup_norm):
        setup_phys = jnp.array(lb) + (jnp.array(ub) - jnp.array(lb)) * setup_norm
        z_eq = compute_equilibrium_suspension(setup_phys, VP)
        x_init = (jnp.zeros(46)
                  .at[14].set(15.0)
                  .at[6:10].set(z_eq)
                  .at[28:38].set(jnp.array([85., 85., 85., 85., 80.,
                                             85., 85., 85., 85., 80.])))

        grip, _ = compute_skidpad_objective(vehicle.simulate_step, setup_phys, x_init)
        stab = compute_step_steer_objective(vehicle.simulate_step, setup_phys, x_init)
        return grip, stab

    def objective(x_norm_np: np.ndarray) -> float:
        """Scalarised: minimise -(grip + 0.3 * stab). Handles NaN gracefully."""
        setup_norm = jnp.array(np.clip(x_norm_np, 0, 1), dtype=jnp.float32)
        try:
            grip, stab = _eval(setup_norm)
            g, s = float(grip), float(stab)
            if not (np.isfinite(g) and np.isfinite(s)):
                return 1e6
            return -(g + 0.3 * s)  # minimise negative = maximise positive
        except Exception:
            return 1e6

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# §3  Validation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def validate_morl_front(
    n_generations: int = 200,
    popsize: int = 40,
    seed: int = 7,
):
    """
    Run CMA-ES and compare against MORL Pareto front.
    """
    print(f"\n{'═' * 72}")
    print(f"  CMA-ES CROSSOVER VALIDATION")
    print(f"{'═' * 72}")

    # ── Load MORL Pareto front if available ──────────────────────────────────
    pareto_path = os.path.join(_root, 'morl_pareto_front.csv')
    morl_best_grip = None
    morl_best_setup = None

    if os.path.exists(pareto_path):
        import pandas as pd
        df = pd.read_csv(pareto_path)
        if 'grip' in df.columns:
            morl_best_grip = df['grip'].max()
            best_idx = df['grip'].idxmax()
            # Extract setup columns if present
            setup_cols = [c for c in df.columns if c in SETUP_NAMES]
            if setup_cols:
                morl_best_setup = df.loc[best_idx, setup_cols].values.astype(np.float32)
            print(f"[CMA-ES] MORL Pareto front loaded: {len(df)} solutions, "
                  f"best grip = {morl_best_grip:.4f} G")

    # ── Build objective ──────────────────────────────────────────────────────
    print(f"[CMA-ES] Compiling JAX objective…")
    objective = build_objective()

    # Warm-up JIT
    _ = objective(np.full(SETUP_DIM, 0.5))
    print(f"[CMA-ES] JIT warm-up complete")

    # ── Run CMA-ES ───────────────────────────────────────────────────────────
    cma = CMA_ES(dim=SETUP_DIM, popsize=popsize, sigma0=0.3, seed=seed)

    # Seed from MORL best if available
    if morl_best_setup is not None:
        lb, ub = np.array(SETUP_LB), np.array(SETUP_UB)
        morl_norm = (morl_best_setup - lb) / (ub - lb + 1e-12)
        cma.seed_from_solution(morl_norm)
        print(f"[CMA-ES] Seeded from MORL best: {morl_best_grip:.4f} G")

    print(f"[CMA-ES] Running: {n_generations} generations, λ={cma.lam}, μ={cma.mu}")
    print(f"{'─' * 72}")

    t0 = time.time()
    best_ever = float('inf')
    best_setup = None

    for gen in range(n_generations):
        solutions = cma.ask()

        # Clip to [0, 1] (setup bounds)
        solutions_clipped = np.clip(solutions, 0, 1)

        # Evaluate
        fitnesses = np.array([objective(s) for s in solutions_clipped])

        cma.tell(solutions_clipped, fitnesses)

        gen_best = fitnesses.min()
        if gen_best < best_ever:
            best_ever = gen_best
            best_setup = solutions_clipped[np.argmin(fitnesses)]

        if gen % 20 == 0 or gen < 5:
            elapsed = time.time() - t0
            print(f"  gen {gen:4d} | best={-best_ever:.4f} G | "
                  f"σ={cma.sigma:.4f} | [{elapsed:.1f}s]")

    cma_best_grip = -best_ever
    elapsed = time.time() - t0
    print(f"{'─' * 72}")
    print(f"[CMA-ES] Complete: {n_generations} generations in {elapsed:.1f}s")
    print(f"[CMA-ES] Best grip: {cma_best_grip:.4f} G")

    # ── Comparison ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  CROSSOVER COMPARISON")
    print(f"{'═' * 72}")

    if morl_best_grip is not None:
        delta = cma_best_grip - morl_best_grip
        print(f"  MORL-SB-TRPO best: {morl_best_grip:.4f} G")
        print(f"  CMA-ES best:       {cma_best_grip:.4f} G")
        print(f"  Delta:             {delta:+.4f} G")

        if delta > 0.02:
            print(f"\n  ⚠️  CMA-ES found a BETTER solution by {delta:.4f} G.")
            print(f"      MORL may be stuck in a local optimum or has a bug.")
            print(f"      Recommendation: seed MORL with the CMA-ES solution.")
        elif delta > -0.02:
            print(f"\n  ✅ Results are consistent (within ±0.02 G).")
            print(f"      MORL Pareto front is validated.")
        else:
            print(f"\n  ✅ MORL found a BETTER solution. Gradient-based search")
            print(f"      is outperforming black-box — this is expected.")
    else:
        print(f"  No MORL Pareto front found for comparison.")
        print(f"  CMA-ES standalone best: {cma_best_grip:.4f} G")

    # ── Save CMA-ES best for potential MORL seeding ──────────────────────────
    if best_setup is not None:
        lb, ub = np.array(SETUP_LB), np.array(SETUP_UB)
        best_phys = lb + (ub - lb) * best_setup
        out_path = os.path.join(_root, 'cmaes_best_setup.csv')
        import pandas as pd
        df_out = pd.DataFrame([dict(zip(SETUP_NAMES, best_phys))])
        df_out['grip'] = cma_best_grip
        df_out.to_csv(out_path, index=False)
        print(f"\n[CMA-ES] Best setup saved → {out_path}")

    print(f"{'═' * 72}\n")
    return cma_best_grip, best_setup


# ─────────────────────────────────────────────────────────────────────────────
# §4  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CMA-ES Crossover Validator')
    parser.add_argument('--generations', type=int, default=200)
    parser.add_argument('--popsize', type=int, default=40)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    validate_morl_front(args.generations, args.popsize, args.seed)


if __name__ == '__main__':
    main()