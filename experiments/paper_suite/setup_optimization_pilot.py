"""Bounded, reproducible two-stiffness setup-optimization pilot.

This is intentionally a feasibility experiment, not an IFT or global-optimum
claim.  Its objective and trajectory are exactly those used by the preceding
conditioning scan.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import SuiteConfig, finite_metrics, jsonable, metadata, manifest
from .setup_conditioning import _components, _trajectory
from .studies import _case, _controls

FRACTION_LOWER = 0.85
FRACTION_UPPER = 1.15
STEP_SIZE = 0.01
N_STEPS = 10


def projected_fraction_step(fractions, gradient_fraction, step_size=STEP_SIZE):
    """One deterministic projected gradient step in nominal-stiffness units."""
    direction = gradient_fraction / jnp.maximum(jnp.linalg.norm(gradient_fraction), 1e-20)
    return jnp.clip(fractions - step_size * direction, FRACTION_LOWER, FRACTION_UPPER)


def gradient_comparison(ad, fd):
    metric = finite_metrics(ad, fd)
    metric["relative_error"] = metric.pop("relative")
    return metric


def _write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def run_pilot(cfg: SuiteConfig):
    vehicle, state0, _, setup, _ = _case(cfg)
    controls = _controls(10)
    nominal = setup[jnp.array([0, 1])]

    def components(theta):
        trial = setup.at[0].set(theta[0]).at[1].set(theta[1])
        return _components(_trajectory(vehicle, state0, trial, controls, cfg.dt))

    reference = components(nominal)
    component_names = tuple(reference)
    scales = {name: jnp.maximum(reference[name], 1e-12) for name in component_names}

    def objective_fraction(fractions):
        values = components(nominal * fractions)
        return sum(values[name] / scales[name] for name in component_names)

    value_and_grad = jax.value_and_grad(objective_fraction)

    def fd_gradient(fractions):
        # 0.1% of nominal stiffness is large enough to avoid subtraction noise
        # while remaining inside the stated local box at every validation point.
        epsilon = 1e-3
        result = []
        for i in range(2):
            offset = jnp.zeros(2).at[i].set(epsilon)
            result.append((objective_fraction(fractions + offset) - objective_fraction(fractions - offset)) / (2 * epsilon))
        return jnp.asarray(result)

    # Explicit preflight points.  They must agree with the conditioning grid.
    preflight_fractions = [(.0, .0), (.15, .0), (-.15, .0), (.0, .15), (.0, -.15),
                           (-.15, -.15), (-.15, .15), (.15, -.15), (.15, .15)]
    preflight = []
    for df, dr in preflight_fractions:
        fractions = jnp.array([1 + df, 1 + dr])
        preflight.append({"k_f_fraction_offset": df, "k_r_fraction_offset": dr,
                          "k_f": float((nominal * fractions)[0]), "k_r": float((nominal * fractions)[1]),
                          "objective": float(objective_fraction(fractions))})
    preflight_values = np.asarray([row["objective"] for row in preflight])
    if not np.all(np.isfinite(preflight_values)):
        raise RuntimeError("Preflight objective contains a non-finite value; optimization not started.")

    fractions = jnp.ones(2)
    rows, trajectory, validation = [], [], []
    for iteration in range(N_STEPS):
        value, grad_fraction = value_and_grad(fractions)
        theta = nominal * fractions
        finite = bool(jnp.isfinite(value) and jnp.all(jnp.isfinite(grad_fraction)))
        if not finite:
            raise RuntimeError(f"Non-finite objective/gradient at iteration {iteration}; optimization stopped.")
        next_fractions = projected_fraction_step(fractions, grad_fraction)
        next_value = objective_fraction(next_fractions)
        bounds_active = bool(jnp.any((next_fractions == FRACTION_LOWER) | (next_fractions == FRACTION_UPPER)))
        row = {"iteration": iteration + 1, "objective_before": float(value), "objective_after": float(next_value),
               "k_f_before": float(theta[0]), "k_r_before": float(theta[1]),
               "k_f_after": float((nominal * next_fractions)[0]), "k_r_after": float((nominal * next_fractions)[1]),
               "fraction_k_f_before": float(fractions[0]), "fraction_k_r_before": float(fractions[1]),
               "ad_grad_fraction_k_f": float(grad_fraction[0]), "ad_grad_fraction_k_r": float(grad_fraction[1]),
               "ad_gradient_norm": float(jnp.linalg.norm(grad_fraction)), "bounds_active": bounds_active,
               "solver_status": "finite_production_glrk", "simulation_finite": True}
        rows.append(row); trajectory.append({"iteration": iteration, "objective": float(value), "k_f": float(theta[0]), "k_r": float(theta[1])})
        fractions = next_fractions
        if iteration + 1 == 5:
            ad_mid = jax.grad(objective_fraction)(fractions); fd_mid = fd_gradient(fractions)
            validation.append({"point": "midpoint_after_step_5", "fractions": fractions, "ad": ad_mid, "fd": fd_mid, **gradient_comparison(ad_mid, fd_mid)})

    final_value, final_ad = value_and_grad(fractions)
    initial_ad = jax.grad(objective_fraction)(jnp.ones(2))
    initial_fd, final_fd = fd_gradient(jnp.ones(2)), fd_gradient(fractions)
    validation = [{"point": "initial", "fractions": jnp.ones(2), "ad": initial_ad, "fd": initial_fd, **gradient_comparison(initial_ad, initial_fd)}, *validation,
                  {"point": "final_after_step_10", "fractions": fractions, "ad": final_ad, "fd": final_fd, **gradient_comparison(final_ad, final_fd)}]
    trajectory.append({"iteration": N_STEPS, "objective": float(final_value), "k_f": float((nominal * fractions)[0]), "k_r": float((nominal * fractions)[1])})

    base = Path(cfg.output); out = base / "setup_optimization"; out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "setup_optimization_pilot.csv", rows)
    _write_csv(out / "setup_objective_surface.csv", preflight)
    reduction = 100 * (trajectory[0]["objective"] - trajectory[-1]["objective"]) / trajectory[0]["objective"]
    payload = metadata(cfg, "setup_optimization_pilot") | {
        "status": "completed", "objective_definition": "sum of roll, front/rear equilibrium-centred heave, and suspension-velocity mean-square terms, each divided by its nominal 10-step value", "horizon_steps": 10,
        "parameters": ["k_f", "k_r"], "nominal": nominal, "bounds": {"fraction": [FRACTION_LOWER, FRACTION_UPPER], "physical": [nominal * FRACTION_LOWER, nominal * FRACTION_UPPER]},
        "optimizer": {"name": "projected normalized gradient descent", "steps": N_STEPS, "step_size_fraction": STEP_SIZE},
        "preflight": preflight, "iterations": rows, "trajectory": trajectory, "gradient_validation": validation,
        "initial_objective": trajectory[0]["objective"], "final_objective": trajectory[-1]["objective"], "objective_reduction_percent": reduction,
        "initial_parameters": nominal, "final_parameters": nominal * fractions,
        "all_steps_finite": True, "any_bound_active": any(row["bounds_active"] for row in rows),
        "ift_comparison": "unavailable in this pilot", "limitations": ["Single deterministic initial condition and 10-step manoeuvre.", "AD is compared only with central FD at initial, midpoint, and final points.", "No IFT comparison, competing optimizer, random restarts, or global-optimality claim."],
    }
    (out / "setup_optimization_pilot.json").write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    tex = ["\\begin{tabular}{rrrrr}", "\\toprule", "Step & Objective & $k_f$ & $k_r$ & $\\|\\nabla J\\|$ \\\\ \\midrule"]
    tex.extend(f"{row['iteration']} & {row['objective_after']:.6g} & {row['k_f_after']:.6g} & {row['k_r_after']:.6g} & {row['ad_gradient_norm']:.4g} \\\\" for row in rows)
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (out / "setup_optimization_summary.tex").write_text("\n".join(tex))
    table = base / "tables" / "table_setup_optimization.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(tex))
    report = f"""# Bounded setup-optimization pilot

## Observed result

The normalised 10-step objective changed from {trajectory[0]['objective']:.6g} to {trajectory[-1]['objective']:.6g} ({reduction:.3f}%). The final stiffnesses were k_f={float((nominal * fractions)[0]):.6g} N/m and k_r={float((nominal * fractions)[1]):.6g} N/m.

## Numerical validation

All production simulations and AD gradients were finite. Central finite differences were evaluated at the initial point, after step 5, and after step 10. IFT comparison: unavailable in this pilot.

## Interpretation

This bounded deterministic pilot tests whether the conditioned state-based objective admits a finite descent trajectory. It does not establish global optimality, model superiority, or a general setup-optimization result.

## Unsupported conclusions and next steps

No conclusion about IFT agreement, competing optimizers, 200-step behaviour, or repeatability over initial setups is supported. Next measure IFT and FD agreement over additional operating points before extending the horizon.
"""
    (out / "setup_optimization_pilot.md").write_text(report)
    figdir = base / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    xs = [point["iteration"] for point in trajectory]
    fig, ax = plt.subplots(figsize=(5.0, 3.2)); ax.plot(xs, [point["objective"] for point in trajectory], marker="o"); ax.set(xlabel="optimization step", ylabel="normalised objective", title="Bounded setup-optimization pilot"); ax.grid(alpha=.3); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(figdir / f"fig_setup_optimization_objective.{ext}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 3.2)); ax.plot(xs, [point["k_f"] for point in trajectory], marker="o", label="k_f"); ax.plot(xs, [point["k_r"] for point in trajectory], marker="o", label="k_r"); ax.set(xlabel="optimization step", ylabel="spring rate [N/m]", title="Bounded stiffness trajectory"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(figdir / f"fig_setup_optimization_parameters.{ext}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 3.2)); labels = [item["point"] for item in validation]
    positions = np.arange(len(labels))
    for component, label in enumerate(("k_f fraction", "k_r fraction")):
        ax.plot(positions, [float(item["ad"][component]) for item in validation], marker="o", label=f"AD {label}")
        ax.plot(positions, [float(item["fd"][component]) for item in validation], marker="x", linestyle="--", label=f"FD {label}")
    ax.set(xticks=positions, xticklabels=labels, ylabel="dJ/d(fraction)", title="AD versus FD setup gradients"); ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(figdir / f"fig_setup_optimization_gradients.{ext}", dpi=300)
    plt.close(fig)
    manifest(cfg)
    return out / "setup_optimization_pilot.json"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output", default=SuiteConfig().output)
    args = parser.parse_args(); print(run_pilot(SuiteConfig(seed=args.seed, output=args.output)))
