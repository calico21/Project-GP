"""Empirical conditioning scan for candidate suspension setup objectives."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.vehicle_dynamics import SETUP_LB, SETUP_UB
from .common import SuiteConfig, jsonable, write_result
from .studies import _case, _controls


def _trajectory(vehicle, x, setup, controls, dt):
    values = []
    for control in controls:
        x = vehicle.simulate_step(x, control, setup, dt=dt, n_substeps=1)
        values.append(x)
    return jnp.stack(values)


def _components(traj):
    # State layout is q[0:14], v[14:28]; q[6:10] are corner suspension z.
    return {
        "roll_rms_rad": jnp.mean(traj[:, 3] ** 2),
        "front_heave_rms_m": jnp.mean(((traj[:, 6] + traj[:, 7]) / 2 - 0.0128) ** 2),
        "rear_heave_rms_m": jnp.mean(((traj[:, 8] + traj[:, 9]) / 2 - 0.0142) ** 2),
        "suspension_velocity_rms_mps": jnp.mean(traj[:, 20:24] ** 2),
    }


def setup_conditioning_study(cfg: SuiteConfig):
    vehicle, x, _, setup, _ = _case(cfg)
    controls = _controls(10)
    indexes = jnp.array([0, 1])
    theta0 = setup[indexes]
    names = list(_components(_trajectory(vehicle, x, setup, controls, cfg.dt)))

    def values(theta):
        trial = setup.at[indexes].set(theta)
        return _components(_trajectory(vehicle, x, trial, controls, cfg.dt))

    base_values = values(theta0)
    # Normalised sum permits a sensitivity comparison without units dominating.
    scales = {key: jnp.maximum(value, 1e-12) for key, value in base_values.items()}
    def combined(theta):
        out = values(theta)
        return sum(out[key] / scales[key] for key in names)

    rows = []
    for key in names + ["combined_normalized"]:
        fn = combined if key == "combined_normalized" else lambda theta, k=key: values(theta)[k]
        ad = jax.grad(fn)(theta0)
        eps = jnp.maximum(jnp.abs(theta0) * 1e-3, 1.0)
        fd = (fn(theta0 + eps) - fn(theta0 - eps)) / (2 * eps)
        rows.append({"objective_component": key, "base_value": float(fn(theta0)), "ad_grad_k_f": float(ad[0]), "ad_grad_k_r": float(ad[1]), "ad_gradient_norm": float(jnp.linalg.norm(ad)), "fd_grad_k_f": float(fd[0]), "fd_grad_k_r": float(fd[1]), "fd_gradient_norm": float(jnp.linalg.norm(fd))})

    fractions = (-.15, -.10, -.05, 0., .05, .10, .15)
    surface = []
    for ff in fractions:
        for fr in fractions:
            theta = jnp.clip(theta0 * jnp.array([1 + ff, 1 + fr]), SETUP_LB[indexes], SETUP_UB[indexes])
            out = values(theta)
            surface.append({"k_f": float(theta[0]), "k_r": float(theta[1]), "k_f_fraction": ff, "k_r_fraction": fr, **{key: float(value) for key, value in out.items()}, "combined_normalized": float(sum(out[key] / scales[key] for key in names))})

    base = Path(cfg.output); out = base / "setup_conditioning"; out.mkdir(parents=True, exist_ok=True)
    (out / "objective_surface.json").write_text(json.dumps(jsonable(surface), indent=2))
    # The combined surface answers whether the initial objective has usable local variation.
    grid = np.asarray([r["combined_normalized"] for r in surface]).reshape(7, 7)
    fig, ax = plt.subplots(figsize=(5.0, 3.8)); im = ax.contourf(np.array(fractions) * 100, np.array(fractions) * 100, grid, levels=16)
    fig.colorbar(im, ax=ax, label="normalised objective"); ax.set(xlabel="k_f perturbation [%]", ylabel="k_r perturbation [%]", title="Setup-objective conditioning")
    fig.tight_layout()
    figdir = base / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(figdir / f"fig_objective_sensitivity.{ext}", dpi=300)
    plt.close(fig)
    tex = ["\\begin{tabular}{lrrr}", "\\toprule", "Component & Value & $\\|\\nabla_{AD}\\|$ & $\\|\\nabla_{FD}\\|$ \\\\ \\midrule"]
    tex += [f"{r['objective_component'].replace('_', ' ')} & {r['base_value']:.4g} & {r['ad_gradient_norm']:.4g} & {r['fd_gradient_norm']:.4g} \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    table = base / "tables" / "table_objective_sensitivity.tex"; table.parent.mkdir(parents=True, exist_ok=True); table.write_text("\n".join(tex))
    variation = float((grid.max() - grid.min()) / max(abs(grid[3, 3]), 1e-12))
    return write_result(cfg, "setup_conditioning", {"horizon_steps": 10, "parameters": ["k_f", "k_r"], "objective_surface_relative_range": variation, "selection_rule": "Choose a candidate only if both AD and FD give a finite non-negligible gradient and the grid shows material variation.", "surface": "setup_conditioning/objective_surface.json"}, status="completed", rows=rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "standard", "paper"), default="smoke"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output", default=SuiteConfig().output)
    args = parser.parse_args(); setup_conditioning_study(SuiteConfig(args.mode, args.seed, args.output))
