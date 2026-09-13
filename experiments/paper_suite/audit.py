"""Evidence audit for paper-suite result files.

This module deliberately analyses only persisted measurements.  It does not
promote a completed command to publication-quality evidence.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import SuiteConfig, jsonable, metadata


def _rows(path: Path):
    csv_path = path / f"{path.name}.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def _baseline_audit(base: Path):
    rows = _rows(base / "baseline")
    groups = defaultdict(list)
    for row in rows:
        try:
            groups[(row["model"], row["manoeuvre"], int(row["horizon_steps"]))].append(float(row["rmse"]))
        except (KeyError, ValueError):
            continue
    summary = []
    for (model, manoeuvre, horizon), values in sorted(groups.items()):
        values = np.asarray(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append({"model": model, "manoeuvre": manoeuvre, "horizon_steps": horizon,
                        "seed_count": len(values), "rmse_mean": mean, "rmse_std": std,
                        "coefficient_of_variation": std / mean if mean else None})
    assessment = {
        "status": "weak",
        "scientific_quality": "weak",
        "reason": "Three seeds and 200-step rollouts are real, but the persisted test set consists of four deterministic trajectories, training histories are only final losses, and no checkpoints remain for independent OOD evaluation.",
        "recommended_action": "Use as protocol/pilot evidence only. Persist checkpoints and epoch-wise losses, then evaluate on independent randomized test trajectories before a main-table claim.",
        "main_table_suitability": "no",
        "supplementary_suitability": "yes, labelled pilot",
        "protocol_only": "yes",
    }
    (base / "baseline" / "baseline_audit.json").write_text(json.dumps(jsonable({"assessment": assessment, "summary": summary}), indent=2))
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Manoeuvre & Horizon & Mean RMSE & Std & CV \\\\ \\midrule"]
    lines.extend(f"{r['model']} & {r['manoeuvre'].replace('_', ' ')} & {r['horizon_steps']} & {r['rmse_mean']:.5g} & {r['rmse_std']:.3g} & {r['coefficient_of_variation']:.3g} \\\\" for r in summary)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (base / "tables" / "results_audit_baseline.tex").parent.mkdir(parents=True, exist_ok=True)
    (base / "tables" / "results_audit_baseline.tex").write_text("\n".join(lines))
    return assessment, summary


def run_audit(cfg: SuiteConfig):
    base = Path(cfg.output)
    baseline, baseline_summary = _baseline_audit(base)
    records = {
        "baseline": baseline,
        "energy": {"status": "complete", "scientific_quality": "moderate", "reason": "Controlled pH bookkeeping closes; vehicle traces are explicitly observational.", "recommended_action": "Use toy balance as supporting numerical evidence; do not claim vehicle global passivity."},
        "compute": {"status": "weak", "scientific_quality": "weak", "reason": "Only two warm samples; compilation is separated but repetitions are insufficient for a stable p95.", "recommended_action": "Repeat warm timings (>=30 samples) on documented hardware."},
        "setup_optimization": {"status": "complete", "scientific_quality": "moderate", "reason": "The bounded 10-step k_f/k_r pilot reduced the conditioned objective by 2.97% with finite states and AD/FD agreement at three points, but uses one deterministic initial condition and no IFT comparison.", "recommended_action": "Treat as Tier C feasibility evidence; add IFT, additional starts, and a 200-step validation before claiming optimization utility."},
        "setup_conditioning": {"status": "complete", "scientific_quality": "moderate", "reason": "A 10-step, ±15% k_f/k_r grid shows a 12.8% range for the normalised state-based candidate objective; AD and central-FD sensitivities are finite.", "recommended_action": "Use the normalised suspension-velocity/heave candidate in additional bounded pilots; do not imply IFT validation until it is measured."},
        "extrapolation": {"status": "partial", "scientific_quality": "weak", "reason": "Only the production map was checked for finiteness; learned baseline checkpoints were not persisted.", "recommended_action": "Persist matched baseline checkpoints and measure interpolation/OOD RMSE."},
        "ablations": {"status": "unavailable", "scientific_quality": "weak", "reason": "No matched trained ablation variants exist.", "recommended_action": "Expose and train only clean A0/A1/A2/A9 variants with matched protocol."},
        "solver": {"status": "unavailable", "scientific_quality": "weak", "reason": "No solver result artifact is present in this output directory.", "recommended_action": "Re-run the already implemented solver study when compute resources permit."},
        "gradients": {"status": "failed", "scientific_quality": "weak", "reason": "The 16-parameter CPU run was stopped without writing an artifact.", "recommended_action": "Run the six-parameter representative subset over a small operating-point set."},
        "figures": {"status": "partial", "scientific_quality": "weak", "reason": "Baseline, toy-energy, conditioning, and bounded setup-pilot figures have source data, but none establishes main-paper readiness.", "recommended_action": "Keep the controlled energy and bounded setup-pilot figures as supporting evidence; regenerate after valid solver/gradient/ablation data exist."},
    }
    tiers = {"Tier A": [], "Tier B": ["controlled pH energy bookkeeping"], "Tier C": ["vehicle energy observations", "baseline pilot", "production-map finiteness diagnostic", "bounded two-stiffness setup-optimization feasibility pilot"], "Tier D": ["ablations", "learned extrapolation", "expanded gradients", "compute p95"]}
    payload = metadata(cfg, "results_audit") | {"status": "completed", "experiments": records, "baseline_summary": baseline_summary, "evidence_tiers": tiers}
    path = base / "results_audit.json"
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    inventory = {
        "fig07_baseline_rollout": {"data_source": "baseline.csv", "scientific_question": "rollout error", "placement": "supplementary", "status": "weak: deterministic four-trajectory test only"},
        "fig09_energy_balance_toy": {"data_source": "energy.csv", "scientific_question": "controlled pH energy bookkeeping", "placement": "supplementary", "status": "valid supporting evidence"},
        "fig23_compute_cost": {"data_source": "compute.csv", "scientific_question": "warm runtime", "placement": "supplementary", "status": "weak: two repetitions"},
        "fig_objective_sensitivity": {"data_source": "setup_conditioning/objective_surface.json", "scientific_question": "whether setup objective is locally conditioned", "placement": "supplementary", "status": "valid diagnostic, not an optimization result"},
        "fig_setup_optimization_objective": {"data_source": "setup_optimization/setup_optimization_pilot.json", "scientific_question": "whether the bounded pilot has a decreasing objective trajectory", "placement": "supplementary", "status": "Tier C local feasibility evidence; not global optimality"},
        "fig_setup_optimization_parameters": {"data_source": "setup_optimization/setup_optimization_pilot.json", "scientific_question": "whether stiffnesses remain inside the local box", "placement": "supplementary", "status": "Tier C local feasibility evidence; no physical validation"},
        "fig_setup_optimization_gradients": {"data_source": "setup_optimization/setup_optimization_pilot.json", "scientific_question": "AD-versus-FD agreement at three pilot points", "placement": "supplementary", "status": "Tier C local numerical validation; IFT unavailable"},
    }
    (base / "figure_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True))
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default=SuiteConfig().output); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); print(run_audit(SuiteConfig(seed=args.seed, output=args.output)))
