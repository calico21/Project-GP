"""Read-only evidence audit for a completed serious-benchmark campaign."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .common import SuiteConfig, jsonable, manifest, metadata

HORIZONS = (1, 10, 20, 50, 100, 200)

def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

def _csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader(); writer.writerows(rows)

def audit_serious_benchmark(output):
    """Aggregate persisted metrics only; this never loads checkpoints or models."""
    base = Path(output); root = base / "serious_benchmark"
    records = json.loads((root / "results.json").read_text())["records"]
    protocol = json.loads((root / "protocol.json").read_text())
    campaign = json.loads((root / "campaign_state.json").read_text())
    expected = {(m, s) for m in protocol["config"]["models"] for s in protocol["config"]["seeds"]}
    observed = {(r["model"], r["seed"]) for r in records}
    if observed != expected or len(records) != len(expected):
        raise ValueError("Campaign is not a complete one-record-per-model/seed result set.")
    aggregate, per_state = [], []
    for model in protocol["config"]["models"]:
        group = [r for r in records if r["model"] == model]
        row = {"model": model, "seed_count": len(group)}
        metrics = {
            "one_step_rmse": [r["one_step_rmse"] for r in group],
            "divergence_rate": [np.mean([x["has_diverged"] for x in r["stability"]]) for r in group],
            "parameter_count": [r["parameter_count"] for r in group],
            "training_time_s": [r["training_time_s"] for r in group],
            "inference_time_s": [r["inference_time_s"] for r in group],
        }
        for key, values in metrics.items():
            row[f"{key}_mean"], row[f"{key}_std"] = _mean_std(values)
        for horizon in HORIZONS:
            row[f"rmse_{horizon}_mean"], row[f"rmse_{horizon}_std"] = _mean_std([r["rollout_rmse"][str(horizon)] for r in group])
        row["energy_diagnostics"] = "unavailable: no shared calibrated physical-energy observable"
        aggregate.append(row)
        for state in sorted(group[0]["per_state_error"]):
            mean, std = _mean_std([r["per_state_error"][state]["rmse"] for r in group])
            per_state.append({"model": model, "state": state, "state_group": "mechanical",
                              "raw_rmse_mean": mean, "raw_rmse_std": std,
                              "normalized_rmse": "unavailable",
                              "normalization_reason": "test-state scales were not persisted"})
        for absent in ("thermal", "slip", "hysteresis/compliance"):
            per_state.append({"model": model, "state": "not_modelled", "state_group": absent,
                              "raw_rmse_mean": None, "raw_rmse_std": None,
                              "normalized_rmse": "not_applicable",
                              "normalization_reason": "28-state mechanical benchmark excludes this 108-state block"})
    payload = {
        "status": "completed_read_only_audit", "record_count": len(records),
        "campaign_completed": campaign["status"] == "completed", "aggregate": aggregate,
        "per_state_errors": {"representation": "28-state [q,p=M_diag*v] mechanical surrogate",
          "normalization": "unavailable from persisted artifacts",
          "groups": {"mechanical": 28, "thermal": 0, "slip": 0, "hysteresis/compliance": 0}},
        "fairness": {
          "same_28_state_target_representation": "yes, for all five wrappers",
          "same_normalization": "yes only because none was applied; this is a scale flaw",
          "same_splits_within_seed": "yes by deterministic split seed and model loop; raw arrays were not persisted",
          "same_rollout_and_stability_definition": "yes: common 200-step controls and relative-RMSE divergence rule",
          "matched_full_108_state_production_model": "no: target is projected to 28 mechanical states"},
        "flaws": [
          "Not a full 108-state PassiveHNet/GLRK benchmark: all wrappers operate on a 28-state mechanical projection.",
          "No state normalization was applied or persisted; raw RMSE mixes heterogeneous q and momentum units.",
          "Test state arrays/scales were not persisted, so normalized per-state errors cannot be reconstructed.",
          "HNN and PassiveHNetBaseline predict_step ignore control input, unlike NeuralODE/PINN/PHNN.",
          "PassiveHNetBaseline is a 28-state symplectic-Euler wrapper, not the production 108-state implicit GLRK map.",
          "Energy diagnostics are unavailable for all models; no energy/passivity conclusion is supported.",
          "Three seeds combine data-split and initialization variation; no confidence intervals or hypothesis tests are justified.",
          "Raw split tensors and source/config hashes were not saved; exact replay depends on current source/configuration.",
          "Relative-error divergence is a numerical diagnostic, not a physical-stability or passivity certificate."],
        "evidence_tier": "Tier C protocol/evaluation evidence only; unsuitable for predictive-superiority or full-vehicle paper claims."}
    _csv(root / "serious_benchmark_audit_aggregate.csv", aggregate)
    _csv(root / "serious_benchmark_audit_per_state.csv", per_state)
    (root / "serious_benchmark_audit.json").write_text(json.dumps(jsonable(metadata(SuiteConfig(output=str(base)), "serious_benchmark_audit") | payload), indent=2, sort_keys=True))
    latex = ["\\begin{tabular}{lrrrr}", "\\toprule", "Model & $\\mathrm{RMSE}_{1}$ & $\\mathrm{RMSE}_{200}$ & Divergence & Train [s] \\\\ \\midrule"]
    latex += [f"{r['model']} & {r['rmse_1_mean']:.4g} $\\pm$ {r['rmse_1_std']:.2g} & {r['rmse_200_mean']:.4g} $\\pm$ {r['rmse_200_std']:.2g} & {r['divergence_rate_mean']:.3g} $\\pm$ {r['divergence_rate_std']:.2g} & {r['training_time_s_mean']:.4g} $\\pm$ {r['training_time_s_std']:.2g} \\\\" for r in aggregate]
    latex += ["\\bottomrule", "\\end{tabular}"]
    (root / "table_serious_benchmark_audit.tex").write_text("\n".join(latex))
    groups = ["\\begin{tabular}{lll}", "\\toprule", "State group & Dimensions & Normalised error \\\\ \\midrule",
      "mechanical & 28 & unavailable (test scales not persisted) \\\\",
      "thermal & 0 & not applicable \\\\", "slip & 0 & not applicable \\\\",
      "hysteresis/compliance & 0 & not applicable \\\\", "\\bottomrule", "\\end{tabular}"]
    (root / "table_serious_benchmark_state_groups.tex").write_text("\n".join(groups))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4)); names = [r["model"] for r in aggregate]
    axes[0].bar(names, [r["rmse_200_mean"] for r in aggregate], yerr=[r["rmse_200_std"] for r in aggregate], capsize=3)
    axes[0].set(title="200-step rollout RMSE", ylabel="raw RMSE"); axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(names, [r["divergence_rate_mean"] for r in aggregate], yerr=[r["divergence_rate_std"] for r in aggregate], capsize=3)
    axes[1].set(title="Relative-error divergence", ylabel="trajectory fraction"); axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(root / f"fig_serious_benchmark_audit.{ext}", dpi=300)
    plt.close(fig)
    report = """# Serious-benchmark audit

## Decision

The 15 records are internally complete, but this benchmark is **not suitable as paper evidence of full-vehicle prediction, PassiveHNet superiority, energy behavior, or physical stability**. It is Tier C protocol/evaluation evidence for 28-state mechanical surrogate wrappers only.

## Fairness

All wrappers use the same 28-state target, per-seed deterministic split stream, 200-step rollout controls, and relative-RMSE divergence definition. No normalization was applied; this uniform absence is a scale defect. Raw arrays and target scales were not retained.

## Required next experiment

Build a matched 108-state benchmark with persisted split tensors/scalers and statewise normalization. Every model must receive equivalent forced-dynamics inputs, and any PassiveHNet claim must evaluate the production GLRK map. Record comparable energy quantities before model-comparison conclusions.
"""
    (root / "serious_benchmark_audit.md").write_text(report)
    manifest(SuiteConfig(output=str(base)))
    return root / "serious_benchmark_audit.json"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default=SuiteConfig().output)
    args = parser.parse_args(); print(audit_serious_benchmark(args.output))
