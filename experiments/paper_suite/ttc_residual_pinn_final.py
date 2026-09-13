"""One-shot locked Run-6 evaluator for an already selected TTC residual pilot.

Do not invoke this module until the Run-4/Run-5 pilot artifact is accepted.
It restores the selected checkpoint and frozen train-only metadata; it performs
no optimization, selection, sign fitting, or normalization fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from .common import ROOT, SuiteConfig, jsonable, manifest, metadata
from .ttc_benchmark import TrainOnlyScaler
from .ttc_pacejka_baseline import AnalyticalPacejkaAdapter, _metrics
from .ttc_residual_pinn import Candidate, TTCAdditiveResidualPINN, _feature_matrix, _load_roles, _predict, coverage_partition


def run_final(pilot_path: Path, output: str | Path, ttc_path: Path | None = None) -> Path:
    pilot = json.loads(pilot_path.read_text())
    if pilot.get("status") != "completed_train_run4_validation_run5_only":
        raise ValueError("Pilot artifact is not an accepted Run-4/Run-5-only selection record.")
    ttc_path = ttc_path or ROOT / "data" / "ttc_round9" / "processed.npz"
    if hashlib.sha256(ttc_path.read_bytes()).hexdigest() != pilot["provenance"]["source_sha256"]:
        raise ValueError("TTC source hash differs from the locked pilot source.")
    base, root = Path(output), Path(output) / "ttc_residual_pinn_final"
    path = root / "ttc_residual_pinn_final.json"
    if path.exists():
        raise FileExistsError("Locked Run-6 artifact already exists; refusing a repeat evaluation.")
    selected_name = pilot["selected"]["name"]
    selected = next(item for item in pilot["candidates"] if item["candidate"]["name"] == selected_name)
    candidate = Candidate(**selected["candidate"])
    model = TTCAdditiveResidualPINN(candidate.widths)
    template = model.init(jax.random.PRNGKey(candidate.seed), jnp.zeros(len(pilot["provenance"]["features"]["names"])))
    params = flax.serialization.from_bytes(template, (pilot_path.parent / selected["checkpoint"]).read_bytes())
    roles, _ = _load_roles(ttc_path, ("test",))
    test = roles["test"]
    adapter = AnalyticalPacejkaAdapter()
    baseline = pilot["provenance"]["frozen_sign"]["model_to_ttc_fy_multiplier"] * adapter.predict(test, np.ones(len(test["Fy_N"]), dtype=bool))[1]
    residual_scaler = pilot["provenance"]["residual_scaler"]
    center, scale = residual_scaler["center"][0], residual_scaler["scale"][0]
    input_scaler_data = pilot["provenance"]["input_scaler"]
    input_scaler = TrainOnlyScaler(np.asarray(input_scaler_data["center"]), np.asarray(input_scaler_data["scale"]), tuple(input_scaler_data["constant_fields"]))
    residual = _predict(model, params, input_scaler.transform(_feature_matrix(test)).astype(np.float32)) * scale + center
    corrected = baseline + residual
    required_threshold_channels = {
        "alpha_rad",
        "Fz_N",
        "gamma_rad",
        "Vx_ms",
        "P_kPa",
    }
    thresholds = pilot["provenance"]["coverage_thresholds"]
    if set(thresholds) != required_threshold_channels:
        raise ValueError(
            "Locked coverage thresholds do not match the required benchmark channels."
        )
    thresholds = pilot["provenance"]["coverage_thresholds"]
    speed_in = (test["Vx_ms"] >= thresholds["Vx_ms"]["min"]) & (test["Vx_ms"] <= thresholds["Vx_ms"]["max"])
    if pilot["provenance"].get("run6_used_for_fitting_or_selection", False):
        raise ValueError("Pilot provenance indicates Run 6 influenced fitting or selection.")
    result = {
        "status": "completed_locked_run6_evaluation", "selected_candidate": selected_name,
        "pilot_path": str(pilot_path), "source_sha256": pilot["provenance"]["source_sha256"],
        "test_run": 6, "global": {"analytical": _metrics(test["Fy_N"], baseline, test["Fz_N"]), "corrected": _metrics(test["Fy_N"], corrected, test["Fz_N"])},
        "speed_interpolation": {"analytical": _metrics(test["Fy_N"][speed_in], baseline[speed_in], test["Fz_N"][speed_in]), "corrected": _metrics(test["Fy_N"][speed_in], corrected[speed_in], test["Fz_N"][speed_in])},
        "speed_extrapolation": {"analytical": _metrics(test["Fy_N"][~speed_in], baseline[~speed_in], test["Fz_N"][~speed_in]), "corrected": _metrics(test["Fy_N"][~speed_in], corrected[~speed_in], test["Fz_N"][~speed_in])},
        "coverage": coverage_partition(test, thresholds),
        "guarantee": "This evaluator restored a locked Run-4/Run-5 selection artifact and performed no fitting, sign selection, scaling fit, or hyperparameter selection.",
    }
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(metadata(SuiteConfig(output=str(base)), "ttc_residual_pinn_final") | result), indent=2, sort_keys=True))
    manifest(SuiteConfig(output=str(base)))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot locked Run-6 TTC residual evaluation.")
    parser.add_argument("--pilot", type=Path, required=True); parser.add_argument("--output", default=str(ROOT / "results" / "paper_suite")); parser.add_argument("--ttc-path", type=Path)
    args = parser.parse_args(); print(run_final(args.pilot, args.output, args.ttc_path))


if __name__ == "__main__":
    main()
