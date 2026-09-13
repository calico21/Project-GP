"""Read-only eligibility audit for the repository's independent data sources.

This is deliberately not a predictive benchmark.  It records what the stored
CAN and TTC files can support before models, scalers, or split tensors are
created, so a later experiment cannot silently turn row-wise partitions into
session- or run-held-out evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import ROOT, SuiteConfig, jsonable, manifest, metadata


# These are direct logged observables, rather than a reconstruction of the
# production model's internal 108-state representation.
CAN_OBSERVED_CHANNELS = (
    "ANGLE", "APPS_AV", "BPPS", "a_x", "a_y", "a_z", "v_x", "v_y", "v_z",
    "Yaw_Rate_z", "speed", "rlRPM", "rrRPM", "rlTRQ", "rrTRQ",
)
TTC_REQUIRED_CHANNELS = (
    "alpha_rad", "kappa", "Fy_N", "Fx_N", "Fz_N", "Mz_Nm", "gamma_rad",
    "P_kPa", "Vx_ms", "T_center_C", "T_inner_C", "T_outer_C", "is_test", "run",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _numeric(values: list[str]) -> np.ndarray:
    result = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        try:
            result[i] = float(value)
        except (TypeError, ValueError):
            pass
    return result


def _can_session(path: Path) -> dict:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        wanted = ["timestamp", *CAN_OBSERVED_CHANNELS]
        values = {name: [] for name in wanted if name in fields}
        rows = 0
        for row in reader:
            rows += 1
            for name in values:
                values[name].append(row.get(name, ""))
    timestamps = _numeric(values.get("timestamp", []))
    deltas = np.diff(timestamps)
    positive = deltas[deltas > 0]
    quality = {
        name: float(np.isfinite(_numeric(values[name])).mean())
        for name in CAN_OBSERVED_CHANNELS if name in values
    }
    return {
        "session_id": path.stem,
        "path": _display_path(path),
        "sha256": _sha256(path),
        "row_count": rows,
        "column_count": len(fields),
        "observed_channels_present": [name for name in CAN_OBSERVED_CHANNELS if name in fields],
        "observed_channel_numeric_fraction": quality,
        "timestamp": {
            "present": "timestamp" in fields,
            "finite_fraction": float(np.isfinite(timestamps).mean()) if len(timestamps) else 0.0,
            "start_s": float(timestamps[0]) if len(timestamps) else None,
            "end_s": float(timestamps[-1]) if len(timestamps) else None,
            "monotone_non_decreasing": bool(np.all(deltas >= 0)) if len(deltas) else False,
            "median_positive_dt_s": float(np.median(positive)) if len(positive) else None,
        },
    }


def audit_can(can_dir: Path) -> dict:
    sessions = [_can_session(path) for path in sorted(can_dir.glob("*.csv"))]
    if not sessions:
        raise FileNotFoundError(f"No CAN CSV files found in {can_dir}")
    shared = set(sessions[0]["observed_channels_present"])
    for session in sessions[1:]:
        shared.intersection_update(session["observed_channels_present"])
    ids = [session["session_id"] for session in sessions]
    # This is a deterministic proposal, not a claim that sessions represent
    # independent drivers, tracks, weather, or missions.
    proposed = {"train": ids[:3], "validation": ids[3:4], "test": ids[4:]}
    return {
        "source": "stored real CAN sessions; external acquisition provenance is not established by these files alone",
        "independence_status": "eligible observed-channel source, pending external provenance confirmation",
        "sessions": sessions,
        "shared_observed_channels": sorted(shared),
        "proposed_session_split": proposed,
        "split_rule": "complete session IDs only; never random CAN rows",
        "permitted_claim": "prediction of the listed directly observed channels only",
        "prohibited_claim": "108-state prediction or validation of unobserved thermal, slip, or compliance states",
    }


def audit_ttc(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        missing = [name for name in TTC_REQUIRED_CHANNELS if name not in data.files]
        if missing:
            raise ValueError(f"TTC archive misses required channels: {missing}")
        sizes = {name: int(data[name].size) for name in TTC_REQUIRED_CHANNELS}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"TTC channel lengths disagree: {sizes}")
        runs = data["run"].astype(int)
        row_test = data["is_test"].astype(bool)
    run_ids = sorted(int(value) for value in np.unique(runs))
    per_run = [
        {"run_id": run, "sample_count": int((runs == run).sum()),
         "existing_row_test_count": int(row_test[runs == run].sum())}
        for run in run_ids
    ]
    mixed = any(0 < entry["existing_row_test_count"] < entry["sample_count"] for entry in per_run)
    return {
        "source": "stored TTC Round 9 processed archive",
        "sha256": _sha256(npz_path),
        "sample_count": int(runs.size),
        "channels": list(TTC_REQUIRED_CHANNELS),
        "runs": per_run,
        "existing_is_test_split": {
            "row_level_within_each_run": mixed,
            "eligible_for_held_out_run_claim": False,
            "decision": "do not use this flag as the evaluation split for a run-generalisation claim",
        },
        "required_evaluation": "leave-one-run-out or another complete-run holdout; fit all scaling and residual correction on training runs only",
        "permitted_claim": "held-out tire-force/submodel performance when evaluated with complete run holdout",
        "prohibited_claim": "full vehicle trajectory validation",
    }


def run_audit(output: str | Path, can_dir: Path | None = None, ttc_path: Path | None = None) -> Path:
    """Write an immutable-source inventory and eligibility decision."""
    can_dir = can_dir or ROOT / "data" / "raw_can_logs"
    ttc_path = ttc_path or ROOT / "data" / "ttc_round9" / "processed.npz"
    payload = {
        "status": "completed_read_only_data_eligibility_audit",
        "can": audit_can(can_dir),
        "ttc": audit_ttc(ttc_path),
        "overall_decision": "No independent 108-state benchmark is eligible. CAN supports an observed-channel, session-held-out study; TTC supports a complete-run-held-out tire study.",
    }
    base = Path(output)
    target = base / "independent_data_audit"
    target.mkdir(parents=True, exist_ok=True)
    result = target / "independent_data_audit.json"
    result.write_text(json.dumps(
        jsonable(metadata(SuiteConfig(output=str(base)), "independent_data_audit") | payload),
        indent=2, sort_keys=True,
    ))
    report = """# Independent-data eligibility audit

The repository does not contain an independent target for all 108 production-model states. The CAN files are eligible only for directly logged, session-held-out observables; source-file hashes and channel availability are recorded in the accompanying JSON. The TTC archive has data from three runs, but its existing `is_test` flag partitions rows within every run. It must not support a run-generalisation claim. Use a complete-run holdout and training-only scaling for a tire submodel study.
"""
    (target / "independent_data_audit.md").write_text(report)
    manifest(SuiteConfig(output=str(base)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "paper_suite"))
    parser.add_argument("--can-dir", type=Path)
    parser.add_argument("--ttc-path", type=Path)
    args = parser.parse_args()
    print(run_audit(args.output, args.can_dir, args.ttc_path))


if __name__ == "__main__":
    main()
