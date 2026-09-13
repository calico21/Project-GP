"""Minimal, leakage-safe foundation for the TTC tire-model benchmark.

The module intentionally prepares no learned model and evaluates no checkpoint.
Its only job is to make complete-run partitions and train-only normalization
auditable before the scientifically substantive benchmark is approved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import ROOT, SuiteConfig, jsonable, manifest, metadata


INPUT_CHANNELS = (
    "alpha_rad", "kappa", "gamma_rad", "Fz_N", "Vx_ms", "P_kPa",
    "T_inner_C", "T_center_C", "T_outer_C",
)
TARGET_CHANNELS = ("Fx_N", "Fy_N")
REQUIRED_CHANNELS = (*INPUT_CHANNELS, *TARGET_CHANNELS, "Mz_Nm", "run")
DEFAULT_SPLIT = {"train": (4,), "validation": (5,), "test": (6,)}


@dataclass(frozen=True)
class CompleteRunSplit:
    train: tuple[int, ...] = DEFAULT_SPLIT["train"]
    validation: tuple[int, ...] = DEFAULT_SPLIT["validation"]
    test: tuple[int, ...] = DEFAULT_SPLIT["test"]

    def validate(self, available_runs: set[int]) -> None:
        roles = {"train": set(self.train), "validation": set(self.validation), "test": set(self.test)}
        if not all(roles.values()):
            raise ValueError("Every partition must contain at least one complete run.")
        if any(roles[left] & roles[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"Runs overlap across partitions: {roles}")
        if set.union(*roles.values()) != available_runs:
            raise ValueError(f"Split must assign every available run exactly once; got {roles}, available={available_runs}.")

    def masks(self, runs: np.ndarray) -> dict[str, np.ndarray]:
        self.validate(set(np.unique(runs.astype(int))))
        return {"train": np.isin(runs, self.train), "validation": np.isin(runs, self.validation), "test": np.isin(runs, self.test)}

    def as_dict(self) -> dict[str, list[int]]:
        return {"train": list(self.train), "validation": list(self.validation), "test": list(self.test)}


@dataclass(frozen=True)
class TrainOnlyScaler:
    """Column-wise standardization whose fit is restricted to training rows."""
    center: np.ndarray
    scale: np.ndarray
    constant_fields: tuple[bool, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "TrainOnlyScaler":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("Scaler fitting requires a non-empty finite two-dimensional training array.")
        center = values.mean(axis=0)
        raw_scale = values.std(axis=0)
        constant = raw_scale <= np.finfo(np.float64).eps
        return cls(center=center, scale=np.where(constant, 1.0, raw_scale), constant_fields=tuple(bool(x) for x in constant))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != len(self.center):
            raise ValueError("Feature count does not match fitted scaler.")
        return (values - self.center) / self.scale

    def serializable(self, channels: tuple[str, ...]) -> dict:
        return {"channels": list(channels), "center": self.center.tolist(), "scale": self.scale.tolist(),
                "constant_fields": list(self.constant_fields), "fit_partition": "train"}


def load_ttc(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = [name for name in REQUIRED_CHANNELS if name not in archive.files]
        if missing:
            raise ValueError(f"TTC archive is missing required channels: {missing}")
        dataset = {name: np.asarray(archive[name]) for name in REQUIRED_CHANNELS}
    count = len(dataset["run"])
    if any(len(values) != count for values in dataset.values()):
        raise ValueError("TTC channel lengths are inconsistent.")
    if not all(np.all(np.isfinite(values)) for values in dataset.values()):
        raise ValueError("TTC archive contains non-finite values in required benchmark channels.")
    return dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coverage(train: np.ndarray, held_out: np.ndarray, channels: tuple[str, ...]) -> list[dict]:
    records = []
    for index, name in enumerate(channels):
        lo, hi = float(train[:, index].min()), float(train[:, index].max())
        outside = (held_out[:, index] < lo) | (held_out[:, index] > hi)
        records.append({"channel": name, "train_min": lo, "train_max": hi,
                        "held_out_outside_train_count": int(outside.sum()),
                        "held_out_outside_train_fraction": float(outside.mean())})
    return records


def prepare_pilot(ttc_path: Path, split: CompleteRunSplit = CompleteRunSplit()) -> dict:
    dataset = load_ttc(ttc_path)
    masks = split.masks(dataset["run"])
    # The mutual-exclusion assertion is intentionally adjacent to the artifact
    # creation: failures cannot be hidden by downstream training code.
    memberships = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(memberships == 1):
        raise AssertionError("Every TTC row must have exactly one partition membership.")
    inputs = np.column_stack([dataset[name] for name in INPUT_CHANNELS])
    targets = np.column_stack([dataset[name] for name in TARGET_CHANNELS])
    input_scaler = TrainOnlyScaler.fit(inputs[masks["train"]])
    target_scaler = TrainOnlyScaler.fit(targets[masks["train"]])
    return {
        "status": "prepared_no_training",
        "source_sha256": _sha256(ttc_path),
        "source_path": str(ttc_path.relative_to(ROOT)) if ttc_path.is_relative_to(ROOT) else str(ttc_path),
        "available_runs": sorted(int(value) for value in np.unique(dataset["run"])),
        "split": split.as_dict(),
        "partition_sample_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "partition_runs_observed": {name: sorted(int(value) for value in np.unique(dataset["run"][mask])) for name, mask in masks.items()},
        "input_scaler": input_scaler.serializable(INPUT_CHANNELS),
        "target_scaler": target_scaler.serializable(TARGET_CHANNELS),
        "test_operating_range_outside_train": _coverage(inputs[masks["train"]], inputs[masks["test"]], INPUT_CHANNELS),
        "stored_is_test_flag": "ignored: it is a row-level partition within every run and is prohibited for this benchmark",
        "model_execution": "none; this pilot neither trains nor evaluates a tire formulation",
    }


def run_pilot(output: str | Path, ttc_path: Path | None = None) -> Path:
    ttc_path = ttc_path or ROOT / "data" / "ttc_round9" / "processed.npz"
    payload = prepare_pilot(ttc_path)
    base = Path(output)
    target = base / "ttc_tire_benchmark_pilot"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "pilot_manifest.json"
    path.write_text(json.dumps(
        jsonable(metadata(SuiteConfig(output=str(base)), "ttc_tire_benchmark_pilot") | payload),
        indent=2, sort_keys=True,
    ))
    manifest(SuiteConfig(output=str(base)))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, but do not train, a complete-run TTC pilot.")
    parser.add_argument("--output", default=str(ROOT / "results" / "paper_suite"))
    parser.add_argument("--ttc-path", type=Path)
    args = parser.parse_args()
    print(run_pilot(args.output, args.ttc_path))


if __name__ == "__main__":
    main()
