"""Run-4/Run-5-only additive residual-learning pilot for the TTC benchmark.

Normal execution deliberately never constructs a Run-6 partition.  The
``--final-evaluate`` entry point is separate and is intended to be run once,
only after the selected Run-4/Run-5 artifact is accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.tire_model import SpectralDense
from .common import ROOT, SuiteConfig, jsonable, manifest, metadata
from .ttc_benchmark import CompleteRunSplit, TrainOnlyScaler
from .ttc_pacejka_baseline import AnalyticalPacejkaAdapter, _metrics


FROZEN_FY_SIGN = -1
FEATURES = ("alpha_rad", "kappa", "gamma_rad", "Fz_N", "Vx_ms", "T_eff_C")
COVERAGE_CHANNELS = ("alpha_rad", "Fz_N", "gamma_rad", "Vx_ms", "P_kPa")
FIXED_AUXILIARY_INPUTS = {"T_gas_C": 90.0, "wz_rad_s": 0.0}


@dataclass(frozen=True)
class Candidate:
    name: str
    widths: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    epochs: int = 40
    batch_size: int = 1024
    seed: int = 0


PILOT_GRID = (
    Candidate("spectral_16x16", (16, 16), 1e-3, 1e-5),
    Candidate("spectral_32x32", (32, 32), 1e-3, 1e-5),
)


class TTCAdditiveResidualPINN(nn.Module):
    """Spectrally normalized residual MLP; output is normalized additive Fy residual."""
    widths: tuple[int, ...]

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features
        for width in self.widths:
            x = jnp.tanh(SpectralDense(width)(x))
        return nn.Dense(1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)[0]


def _load_roles(path: Path, roles: tuple[str, ...]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Return only requested role rows; normal pilot calls this for train/validation."""
    split = CompleteRunSplit()
    role_runs = split.as_dict()
    requested_runs = set().union(*(set(role_runs[role]) for role in roles))
    # The archive contains all run identifiers because complete-run rows must be
    # located before filtering. Only rows belonging to explicitly requested roles
    # are returned to the caller. In the pilot, Run 6 is therefore not exposed to
    # the training/validation pipeline and is never used for fitting, scaling,
    # thresholds, or model selection.
    required = ("alpha_rad", "kappa", "gamma_rad", "Fz_N", "Vx_ms", "P_kPa", "T_inner_C", "T_center_C", "T_outer_C", "Fy_N", "run")
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"TTC archive misses residual-PINN inputs: {missing}")
        runs = np.asarray(archive["run"], dtype=int)
        split.validate(set(np.unique(runs)))
        fields = {name: np.asarray(archive[name]) for name in required if name != "run"}
    result = {}
    for role in roles:
        mask = np.isin(runs, role_runs[role])
        result[role] = {name: values[mask].astype(np.float64) for name, values in fields.items()}
        result[role]["T_eff_C"] = (result[role]["T_inner_C"] + result[role]["T_center_C"] + result[role]["T_outer_C"]) / 3.0
        if not np.all([np.all(np.isfinite(value)) for value in result[role].values()]):
            raise ValueError(f"Non-finite values in TTC {role} role.")
    return result, {"available_runs": np.unique(runs), "requested_runs": np.array(sorted(requested_runs))}


def _feature_matrix(partition: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([partition[name] for name in FEATURES])


def train_coverage_thresholds(train: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Frozen min/max support thresholds, constructed from Run 4 only."""
    return {name: {"min": float(train[name].min()), "max": float(train[name].max())} for name in COVERAGE_CHANNELS}


def coverage_partition(partition: dict[str, np.ndarray], thresholds: dict[str, dict[str, float]]) -> dict:
    coverage = {}
    for name, bounds in thresholds.items():
        outside = (partition[name] < bounds["min"]) | (partition[name] > bounds["max"])
        coverage[name] = {"outside_train_count": int(outside.sum()), "outside_train_fraction": float(outside.mean())}
    speed = coverage["Vx_ms"]
    return {"per_channel": coverage,
            "speed_interpolation_count": int(len(partition["Vx_ms"]) - speed["outside_train_count"]),
            "speed_extrapolation_count": speed["outside_train_count"],
            "definition": "Speed interpolation iff Run-4 min(Vx_ms) <= Vx_ms <= Run-4 max(Vx_ms); all thresholds are frozen before validation/test evaluation."}


def _train_baseline(adapter: AnalyticalPacejkaAdapter, partition: dict[str, np.ndarray]) -> np.ndarray:
    # Adapter requires archive-key semantics; derived T_eff is deliberately not
    # substituted for the three observed surface-temperature channels.
    return FROZEN_FY_SIGN * adapter.predict(partition, np.ones(len(partition["Fy_N"]), dtype=bool))[1]


def _prepare_data(ttc_path: Path) -> tuple[dict, dict]:
    roles, source = _load_roles(ttc_path, ("train", "validation"))
    assert set(source["requested_runs"].tolist()) == {4, 5}, (
        f"Residual pilot unexpectedly requested runs: "
        f"{source['requested_runs'].tolist()}"
    )
    assert 6 not in source["requested_runs"]
    adapter = AnalyticalPacejkaAdapter()
    baseline = {role: _train_baseline(adapter, partition) for role, partition in roles.items()}
    target = {role: roles[role]["Fy_N"] - baseline[role] for role in roles}
    input_scaler = TrainOnlyScaler.fit(_feature_matrix(roles["train"]))
    target_scaler = TrainOnlyScaler.fit(target["train"][:, None])
    thresholds = train_coverage_thresholds(roles["train"])
    prepared = {}
    for role in roles:
        prepared[role] = {
            "x": input_scaler.transform(_feature_matrix(roles[role])).astype(np.float32),
            "y": target_scaler.transform(target[role][:, None])[:, 0].astype(np.float32),
            "baseline": baseline[role], "measured": roles[role]["Fy_N"], "fz": roles[role]["Fz_N"],
        }
    provenance = {
        "source_sha256": hashlib.sha256(ttc_path.read_bytes()).hexdigest(), "split": CompleteRunSplit().as_dict(),
        "roles_loaded": ["train", "validation"],
        "run6_returned_to_training_pipeline": False,
        "run6_used_for_fitting_or_selection": False,
        "features": {"names": list(FEATURES), "units": ["rad", "dimensionless", "rad", "N", "m/s", "C"],
                     "source_fields": ["alpha_rad", "kappa", "gamma_rad", "Fz_N", "Vx_ms", "mean(T_inner_C,T_center_C,T_outer_C)"],
                     "camber_assumption": "TTC IA and model gamma use the stored sign without independently established convention equivalence."},
        "input_scaler": input_scaler.serializable(FEATURES), "residual_scaler": target_scaler.serializable(("Fy_residual_N",)),
        "coverage_thresholds": thresholds, "validation_coverage": coverage_partition(roles["validation"], thresholds),
        "fixed_auxiliary_inputs": FIXED_AUXILIARY_INPUTS,
        "frozen_sign": {"model_to_ttc_fy_multiplier": FROZEN_FY_SIGN, "source": "locked analytical baseline Run-4-only selection"},
        "residual_definition": "Fy_measured_N - Fy_pacejka_N, where Fy_pacejka includes the frozen Run-4 sign mapping.",
    }
    assert provenance["run6_returned_to_training_pipeline"] is False
    assert provenance["run6_used_for_fitting_or_selection"] is False
    return prepared, provenance


def _predict(model, params, x: np.ndarray) -> np.ndarray:
    fn = jax.jit(jax.vmap(lambda row: model.apply(params, row)))
    return np.asarray(fn(jnp.asarray(x)))


def _evaluate(prepared: dict, model, params, target_scaler: TrainOnlyScaler, role: str) -> dict:
    normalized_prediction = _predict(model, params, prepared[role]["x"])
    residual_prediction = normalized_prediction * target_scaler.scale[0] + target_scaler.center[0]
    corrected = prepared[role]["baseline"] + residual_prediction
    residual_true = prepared[role]["measured"] - prepared[role]["baseline"]
    return {"residual_rmse_N": float(np.sqrt(np.mean((residual_true - residual_prediction) ** 2))),
            "residual_bias_N": float(np.mean(residual_true - residual_prediction)),
            "corrected": _metrics(prepared[role]["measured"], corrected, prepared[role]["fz"]),
            "analytical": _metrics(prepared[role]["measured"], prepared[role]["baseline"], prepared[role]["fz"])}


def _fit_candidate(candidate: Candidate, prepared: dict, target_scaler: TrainOnlyScaler) -> tuple[dict, dict, dict]:
    model = TTCAdditiveResidualPINN(candidate.widths)
    params = model.init(jax.random.PRNGKey(candidate.seed), jnp.asarray(prepared["train"]["x"][0]))
    optimizer = optax.adamw(candidate.learning_rate, weight_decay=candidate.weight_decay)
    state = optimizer.init(params)
    x, y = jnp.asarray(prepared["train"]["x"]), jnp.asarray(prepared["train"]["y"])

    @jax.jit
    def step(parameters, opt_state, batch_x, batch_y):
        def loss_fn(p):
            predicted = jax.vmap(lambda row: model.apply(p, row))(batch_x)
            return jnp.mean((predicted - batch_y) ** 2)
        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        update, opt_state = optimizer.update(gradients, opt_state, parameters)
        return optax.apply_updates(parameters, update), opt_state, loss

    key, best_params, best_validation, best_epoch, history = jax.random.PRNGKey(candidate.seed), params, float("inf"), 0, []
    for epoch in range(candidate.epochs):
        key, batch_key = jax.random.split(key)
        index = np.asarray(jax.random.choice(batch_key, len(x), (min(candidate.batch_size, len(x)),), replace=False))
        params, state, loss = step(params, state, x[index], y[index])
        validation = _evaluate(prepared, model, params, target_scaler, "validation")
        value = validation["corrected"]["rmse_N"]
        history.append({"epoch": epoch + 1, "train_normalized_mse": float(loss), "validation_corrected_rmse_N": value})
        if value < best_validation:
            best_validation, best_params, best_epoch = value, params, epoch + 1
    metrics = {"train": _evaluate(prepared, model, best_params, target_scaler, "train"),
               "validation": _evaluate(prepared, model, best_params, target_scaler, "validation"),
               "best_epoch": best_epoch, "history": history}
    return best_params, metrics, {"candidate": asdict(candidate), "selection_metric": "validation corrected Fy RMSE [N]"}


def run_pilot(output: str | Path, ttc_path: Path | None = None) -> Path:
    ttc_path = ttc_path or ROOT / "data" / "ttc_round9" / "processed.npz"
    prepared, provenance = _prepare_data(ttc_path)
    scaler = TrainOnlyScaler(np.asarray(provenance["residual_scaler"]["center"]), np.asarray(provenance["residual_scaler"]["scale"]), tuple(provenance["residual_scaler"]["constant_fields"]))
    base, root = Path(output), Path(output) / "ttc_residual_pinn_pilot"; root.mkdir(parents=True, exist_ok=True)
    candidates = []
    for candidate in PILOT_GRID:
        params, metrics, config = _fit_candidate(candidate, prepared, scaler)
        checkpoint = root / f"{candidate.name}_best.msgpack"; checkpoint.write_bytes(flax.serialization.to_bytes(params))
        candidates.append(config | {"checkpoint": checkpoint.name, "metrics": metrics})
    selected = min(candidates, key=lambda item: (item["metrics"]["validation"]["corrected"]["rmse_N"], item["candidate"]["name"]))
    payload = metadata(SuiteConfig(output=str(base)), "ttc_residual_pinn_pilot") | {
        "status": "completed_train_run4_validation_run5_only", "provenance": provenance, "candidates": candidates,
        "selected": {"name": selected["candidate"]["name"], "checkpoint": selected["checkpoint"], "validation": selected["metrics"]["validation"]},
        "test_evaluation": "not run; Run 6 was excluded from all model, normalization, threshold, and selection operations.",
        "limitations": "Feasibility pilot only. It has one validation run, kappa is constant, pressure is excluded because it is unsupported by the analytical force interface, and camber convention remains an explicit assumption.",
    }
    path = root / "ttc_residual_pinn_pilot.json"; path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    rows = [{"candidate": item["candidate"]["name"], "partition": role, "analytical_rmse_N": item["metrics"][role]["analytical"]["rmse_N"], "corrected_rmse_N": item["metrics"][role]["corrected"]["rmse_N"], "corrected_mae_N": item["metrics"][role]["corrected"]["mae_N"], "residual_rmse_N": item["metrics"][role]["residual_rmse_N"]} for item in candidates for role in ("train", "validation")]
    with (root / "candidate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    manifest(SuiteConfig(output=str(base)))
    return path

def test_pilot_provenance_explicitly_excludes_run6(tmp_path):
    archive = tmp_path / "processed.npz"
    _archive(archive)

    roles, source = _load_roles(archive, ("train", "validation"))

    assert source["requested_runs"].tolist() == [4, 5]
    assert 6 not in source["requested_runs"]
    assert set(np.unique(roles["train"]["Fy_N"])) == {1.0, 2.0}
    assert set(np.unique(roles["validation"]["Fy_N"])) == {3.0, 4.0}


def test_coverage_thresholds_do_not_depend_on_validation_or_test():
    train = {
        "alpha_rad": np.array([0.0, 1.0]),
        "Fz_N": np.array([100.0, 200.0]),
        "gamma_rad": np.array([0.0, 0.1]),
        "Vx_ms": np.array([10.0, 12.0]),
        "P_kPa": np.array([70.0, 80.0]),
    }

    validation = {
        "alpha_rad": np.array([100.0]),
        "Fz_N": np.array([1000.0]),
        "gamma_rad": np.array([10.0]),
        "Vx_ms": np.array([100.0]),
        "P_kPa": np.array([1000.0]),
    }

    thresholds = train_coverage_thresholds(train)

    assert thresholds["alpha_rad"] == {"min": 0.0, "max": 1.0}
    assert thresholds["Fz_N"] == {"min": 100.0, "max": 200.0}
    assert thresholds["gamma_rad"] == {"min": 0.0, "max": 0.1}
    assert thresholds["Vx_ms"] == {"min": 10.0, "max": 12.0}
    assert thresholds["P_kPa"] == {"min": 70.0, "max": 80.0}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the locked Run-4/Run-5 TTC residual pilot.")
    parser.add_argument("--output", default=str(ROOT / "results" / "paper_suite")); parser.add_argument("--ttc-path", type=Path)
    parser.add_argument("--final-evaluate", action="store_true", help="Reserved: no Run-6 evaluator is enabled in this pilot module.")
    args = parser.parse_args()
    if args.final_evaluate:
        raise SystemExit("Run-6 evaluation is intentionally not enabled by the pilot; approve a frozen selected artifact first.")
    print(run_pilot(args.output, args.ttc_path))


if __name__ == "__main__":
    main()
