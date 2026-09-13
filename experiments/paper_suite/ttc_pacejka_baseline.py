"""Locked-split analytical Pacejka/MF6.2 TTC baseline pilot.

This adapter intentionally has no optimizer, scaler fit, checkpoint loading, or
row-wise split.  Its sole fitted quantity is a global force-sign mapping chosen
from Run 4; the mapping is frozen before runs 5 and 6 are evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.tire_coeffs import tire_coeffs
from models.tire_model import PacejkaTire
from .common import ROOT, SuiteConfig, jsonable, manifest, metadata
from .ttc_benchmark import CompleteRunSplit, INPUT_CHANNELS, TARGET_CHANNELS, TrainOnlyScaler, load_ttc


FIXED_AUXILIARY_INPUTS = {
    "T_gas_C": 90.0,
    "wz_rad_s": 0.0,
    "mu_scale": 1.0,
    "T_opt_override_C": -1.0,
    "alpha_scale": 1.0,
    "rby1_scale": 1.0,
    "rby2_scale": 1.0,
}


def choose_force_sign(measured_fy: np.ndarray, raw_predicted_fy: np.ndarray) -> dict:
    """Choose the one global TTC-to-model Fy sign mapping from training data."""
    candidates = {sign: float(np.sqrt(np.mean((measured_fy - sign * raw_predicted_fy) ** 2))) for sign in (1, -1)}
    sign = min(candidates, key=lambda value: (candidates[value], -value))
    return {"model_to_ttc_fy_multiplier": sign, "training_rmse_by_multiplier_N": candidates,
            "selection_partition": "train (Run 4) only"}


class AnalyticalPacejkaAdapter:
    """Existing MF6.2 force path with learned residual and penalty disabled."""
    name = "Pacejka MF6.2 analytical (residual disabled)"

    def __init__(self):
        self.tire = PacejkaTire(tire_coeffs, rng_seed=0)

        def one(alpha, kappa, fz, gamma, vx, t_inner, t_center, t_outer):
            return self.tire.compute_force_and_sigma(
                alpha=alpha, kappa=kappa, Fz=fz, gamma=gamma,
                T_ribs=jnp.array([t_inner, t_center, t_outer]),
                T_gas=jnp.array(FIXED_AUXILIARY_INPUTS["T_gas_C"]), Vx=vx,
                stochastic_key=None, apply_residual=False,
                wz=jnp.array(FIXED_AUXILIARY_INPUTS["wz_rad_s"]),
                mu_scale=jnp.array(FIXED_AUXILIARY_INPUTS["mu_scale"]),
                T_opt_override=jnp.array(FIXED_AUXILIARY_INPUTS["T_opt_override_C"]),
                alpha_scale=jnp.array(FIXED_AUXILIARY_INPUTS["alpha_scale"]),
                rby1_scale=jnp.array(FIXED_AUXILIARY_INPUTS["rby1_scale"]),
                rby2_scale=jnp.array(FIXED_AUXILIARY_INPUTS["rby2_scale"]),
            )[:2]
        self._predict = jax.jit(jax.vmap(one))

    def predict(self, dataset: dict[str, np.ndarray], mask: np.ndarray, chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
        required = ("alpha_rad", "kappa", "Fz_N", "gamma_rad", "Vx_ms", "T_inner_C", "T_center_C", "T_outer_C")
        if any(name not in dataset for name in required):
            raise ValueError("TTC baseline inputs are incomplete; no default substitution is permitted.")
        selected = {name: np.asarray(dataset[name][mask], dtype=np.float64) for name in required}
        fx, fy = [], []
        for start in range(0, len(selected["alpha_rad"]), chunk_size):
            sl = slice(start, min(start + chunk_size, len(selected["alpha_rad"])))
            output = self._predict(*[jnp.asarray(selected[name][sl]) for name in required])
            fx.append(np.asarray(output[0])); fy.append(np.asarray(output[1]))
        return np.concatenate(fx), np.concatenate(fy)


def _metrics(measured: np.ndarray, predicted: np.ndarray, fz: np.ndarray) -> dict:
    error = measured - predicted
    variance = np.sum((measured - measured.mean()) ** 2)
    mu_measured, mu_predicted = measured / fz, predicted / fz
    return {
        "count": int(len(measured)), "rmse_N": float(np.sqrt(np.mean(error ** 2))),
        "mae_N": float(np.mean(np.abs(error))), "bias_N": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error ** 2) / (variance + 1e-12)),
        "mu_rmse": float(np.sqrt(np.mean((mu_measured - mu_predicted) ** 2))),
        "mu_mae": float(np.mean(np.abs(mu_measured - mu_predicted))),
        "finite": bool(np.all(np.isfinite(predicted))),
    }


def _binned_metrics(dataset: dict, mask: np.ndarray, predicted_fy: np.ndarray) -> list[dict]:
    values = {name: np.asarray(dataset[name][mask]) for name in ("alpha_rad", "Fz_N", "gamma_rad", "Vx_ms", "Fy_N")}
    specs = {
        "abs_slip_angle_deg": (np.abs(np.rad2deg(values["alpha_rad"])), (0, 2, 4, 8, 13)),
        "Fz_N": (values["Fz_N"], (100, 350, 550, 750, 1000, 1300)),
        "camber_deg": (np.rad2deg(values["gamma_rad"]), (-1, 0, 1, 3, 5)),
        "speed_ms": (values["Vx_ms"], (0, 10, 12, 15, 25)),
    }
    rows = []
    for name, (x, edges) in specs.items():
        for low, high in zip(edges[:-1], edges[1:]):
            choose = (x >= low) & (x < high)
            if choose.any():
                rows.append({"stratum": name, "low": low, "high": high,
                             **_metrics(values["Fy_N"][choose], predicted_fy[choose], values["Fz_N"][choose])})
    return rows


def _coverage(train: dict, test: dict) -> list[dict]:
    rows = []
    for name in INPUT_CHANNELS:
        lo, hi = float(train[name].min()), float(train[name].max())
        out = (test[name] < lo) | (test[name] > hi)
        rows.append({"channel": name, "train_min": lo, "train_max": hi,
                     "test_outside_train_count": int(out.sum()), "test_outside_train_fraction": float(out.mean())})
    return rows


def _figures(path: Path, test: dict, fy_pred: np.ndarray, coverage: list[dict]) -> None:
    measured, fz = test["Fy_N"], test["Fz_N"]
    residual = measured - fy_pred
    alpha_deg = np.rad2deg(test["alpha_rad"])
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    limit = max(np.max(np.abs(measured)), np.max(np.abs(fy_pred)))
    axes[0, 0].hexbin(measured, fy_pred, gridsize=70, mincnt=1, cmap="viridis")
    axes[0, 0].plot([-limit, limit], [-limit, limit], "k--", lw=1)
    axes[0, 0].set(xlabel="Measured $F_y$ [N]", ylabel="Predicted $F_y$ [N]", title="Locked Run 6: predicted vs measured")
    axes[0, 1].hist(residual, bins=70, color="#386cb0")
    axes[0, 1].set(xlabel="$F_y$ residual (measured − predicted) [N]", ylabel="Count", title="Residual distribution")
    axes[1, 0].hexbin(alpha_deg, residual, gridsize=70, mincnt=1, cmap="magma")
    axes[1, 0].axhline(0, color="k", lw=1)
    axes[1, 0].set(xlabel="Slip angle [deg]", ylabel="Residual [N]", title="Error versus slip angle")
    axes[1, 1].hexbin(fz, residual, gridsize=70, mincnt=1, cmap="magma")
    axes[1, 1].axhline(0, color="k", lw=1)
    axes[1, 1].set(xlabel="$F_z$ [N]", ylabel="Residual [N]", title="Error versus normal load")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(path / f"fig_ttc_pacejka_force_error.{ext}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    train_alpha, train_fz = np.rad2deg(test["_train_alpha"]), test["_train_fz"]
    axes[0].scatter(train_alpha, train_fz, s=1, alpha=.08, label="Run 4 train")
    axes[0].scatter(alpha_deg, fz, s=1, alpha=.08, label="Run 6 test")
    axes[0].set(xlabel="Slip angle [deg]", ylabel="$F_z$ [N]", title="Operating range")
    axes[0].legend(markerscale=4)
    test_speed, train_speed = test["Vx_ms"], test["_train_speed"]
    speed_limits = next(row for row in coverage if row["channel"] == "Vx_ms")
    extrapolative = (test_speed < speed_limits["train_min"]) | (test_speed > speed_limits["train_max"])
    axes[1].scatter(train_speed, test["_train_pressure"], s=1, alpha=.08, label="Run 4 train")
    axes[1].scatter(test_speed[~extrapolative], test["P_kPa"][~extrapolative], s=1, alpha=.08, label="Run 6 in range")
    axes[1].scatter(test_speed[extrapolative], test["P_kPa"][extrapolative], s=2, alpha=.3, color="crimson", label="Run 6 speed extrapolation")
    axes[1].set(xlabel="Belt speed [m/s]", ylabel="Pressure [kPa]", title="Explicit speed extrapolation")
    axes[1].legend(markerscale=4)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(path / f"fig_ttc_pacejka_operating_range.{ext}", dpi=300)
    plt.close(fig)

    # Curves make the force behaviour visible without implying that the
    # high-speed held-out points are interpolative.  Each curve is conditional
    # on a fixed normal-load band; sparse bins are left blank rather than filled.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    alpha_edges = np.linspace(-13, 13, 27)
    for axis, (lo, hi) in zip(axes, ((350, 550), (750, 1000))):
        selected = (fz >= lo) & (fz < hi)
        centers, measured_curve, predicted_curve = [], [], []
        for left, right in zip(alpha_edges[:-1], alpha_edges[1:]):
            in_bin = selected & (alpha_deg >= left) & (alpha_deg < right)
            if in_bin.sum() >= 20:
                centers.append((left + right) / 2)
                measured_curve.append(float(measured[in_bin].mean()))
                predicted_curve.append(float(fy_pred[in_bin].mean()))
        axis.plot(centers, measured_curve, "o-", label="Measured")
        axis.plot(centers, predicted_curve, "s--", label="Analytical Pacejka")
        axis.set(title=f"Run 6: $F_z$ {lo}–{hi} N", xlabel="Slip angle [deg]", ylabel="$F_y$ [N]")
        axis.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(path / f"fig_ttc_pacejka_fy_operating_range.{ext}", dpi=300)
    plt.close(fig)


def run_pilot(output: str | Path, ttc_path: Path | None = None) -> Path:
    ttc_path = ttc_path or ROOT / "data" / "ttc_round9" / "processed.npz"
    dataset, split = load_ttc(ttc_path), CompleteRunSplit()
    masks = split.masks(dataset["run"])
    adapter = AnalyticalPacejkaAdapter()
    fx_train, raw_fy_train = adapter.predict(dataset, masks["train"])
    sign = choose_force_sign(dataset["Fy_N"][masks["train"]], raw_fy_train)
    multiplier = sign["model_to_ttc_fy_multiplier"]
    predictions = {}
    for partition in ("train", "validation", "test"):
        fx, fy = (fx_train, raw_fy_train) if partition == "train" else adapter.predict(dataset, masks[partition])
        predictions[partition] = {"Fx_N": fx, "Fy_N": multiplier * fy}
    inputs = np.column_stack([dataset[name] for name in INPUT_CHANNELS])
    targets = np.column_stack([dataset[name] for name in TARGET_CHANNELS])
    input_scaler = TrainOnlyScaler.fit(inputs[masks["train"]])
    target_scaler = TrainOnlyScaler.fit(targets[masks["train"]])
    partitions = {name: {key: dataset[key][mask] for key in dataset} for name, mask in masks.items()}
    test_for_figures = dict(partitions["test"])
    test_for_figures.update({"_train_alpha": partitions["train"]["alpha_rad"], "_train_fz": partitions["train"]["Fz_N"],
                             "_train_speed": partitions["train"]["Vx_ms"], "_train_pressure": partitions["train"]["P_kPa"]})
    coverage = _coverage(partitions["train"], partitions["test"])
    result = {
        "status": "completed_lightweight_analytical_pilot",
        "model": adapter.name,
        "split": split.as_dict(),
        "input_mapping": {
            "alpha": "alpha_rad: TTC SA converted from degrees to radians during preprocessing", "kappa": "kappa: TTC SL; all stored values are zero",
            "Fz": "Fz_N: TTC FZ sign-flipped during preprocessing to positive loaded force [N]", "gamma": "gamma_rad: TTC IA converted degrees to radians without additional sign conversion",
            "Vx": "Vx_ms: TTC V converted km/h to m/s; required, no fallback", "T_ribs": "[T_inner_C, T_center_C, T_outer_C] [C]",
            "force_units_and_sign": "Fx_N and Fy_N are TTC SAE-convention forces in N, preserved by preprocessing. Fy alone receives the frozen Run-4 model-to-TTC multiplier; Fx receives no fitted sign mapping.",
            "pressure": "P_kPa is TTC pressure in kPa; it is retained for coverage diagnostics but unsupported by the selected model force interface.",
            "unmapped_archive_channels": "P_kPa and Mz_Nm are retained for diagnostics; the selected force path has no pressure input and does not predict Mz",
            "tire_interpretation": "single Hoosier 43075 16x7.5-10 R20 on 7 inch rim; no vehicle corner/front/rear or side label is available in these run files",
        },
        "equation_path": "models.tire_model.PacejkaTire.compute_force_and_sigma with apply_residual=False: MF6.2 pure forces Fx0/Fy0, combined-slip Fx=Fx0*Gxa and Fy=Fy0*Gyk, then the implemented zero-turn-rate turn-slip factor. PINN drift and GP-style penalty are bypassed exactly.",
        "fixed_auxiliary_inputs": FIXED_AUXILIARY_INPUTS,
        "sign_convention": sign,
        "metrics": {name: {"Fy": _metrics(partitions[name]["Fy_N"], predictions[name]["Fy_N"], partitions[name]["Fz_N"]),
                            "Fx_secondary_zero_kappa_diagnostic": _metrics(partitions[name]["Fx_N"], predictions[name]["Fx_N"], partitions[name]["Fz_N"])} for name in masks},
        "test_fy_strata": _binned_metrics(dataset, masks["test"], predictions["test"]["Fy_N"]),
        "test_operating_range_outside_train": coverage,
        "input_scaler": input_scaler.serializable(INPUT_CHANNELS), "target_scaler": target_scaler.serializable(TARGET_CHANNELS),
        "leakage_protection": "Only Run 4 was used to select the global Fy sign or fit scaler metadata. Runs 5 and 6 are evaluation-only; is_test was never read.",
        "limitations": "Fx has no longitudinal-slip support because kappa is zero. Gas temperature and turn rate are unavailable and fixed, pressure is not consumed by this force path, and TTC/model camber convention equivalence is assumed rather than independently verified.",
    }
    base, target = Path(output), Path(output) / "ttc_pacejka_baseline_pilot"; target.mkdir(parents=True, exist_ok=True)
    path = target / "ttc_pacejka_baseline_pilot.json"
    result["source_sha256"] = hashlib.sha256(ttc_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(jsonable(metadata(SuiteConfig(output=str(base)), "ttc_pacejka_baseline_pilot") | result), indent=2, sort_keys=True))
    rows = [{"partition": partition, "force": force, **metrics} for partition, values in result["metrics"].items() for force, metrics in values.items()]
    with (target / "ttc_pacejka_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    _figures(target, test_for_figures, predictions["test"]["Fy_N"], coverage)
    manifest(SuiteConfig(output=str(base)))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked analytical TTC Pacejka pilot.")
    parser.add_argument("--output", default=str(ROOT / "results" / "paper_suite")); parser.add_argument("--ttc-path", type=Path)
    args = parser.parse_args(); print(run_pilot(args.output, args.ttc_path))


if __name__ == "__main__":
    main()
