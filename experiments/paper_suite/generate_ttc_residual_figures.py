"""Generate publication figures for the locked TTC Run-6 residual experiment.

This script performs inference only. It restores:
    - the selected Run-4/Run-5 checkpoint,
    - the Run-4-derived input scaler,
    - the Run-4-derived residual scaler,
    - the frozen Pacejka sign convention.

It never fits, selects, normalizes, or retrains anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .common import ROOT
from .ttc_benchmark import TrainOnlyScaler
from .ttc_pacejka_baseline import AnalyticalPacejkaAdapter
from .ttc_residual_pinn import (
    Candidate,
    TTCAdditiveResidualPINN,
    _feature_matrix,
    _load_roles,
    _predict,
)

# Consistent palette across every figure in this module.
C_PACEJKA = "#4C72B0"
C_RESIDUAL = "#DD8452"
C_RANGE = "0.85"


def load_locked_artifacts(pilot_path: Path, final_path: Path, ttc_path: Path):
    """Restore the exact locked experiment artifacts. Inference only."""
    pilot = json.loads(pilot_path.read_text())
    final = json.loads(final_path.read_text())

    if pilot.get("status") != "completed_train_run4_validation_run5_only":
        raise ValueError("Pilot artifact is not an accepted locked Run-4/Run-5 artifact.")
    if final.get("status") != "completed_locked_run6_evaluation":
        raise ValueError("Final artifact is not a completed locked Run-6 evaluation.")

    expected_hash = pilot["provenance"]["source_sha256"]
    actual_hash = hashlib.sha256(ttc_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("TTC source hash does not match the frozen pilot source hash.")
    if pilot["provenance"].get("run6_used_for_fitting_or_selection", False):
        raise ValueError("Pilot provenance indicates Run 6 influenced fitting/selection.")

    selected_name = pilot["selected"]["name"]
    selected = next(item for item in pilot["candidates"] if item["candidate"]["name"] == selected_name)
    candidate = Candidate(**selected["candidate"])

    model = TTCAdditiveResidualPINN(candidate.widths)
    template = model.init(
        jax.random.PRNGKey(candidate.seed),
        jnp.zeros(len(pilot["provenance"]["features"]["names"])),
    )
    checkpoint_path = pilot_path.parent / selected["checkpoint"]
    params = flax.serialization.from_bytes(template, checkpoint_path.read_bytes())

    roles, _ = _load_roles(ttc_path, ("test",))
    test = roles["test"]

    sign = pilot["provenance"]["frozen_sign"]["model_to_ttc_fy_multiplier"]
    adapter = AnalyticalPacejkaAdapter()
    baseline = sign * adapter.predict(test, np.ones(len(test["Fy_N"]), dtype=bool))[1]

    input_scaler_data = pilot["provenance"]["input_scaler"]
    input_scaler = TrainOnlyScaler(
        np.asarray(input_scaler_data["center"]),
        np.asarray(input_scaler_data["scale"]),
        tuple(input_scaler_data["constant_fields"]),
    )

    residual_scaler_data = pilot["provenance"]["residual_scaler"]
    center = residual_scaler_data["center"][0]
    scale = residual_scaler_data["scale"][0]

    x = input_scaler.transform(_feature_matrix(test)).astype(np.float32)
    residual = _predict(model, params, x) * scale + center
    corrected = baseline + residual

    thresholds = pilot["provenance"]["coverage_thresholds"]
    vmin, vmax = thresholds["Vx_ms"]["min"], thresholds["Vx_ms"]["max"]
    interpolation_mask = (test["Vx_ms"] >= vmin) & (test["Vx_ms"] <= vmax)

    return {
        "test": test,
        "baseline": baseline,
        "corrected": corrected,
        "residual": residual,
        "interpolation_mask": interpolation_mask,
        "vmin": vmin,
        "vmax": vmax,
        "pilot": pilot,
        "final": final,
    }


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _robust_limits(*arrays: np.ndarray, percentile: tuple[float, float] = (0.5, 99.5)):
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    lo, hi = np.percentile(values, percentile)
    half = max(abs(lo), abs(hi))
    return -half, half


def figure_predicted_vs_measured(data: dict, output: Path) -> None:
    """Publication-quality Run-6 measured-vs-predicted density figure."""
    test = data["test"]
    measured = np.asarray(test["Fy_N"], dtype=float)
    pacejka = np.asarray(data["baseline"], dtype=float)
    corrected = np.asarray(data["corrected"], dtype=float)

    lo, hi = _robust_limits(measured, pacejka, corrected)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True, sharex=True, sharey=True)
    datasets = [("Pacejka", pacejka), ("Pacejka + residual NN", corrected)]
    hb = None

    for ax, (title, prediction) in zip(axes, datasets):
        hb = ax.hexbin(measured, prediction, gridsize=65, mincnt=1, bins="log", cmap="viridis", linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.35")

        error = prediction - measured
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        bias = np.mean(error)

        ax.text(
            0.04, 0.96,
            rf"RMSE = {rmse:.1f} N" "\n" rf"MAE = {mae:.1f} N" "\n" rf"Bias = {bias:.1f} N",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.75", alpha=0.92),
        )
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel(r"Predicted $F_y$ [N]")
    axes[0].set_xlabel(r"Measured $F_y$ [N]")
    axes[1].set_xlabel(r"Measured $F_y$ [N]")
    fig.suptitle("Held-out TTC Run 6", fontsize=11)

    if hb is not None:
        cbar = fig.colorbar(hb, ax=axes, location="right", shrink=0.90, pad=0.02)
        cbar.set_label("Sample density (log scale)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _cluster_speeds(speed: np.ndarray, min_gap: float = 0.5, min_count: int = 100):
    """Group samples into physically real speed setpoints, not decimal-rounding artifacts.

    TTC data has a handful of discrete target speeds with tiny continuous drift
    around each one. Sorting and splitting on genuine gaps (default > 0.5 m/s)
    recovers the true setpoints instead of exploding into dozens of near-duplicate
    rounded values.
    """
    speed = np.asarray(speed, dtype=float)
    order = np.argsort(speed)
    sorted_speed = speed[order]
    breaks = np.where(np.diff(sorted_speed) > min_gap)[0] + 1
    chunks = np.split(order, breaks)
    groups = [idx for idx in chunks if len(idx) >= min_count]
    groups.sort(key=lambda idx: speed[idx].mean())
    return groups


def figure_error_vs_speed(data: dict, output: Path) -> None:
    """Run-6 absolute-error distribution by real speed condition."""
    test = data["test"]
    speed = np.asarray(test["Vx_ms"], dtype=float)
    measured = np.asarray(test["Fy_N"], dtype=float)
    pacejka = np.asarray(data["baseline"], dtype=float)
    corrected = np.asarray(data["corrected"], dtype=float)
    vmin, vmax = data["vmin"], data["vmax"]

    pacejka_error = np.abs(pacejka - measured)
    corrected_error = np.abs(corrected - measured)

    groups = _cluster_speeds(speed, min_gap=0.5, min_count=100)
    centers = [float(speed[idx].mean()) for idx in groups]
    counts = [len(idx) for idx in groups]
    in_range = [vmin <= c <= vmax for c in centers]

    pace_groups = [pacejka_error[idx] for idx in groups]
    corr_groups = [corrected_error[idx] for idx in groups]

    positions = np.arange(len(groups))
    width = 0.34
    gap = 0.02

    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)

    for i, flag in enumerate(in_range):
        if flag:
            ax.axvspan(i - 0.5, i + 0.5, color=C_RANGE, zorder=0)

    bp1 = ax.boxplot(
        pace_groups, positions=positions - width / 2 - gap, widths=width,
        patch_artist=True, showfliers=False, whis=(10, 90), zorder=2,
    )
    bp2 = ax.boxplot(
        corr_groups, positions=positions + width / 2 + gap, widths=width,
        patch_artist=True, showfliers=False, whis=(10, 90), zorder=2,
    )

    for box_dict, color in ((bp1, C_PACEJKA), (bp2, C_RESIDUAL)):
        for patch in box_dict["boxes"]:
            patch.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=1.1)
        for key in ("whiskers", "caps"):
            for line in box_dict[key]:
                line.set(color=color, linewidth=1.1)
        for line in box_dict["medians"]:
            line.set(color="0.15", linewidth=1.3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c:.1f} m/s\n(n={n:,})" for c, n in zip(centers, counts)])
    ax.set_xlim(-0.5, len(groups) - 0.5)

    ax.set_ylabel(r"Absolute $F_y$ error [N]")
    ax.set_xlabel("Speed condition")
    ax.set_title("Held-out Run 6: error distribution by speed condition")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.2)

    legend_handles = [
        Patch(facecolor=C_PACEJKA, edgecolor=C_PACEJKA, alpha=0.55, label="Pacejka"),
        Patch(facecolor=C_RESIDUAL, edgecolor=C_RESIDUAL, alpha=0.55, label="Pacejka + residual NN"),
    ]
    if any(in_range):
        legend_handles.append(Patch(facecolor=C_RANGE, label="Run-4 training speed range"))
    ax.legend(handles=legend_handles, loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.16))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure_residual_vs_alpha(data: dict, output: Path) -> None:
    """Binned learned residual correction versus slip angle."""
    test = data["test"]
    alpha_deg = np.rad2deg(np.asarray(test["alpha_rad"], dtype=float))
    residual = np.asarray(data["residual"], dtype=float)

    bins = np.linspace(np.nanpercentile(alpha_deg, 0.5), np.nanpercentile(alpha_deg, 99.5), 45)
    centers = 0.5 * (bins[:-1] + bins[1:])

    medians, lower, upper, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (alpha_deg >= lo) & (alpha_deg < hi)
        if mask.sum() < 30:
            medians.append(np.nan); lower.append(np.nan); upper.append(np.nan); counts.append(0)
            continue
        values = residual[mask]
        medians.append(np.median(values))
        lower.append(np.percentile(values, 10))
        upper.append(np.percentile(values, 90))
        counts.append(int(mask.sum()))

    medians, lower, upper = map(np.asarray, (medians, lower, upper))
    valid = np.isfinite(medians)

    fig, ax = plt.subplots(figsize=(7.0, 3.1), constrained_layout=True)

    ax.fill_between(centers[valid], lower[valid], upper[valid], color=C_RESIDUAL, alpha=0.20, label="10\u201390th percentile")
    ax.plot(centers[valid], medians[valid], color=C_RESIDUAL, linewidth=1.8, label="Median learned residual")
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color="0.35")

    ax.set_xlabel(r"Slip angle $\alpha$ [deg]")
    ax.set_ylabel(r"Learned residual $\Delta F_y$ [N]")
    ax.set_title("Learned residual correction on held-out TTC Run 6")
    ax.legend(frameon=False, loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for the locked TTC residual Run-6 experiment.")
    parser.add_argument("--pilot", type=Path, default=ROOT / "results" / "paper_suite" / "ttc_residual_pinn_pilot" / "ttc_residual_pinn_pilot.json")
    parser.add_argument("--final", type=Path, default=ROOT / "results" / "paper_suite" / "ttc_residual_pinn_final" / "ttc_residual_pinn_final.json")
    parser.add_argument("--ttc-path", type=Path, default=ROOT / "data" / "ttc_round9" / "processed.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "ttc_residual_pinn")
    args = parser.parse_args()

    setup_matplotlib()
    data = load_locked_artifacts(args.pilot, args.final, args.ttc_path)

    figure_predicted_vs_measured(data, args.output / "run6_predicted_vs_measured.pdf")
    figure_error_vs_speed(data, args.output / "run6_error_vs_speed.pdf")
    figure_residual_vs_alpha(data, args.output / "run6_residual_vs_alpha.pdf")

    print(f"Figures written to: {args.output}")


if __name__ == "__main__":
    main()