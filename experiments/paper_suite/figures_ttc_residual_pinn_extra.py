"""Five additional figures for the locked TTC Run-6 residual experiment.

Reuses the restore-only loading logic from ``figures_ttc_residual_pinn`` --
this module never re-fits, re-selects, or re-normalizes anything either.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .common import ROOT
from .figures_ttc_residual_pinn import (
    C_PACEJKA,
    C_RESIDUAL,
    C_RANGE,
    _robust_limits,
    load_locked_artifacts,
    setup_matplotlib,
)


def figure_summary_bars(data: dict, output: Path) -> None:
    """Headline comparison: RMSE and R^2 across global / in-range / out-of-range."""
    final = data["final"]
    subsets = [("Global", "global"), ("In-range", "speed_interpolation"), ("Out-of-range", "speed_extrapolation")]
    labels = [label for label, _ in subsets]

    rmse_pace = [final[key]["analytical"]["rmse_N"] for _, key in subsets]
    rmse_corr = [final[key]["corrected"]["rmse_N"] for _, key in subsets]
    r2_pace = [final[key]["analytical"]["r2"] for _, key in subsets]
    r2_corr = [final[key]["corrected"]["r2"] for _, key in subsets]

    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)

    ax = axes[0]
    ax.bar(x - width / 2, rmse_pace, width, color=C_PACEJKA, label="Pacejka")
    ax.bar(x + width / 2, rmse_corr, width, color=C_RESIDUAL, label="Pacejka + residual NN")
    for xi, rp, rc in zip(x, rmse_pace, rmse_corr):
        pct = (rc - rp) / rp * 100
        color = "seagreen" if pct < 0 else "firebrick"
        ax.annotate(f"{pct:+.1f}%", (xi, max(rp, rc) + 25), ha="center", fontsize=8, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE [N]")
    ax.set_title("Lateral-force RMSE")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)

    ax = axes[1]
    ax.bar(x - width / 2, r2_pace, width, color=C_PACEJKA)
    ax.bar(x + width / 2, r2_corr, width, color=C_RESIDUAL)
    for xi, rp, rc in zip(x, r2_pace, r2_corr):
        ax.annotate(f"{rc - rp:+.3f}", (xi, max(rp, rc) + 0.02), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0, 1.0)
    ax.set_title(r"Coefficient of determination")

    fig.suptitle("Held-out Run 6: Pacejka vs. residual-corrected model", fontsize=11)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure_training_curves(data: dict, output: Path) -> None:
    """Run-5 validation RMSE by epoch for each candidate architecture."""
    pilot = data["pilot"]
    selected_name = pilot["selected"]["name"]

    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)

    palette = [C_PACEJKA, C_RESIDUAL, "#55A868", "#8172B2"]
    for i, cand in enumerate(pilot["candidates"]):
        name = cand["candidate"]["name"]
        history = cand["metrics"]["history"]
        epochs = [h["epoch"] for h in history]
        rmse = [h["validation_corrected_rmse_N"] for h in history]
        is_selected = name == selected_name
        ax.plot(
            epochs, rmse,
            color=palette[i % len(palette)],
            linewidth=2.2 if is_selected else 1.3,
            linestyle="-" if is_selected else "--",
            label=f"{name}" + (" (selected)" if is_selected else ""),
        )

    baseline_rmse = pilot["candidates"][0]["metrics"]["validation"]["analytical"]["rmse_N"]
    ax.axhline(baseline_rmse, color="0.35", linestyle=":", linewidth=1.2, label=f"Pacejka baseline ({baseline_rmse:.0f} N)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation corrected RMSE [N]")
    ax.set_title("Run-5 model selection across candidate architectures")
    ax.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _quantile_bins(values: np.ndarray, n_bins: int = 5, min_count: int = 200):
    """Equal-population bins for a continuous variable (unlike discrete speed setpoints)."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    masks, centers = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values <= hi if hi == edges[-1] else values < hi)
        if mask.sum() < min_count:
            continue
        masks.append(mask)
        centers.append(float(values[mask].mean()))
    return masks, centers


def figure_error_vs_load(data: dict, output: Path) -> None:
    """Absolute error distribution by normal load, quantile-binned."""
    test = data["test"]
    fz = np.asarray(test["Fz_N"], dtype=float)
    measured = np.asarray(test["Fy_N"], dtype=float)
    pacejka = np.asarray(data["baseline"], dtype=float)
    corrected = np.asarray(data["corrected"], dtype=float)

    pace_err = np.abs(pacejka - measured)
    corr_err = np.abs(corrected - measured)

    masks, centers = _quantile_bins(fz, n_bins=5, min_count=200)
    counts = [int(m.sum()) for m in masks]
    pace_groups = [pace_err[m] for m in masks]
    corr_groups = [corr_err[m] for m in masks]

    positions = np.arange(len(masks))
    width = 0.34
    gap = 0.02

    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)

    bp1 = ax.boxplot(pace_groups, positions=positions - width / 2 - gap, widths=width, patch_artist=True, showfliers=False, whis=(10, 90))
    bp2 = ax.boxplot(corr_groups, positions=positions + width / 2 + gap, widths=width, patch_artist=True, showfliers=False, whis=(10, 90))

    for box_dict, color in ((bp1, C_PACEJKA), (bp2, C_RESIDUAL)):
        for patch in box_dict["boxes"]:
            patch.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=1.1)
        for key in ("whiskers", "caps"):
            for line in box_dict[key]:
                line.set(color=color, linewidth=1.1)
        for line in box_dict["medians"]:
            line.set(color="0.15", linewidth=1.3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c:.0f} N\n(n={n:,})" for c, n in zip(centers, counts)])
    ax.set_xlim(-0.5, len(masks) - 0.5)
    ax.set_xlabel(r"Normal load $F_z$ [N] (quantile bin)")
    ax.set_ylabel(r"Absolute $F_y$ error [N]")
    ax.set_title("Held-out Run 6: error distribution by normal load")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.2)

    legend_handles = [
        Patch(facecolor=C_PACEJKA, edgecolor=C_PACEJKA, alpha=0.55, label="Pacejka"),
        Patch(facecolor=C_RESIDUAL, edgecolor=C_RESIDUAL, alpha=0.55, label="Pacejka + residual NN"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.16))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure_residual_vs_load(data: dict, output: Path) -> None:
    """Binned learned residual correction versus normal load."""
    test = data["test"]
    fz = np.asarray(test["Fz_N"], dtype=float)
    residual = np.asarray(data["residual"], dtype=float)

    bins = np.linspace(np.nanpercentile(fz, 0.5), np.nanpercentile(fz, 99.5), 40)
    centers = 0.5 * (bins[:-1] + bins[1:])

    medians, lower, upper = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (fz >= lo) & (fz < hi)
        if mask.sum() < 30:
            medians.append(np.nan); lower.append(np.nan); upper.append(np.nan)
            continue
        values = residual[mask]
        medians.append(np.median(values))
        lower.append(np.percentile(values, 10))
        upper.append(np.percentile(values, 90))

    medians, lower, upper = map(np.asarray, (medians, lower, upper))
    valid = np.isfinite(medians)

    fig, ax = plt.subplots(figsize=(7.0, 3.1), constrained_layout=True)
    ax.fill_between(centers[valid], lower[valid], upper[valid], color=C_RESIDUAL, alpha=0.20, label="10\u201390th percentile")
    ax.plot(centers[valid], medians[valid], color=C_RESIDUAL, linewidth=1.8, label="Median learned residual")
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color="0.35")

    ax.set_xlabel(r"Normal load $F_z$ [N]")
    ax.set_ylabel(r"Learned residual $\Delta F_y$ [N]")
    ax.set_title("Learned residual correction vs. normal load, held-out Run 6")
    ax.legend(frameon=False, loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure_error_distribution(data: dict, output: Path) -> None:
    """Signed-error histograms: bias and spread, Pacejka vs. residual-corrected."""
    test = data["test"]
    measured = np.asarray(test["Fy_N"], dtype=float)
    pacejka = np.asarray(data["baseline"], dtype=float)
    corrected = np.asarray(data["corrected"], dtype=float)

    pace_err = pacejka - measured
    corr_err = corrected - measured

    lo, hi = _robust_limits(pace_err, corr_err, percentile=(0.5, 99.5))
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    ax.hist(pace_err, bins=bins, color=C_PACEJKA, alpha=0.55, density=True,
            label=f"Pacejka (bias = {pace_err.mean():.1f} N)")
    ax.hist(corr_err, bins=bins, color=C_RESIDUAL, alpha=0.55, density=True,
            label=f"Pacejka + residual NN (bias = {corr_err.mean():.1f} N)")
    ax.axvline(0.0, color="0.2", linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"Signed $F_y$ error (prediction $-$ measured) [N]")
    ax.set_ylabel("Density")
    ax.set_title("Held-out Run 6: signed error distribution")
    ax.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5 additional figures for the locked TTC residual Run-6 experiment.")
    parser.add_argument("--pilot", type=Path, default=ROOT / "results" / "paper_suite" / "ttc_residual_pinn_pilot" / "ttc_residual_pinn_pilot.json")
    parser.add_argument("--final", type=Path, default=ROOT / "results" / "paper_suite" / "ttc_residual_pinn_final" / "ttc_residual_pinn_final.json")
    parser.add_argument("--ttc-path", type=Path, default=ROOT / "data" / "ttc_round9" / "processed.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "ttc_residual_pinn")
    args = parser.parse_args()

    setup_matplotlib()
    data = load_locked_artifacts(args.pilot, args.final, args.ttc_path)

    figure_summary_bars(data, args.output / "run6_summary_bars.pdf")
    figure_training_curves(data, args.output / "run6_training_curves.pdf")
    figure_error_vs_load(data, args.output / "run6_error_vs_load.pdf")
    figure_residual_vs_load(data, args.output / "run6_residual_vs_load.pdf")
    figure_error_distribution(data, args.output / "run6_error_distribution.pdf")

    print(f"Figures written to: {args.output}")


if __name__ == "__main__":
    main()