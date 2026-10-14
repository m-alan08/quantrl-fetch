from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def load_progress(run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        progress = pd.read_csv(Path(run_dir) / "progress.csv")
        progress["run"] = Path(run_dir).name
        frames.append(progress)
    return pd.concat(frames, ignore_index=True)


def plot_training_curves(
    progress: pd.DataFrame,
    output_path: str | Path,
    title: str,
    metrics: Sequence[tuple[str, str]] = (
        ("eval_success_rate", "Eval success rate"),
        ("eval_mean_return", "Eval mean return"),
    ),
) -> None:
    apply_style()
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 3.8))
    axes = [axes] if len(metrics) == 1 else list(axes)
    grouped = progress.groupby("step")
    steps = grouped.size().index.to_numpy()
    n_runs = progress["run"].nunique()
    for axis, (column, label) in zip(axes, metrics, strict=True):
        for index, (_, run_frame) in enumerate(progress.groupby("run")):
            axis.plot(
                run_frame["step"],
                run_frame[column],
                color=PALETTE[0],
                alpha=0.25,
                linewidth=1,
                label="individual seeds" if index == 0 else None,
            )
        mean = grouped[column].mean().to_numpy()
        low = grouped[column].min().to_numpy()
        high = grouped[column].max().to_numpy()
        axis.fill_between(steps, low, high, color=PALETTE[0], alpha=0.15, linewidth=0)
        axis.plot(steps, mean, color=PALETTE[0], linewidth=2, label=f"mean of {n_runs} seeds")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(label)
        if column == "eval_success_rate":
            axis.set_ylim(-0.03, 1.03)
    axes[0].legend(loc="lower right")
    figure.suptitle(title)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def bit_width_label(bits: int) -> str:
    return "fp32" if bits >= 32 else str(bits)


def plot_success_vs_bits(
    per_seed: pd.DataFrame,
    output_path: str | Path,
    title: str,
    series_column: str = "series",
    series_order: Sequence[str] | None = None,
    metrics: Sequence[tuple[str, str, float]] = (
        ("success_rate", "Success rate", 1.0),
        ("mean_final_distance", "Final distance to goal (cm)", 100.0),
    ),
    success_radius_cm: float | None = 5.0,
    log_scale_columns: Sequence[str] = ("mean_final_distance",),
) -> None:
    apply_style()
    bit_widths = sorted(per_seed["bits"].unique(), reverse=True)
    positions = {bits: index for index, bits in enumerate(bit_widths)}
    series_names = list(series_order or sorted(per_seed[series_column].unique()))
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 3.9))
    axes = [axes] if len(metrics) == 1 else list(axes)
    n_seeds = per_seed["seed"].nunique()
    for axis, (column, label, multiplier) in zip(axes, metrics, strict=True):
        for color, series in zip(PALETTE, series_names, strict=False):
            frame = per_seed[per_seed[series_column] == series]
            grouped = frame.groupby("bits")[column]
            bits = sorted(grouped.groups, reverse=True)
            x = [positions[value] for value in bits]
            mean = grouped.mean().reindex(bits).to_numpy() * multiplier
            low = grouped.min().reindex(bits).to_numpy() * multiplier
            high = grouped.max().reindex(bits).to_numpy() * multiplier
            axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            axis.plot(x, mean, color=color, linewidth=2, marker="o", markersize=4, label=series)
        axis.set_xticks(range(len(bit_widths)))
        axis.set_xticklabels([bit_width_label(bits) for bits in bit_widths])
        axis.set_xlabel("Bit width")
        axis.set_ylabel(label)
        if column == "success_rate":
            axis.set_ylim(-0.03, 1.03)
        if column in log_scale_columns:
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.set_ylabel(label.replace(" to goal (cm)", " (cm, log scale)"))
        if column == "mean_final_distance" and success_radius_cm is not None:
            axis.axhline(success_radius_cm, color="#6b7280", linestyle="--", linewidth=1)
            axis.annotate(
                "success radius",
                (0, success_radius_cm),
                xytext=(2, 3),
                textcoords="offset points",
                color="#6b7280",
                fontsize=9,
            )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(f"{title}\nlines: mean of {n_seeds} training seeds, shading: min to max")
    figure.tight_layout(rect=(0, 0.06 * ((len(labels) + 2) // 3), 1, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
