from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def plot_success_vs_bits(*args, **kwargs):
    raise NotImplementedError("plot_success_vs_bits is planned for Phase 4, see ROADMAP.md")
