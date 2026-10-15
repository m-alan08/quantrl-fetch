from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quantrl.sweep import baseline_rows_for_methods
from quantrl.viz.plots import load_progress, plot_success_vs_bits, plot_training_curves

TARGET_LABELS = {
    "both": "weights and activations",
    "weights": "weights only",
    "activations": "activations only",
}
GRANULARITY_LABELS = {"per_tensor": "per-tensor", "per_channel": "per-channel"}
METHOD_LABELS = {
    "ptq": "PTQ",
    "qat_finetune": "QAT fine-tune",
    "qat_scratch": "QAT from scratch",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves or bit-width sweeps.")
    parser.add_argument("--runs", nargs="+", default=None, help="Run directories for curves")
    parser.add_argument("--ptq-csv", default=None, help="Per-seed CSV from evaluate_ptq.py")
    parser.add_argument("--sweep-csv", default=None, help="Per-seed CSV from sweep.py")
    parser.add_argument(
        "--targets", nargs="+", default=None, help="Quantization targets to include"
    )
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--title", default="")
    parser.add_argument("--csv", default=None, help="Optional path for the combined progress CSV")
    return parser.parse_args()


def series_label(row: pd.Series, show_grid: bool = False) -> str:
    target = TARGET_LABELS[row["target"]]
    if row["target"] == "activations":
        label = target
    else:
        label = f"{target}, {GRANULARITY_LABELS[row['weight_granularity']]} weights"
    if show_grid and row["target"] != "weights":
        label += f", {row['relu_activation_grid']} ReLU grid"
    return label


def plot_ptq(arguments: argparse.Namespace) -> None:
    frame = pd.read_csv(arguments.ptq_csv)
    if arguments.targets:
        frame = frame[frame["target"].isin(arguments.targets)]
    show_grid = (
        "relu_activation_grid" in frame.columns and frame["relu_activation_grid"].nunique() > 1
    )
    frame = frame.assign(series=frame.apply(series_label, axis=1, show_grid=show_grid))
    present = list(dict.fromkeys(frame["series"]))
    plot_success_vs_bits(frame, arguments.out, arguments.title, "series", present)


def plot_sweep(arguments: argparse.Namespace) -> None:
    frame = pd.read_csv(arguments.sweep_csv)
    methods = [method for method in METHOD_LABELS if method in set(frame["method"])]
    records = [{str(key): value for key, value in row.items()} for row in frame.to_dict("records")]
    expanded = pd.DataFrame(baseline_rows_for_methods(records, methods))
    expanded = expanded.assign(series=expanded["method"].map(METHOD_LABELS))
    order = [METHOD_LABELS[method] for method in methods]
    plot_success_vs_bits(expanded, arguments.out, arguments.title, "series", order)


def main() -> None:
    arguments = parse_arguments()
    if arguments.sweep_csv:
        plot_sweep(arguments)
    elif arguments.ptq_csv:
        plot_ptq(arguments)
    elif arguments.runs:
        progress = load_progress(arguments.runs)
        plot_training_curves(progress, arguments.out, arguments.title or "Training curves")
        if arguments.csv:
            Path(arguments.csv).parent.mkdir(parents=True, exist_ok=True)
            progress.to_csv(arguments.csv, index=False)
    else:
        raise SystemExit("Pass --runs, --ptq-csv, or --sweep-csv")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
