from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quantrl.viz.plots import load_progress, plot_success_vs_bits, plot_training_curves

TARGET_LABELS = {
    "both": "weights and activations",
    "weights": "weights only",
    "activations": "activations only",
}
GRANULARITY_LABELS = {"per_tensor": "per-tensor", "per_channel": "per-channel"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves or bit-width sweeps.")
    parser.add_argument("--runs", nargs="+", default=None, help="Run directories for curves")
    parser.add_argument("--ptq-csv", default=None, help="Per-seed CSV from evaluate_ptq.py")
    parser.add_argument(
        "--targets", nargs="+", default=None, help="Quantization targets to include"
    )
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--title", default="")
    parser.add_argument("--csv", default=None, help="Optional path for the combined progress CSV")
    return parser.parse_args()


def series_label(row: pd.Series) -> str:
    target = TARGET_LABELS[row["target"]]
    if row["target"] == "activations":
        return target
    return f"{target}, {GRANULARITY_LABELS[row['weight_granularity']]} weights"


def plot_ptq(arguments: argparse.Namespace) -> None:
    frame = pd.read_csv(arguments.ptq_csv)
    if arguments.targets:
        frame = frame[frame["target"].isin(arguments.targets)]
    frame = frame.assign(series=frame.apply(series_label, axis=1))
    order = [
        f"{TARGET_LABELS[target]}, {GRANULARITY_LABELS[granularity]} weights"
        for target in ("both", "weights")
        for granularity in ("per_tensor", "per_channel")
    ] + [TARGET_LABELS["activations"]]
    present = [name for name in order if name in set(frame["series"])]
    plot_success_vs_bits(frame, arguments.out, arguments.title, "series", present)


def main() -> None:
    arguments = parse_arguments()
    if arguments.ptq_csv:
        plot_ptq(arguments)
    elif arguments.runs:
        progress = load_progress(arguments.runs)
        plot_training_curves(progress, arguments.out, arguments.title or "Training curves")
        if arguments.csv:
            Path(arguments.csv).parent.mkdir(parents=True, exist_ok=True)
            progress.to_csv(arguments.csv, index=False)
    else:
        raise SystemExit("Pass --runs for training curves or --ptq-csv for a bit-width plot")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
