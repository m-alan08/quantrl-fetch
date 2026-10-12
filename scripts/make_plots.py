from __future__ import annotations

import argparse
from pathlib import Path

from quantrl.viz.plots import load_progress, plot_training_curves


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves across seeds.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--title", default="Training curves")
    parser.add_argument("--csv", default=None, help="Optional path for the combined progress CSV")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    progress = load_progress(arguments.runs)
    plot_training_curves(progress, arguments.out, arguments.title)
    if arguments.csv:
        Path(arguments.csv).parent.mkdir(parents=True, exist_ok=True)
        progress.to_csv(arguments.csv, index=False)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
