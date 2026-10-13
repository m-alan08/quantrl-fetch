from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantrl.checkpoint import load_policy
from quantrl.eval.quantized import QUANTIZATION_TARGETS, evaluate_ptq_settings, ptq_settings
from quantrl.eval.stats import aggregate_across_seeds

GROUP_COLUMNS = ["env_id", "name", "method", "bits", "target", "weight_granularity"]
SUMMARY_METRICS = ("success_rate", "mean_return", "mean_final_distance")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training quantization of trained actors at several bit widths."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[32, 8, 6, 5, 4, 3, 2])
    parser.add_argument(
        "--targets", nargs="+", default=["both"], choices=list(QUANTIZATION_TARGETS)
    )
    parser.add_argument(
        "--granularity",
        nargs="+",
        default=["per_tensor"],
        choices=["per_tensor", "per_channel"],
        help="Weight scale granularity",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", required=True, help="Tidy per-seed CSV")
    parser.add_argument("--summary", default=None, help="Aggregate CSV across training seeds")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    settings = ptq_settings(
        arguments.bits,
        arguments.targets,
        [granularity == "per_channel" for granularity in arguments.granularity],
    )
    rows = []
    for checkpoint_path in arguments.checkpoints:
        loaded = load_policy(checkpoint_path)
        for row in evaluate_ptq_settings(loaded, settings, arguments.episodes):
            rows.append(row)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "seed",
                            "bits",
                            "target",
                            "weight_granularity",
                            "success_rate",
                            "mean_final_distance",
                        )
                    }
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.out, index=False)
    summary = pd.concat(
        [aggregate_across_seeds(frame, GROUP_COLUMNS, metric) for metric in SUMMARY_METRICS],
        ignore_index=True,
    )
    if arguments.summary:
        summary.to_csv(arguments.summary, index=False)
    print(summary[summary.metric == "success_rate"].to_string(index=False))


if __name__ == "__main__":
    main()
