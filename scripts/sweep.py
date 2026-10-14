from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from quantrl.eval.stats import aggregate_across_seeds
from quantrl.sweep import METHODS, SweepPlan, run_sweep

GROUP_COLUMNS = ["env_id", "method", "bits", "weight_granularity"]
SUMMARY_METRICS = ("success_rate", "mean_return", "mean_final_distance")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bit width by method sweep: PTQ, QAT fine-tuning, and QAT from scratch."
    )
    parser.add_argument("--base-config", required=True, help="Config used for QAT from scratch")
    parser.add_argument("--fp32-runs", nargs="+", required=True, help="Trained fp32 run dirs")
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 5, 4, 3, 2])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--per-tensor", action="store_true", help="Use per-tensor weight scales")
    parser.add_argument("--finetune-set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--scratch-set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", required=True, help="Tidy per-seed CSV")
    parser.add_argument("--summary", required=True, help="Summary CSV across seeds")
    parser.add_argument("--timings", default=None, help="JSON with per-run training times")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    plan = SweepPlan(
        base_config=arguments.base_config,
        fp32_run_dirs=arguments.fp32_runs,
        bit_widths=arguments.bits,
        methods=arguments.methods,
        per_channel_weights=not arguments.per_tensor,
        output_dir=arguments.output_dir,
        finetune_overrides=arguments.finetune_set,
        scratch_overrides=arguments.scratch_set,
    )
    started = time.perf_counter()
    rows, training_summaries = run_sweep(plan, arguments.workers, arguments.episodes)
    frame = pd.DataFrame(rows).sort_values(
        ["method", "bits", "seed"], ascending=[True, False, True]
    )
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.out, index=False)
    summary = pd.concat(
        [aggregate_across_seeds(frame, GROUP_COLUMNS, metric) for metric in SUMMARY_METRICS],
        ignore_index=True,
    )
    Path(arguments.summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(arguments.summary, index=False)
    wall_seconds = time.perf_counter() - started
    if arguments.timings:
        Path(arguments.timings).parent.mkdir(parents=True, exist_ok=True)
        timings = {
            "sweep_wall_seconds": round(wall_seconds, 1),
            "workers": arguments.workers,
            "runs": sorted(
                (
                    {
                        "name": item["name"],
                        "seed": item["seed"],
                        "train_wall_seconds": item["train_wall_seconds"],
                        "total_steps": item["total_steps"],
                        "reused": item["reused"],
                    }
                    for item in training_summaries
                ),
                key=lambda item: (item["name"], item["seed"]),
            ),
        }
        Path(arguments.timings).write_text(json.dumps(timings, indent=2) + "\n")
    table = summary[summary.metric == "success_rate"].pivot_table(
        index="bits", columns="method", values="mean"
    )
    print(table.sort_index(ascending=False).round(3).to_string())
    print(f"sweep took {wall_seconds:.0f}s")


if __name__ == "__main__":
    main()
