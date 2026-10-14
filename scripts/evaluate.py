from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantrl.checkpoint import deterministic_policy_function, load_policy
from quantrl.envs.factory import make_env
from quantrl.eval.evaluate import evaluate_policy, evaluation_seeds
from quantrl.eval.stats import aggregate_across_seeds


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints with deterministic actions on fixed evaluation seeds."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", default=None, help="Tidy per-seed CSV")
    parser.add_argument("--summary", default=None, help="Aggregate JSON across training seeds")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    rows = []
    for checkpoint_path in arguments.checkpoints:
        loaded = load_policy(checkpoint_path)
        config = loaded.config
        episodes = arguments.episodes or config.eval.episodes
        env = make_env(config.env)
        result, _ = evaluate_policy(
            env,
            deterministic_policy_function(loaded.actor),
            loaded.normalizer,
            evaluation_seeds(config.eval.seed_offset, episodes),
            config.eval.bootstrap_resamples,
            config.eval.confidence,
        )
        env.close()
        row = {
            "env_id": config.env.env_id,
            "name": config.name,
            "seed": config.seed,
            "weight_bits": config.quant.weight_bits,
            "activation_bits": config.quant.activation_bits,
            "method": config.quant.method,
            **result.to_dict(),
        }
        rows.append(row)
        print(json.dumps(row))
    frame = pd.DataFrame(rows)
    if arguments.out:
        Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(arguments.out, index=False)
    group_columns = ["env_id", "name", "method", "weight_bits", "activation_bits"]
    summary = pd.concat(
        [
            aggregate_across_seeds(frame, group_columns, metric)
            for metric in ("success_rate", "mean_return", "mean_final_distance")
        ],
        ignore_index=True,
    )
    print(summary.to_string(index=False))
    if arguments.summary:
        Path(arguments.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.summary).write_text(
            json.dumps(summary.to_dict(orient="records"), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
