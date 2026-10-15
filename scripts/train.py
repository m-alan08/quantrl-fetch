from __future__ import annotations

import argparse
import json

from quantrl.config import load_config
from quantrl.training import train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC with HER on a Fetch task.")
    parser.add_argument("--config", default="configs/fetch_reach.json")
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="auto (cuda, then mps, then cpu), cpu, cuda, cuda:<index>, or mps",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from resume_state.pt in the run directory if it exists",
    )
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.device is not None:
        arguments.overrides = [*arguments.overrides, f"device={arguments.device}"]
    if arguments.resume:
        arguments.overrides = [*arguments.overrides, "train.resume=true"]
    seeds = arguments.seed
    if seeds is None:
        seeds = [load_config(arguments.config, arguments.overrides).seed]
    for seed in seeds:
        config = load_config(arguments.config, [*arguments.overrides, f"seed={seed}"])
        summary = train(config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
