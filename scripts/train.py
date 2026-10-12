from __future__ import annotations

import argparse
import json

from quantrl.config import load_config
from quantrl.training import train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC with HER on a Fetch task.")
    parser.add_argument("--config", default="configs/fetch_reach.json")
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    seeds = arguments.seed
    if seeds is None:
        seeds = [load_config(arguments.config, arguments.overrides).seed]
    for seed in seeds:
        config = load_config(arguments.config, [*arguments.overrides, f"seed={seed}"])
        summary = train(config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
