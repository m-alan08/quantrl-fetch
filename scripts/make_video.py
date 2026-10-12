from __future__ import annotations

import argparse

from quantrl.checkpoint import deterministic_policy_function, load_policy
from quantrl.eval.evaluate import evaluation_seeds
from quantrl.viz.video import render_policy_episodes, write_gif, write_video


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render evaluation episodes of a policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--first-eval-seed", type=int, default=None)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--gif", default=None, help="Optional output gif path")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    loaded = load_policy(arguments.checkpoint)
    config = loaded.config
    first_seed = arguments.first_eval_seed
    if first_seed is None:
        first_seed = config.eval.seed_offset
    seeds = evaluation_seeds(first_seed, arguments.episodes)
    bits = config.quant.weight_bits
    title = arguments.title or f"{config.env.env_id}  {'fp32' if bits == 32 else f'{bits}-bit'}"
    frames, outcomes = render_policy_episodes(
        config.env,
        deterministic_policy_function(loaded.actor),
        loaded.normalizer,
        seeds,
        title,
        size=arguments.size,
    )
    write_video(frames, arguments.out, fps=arguments.fps)
    print(f"wrote {arguments.out} ({len(frames)} frames, successes {outcomes})")
    if arguments.gif:
        write_gif(frames, arguments.gif, fps=arguments.fps)
        print(f"wrote {arguments.gif}")


if __name__ == "__main__":
    main()
