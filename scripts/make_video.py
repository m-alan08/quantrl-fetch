from __future__ import annotations

import argparse

import numpy as np

from quantrl.checkpoint import deterministic_policy_function, load_policy
from quantrl.config import QuantConfig
from quantrl.envs.factory import make_env
from quantrl.eval.evaluate import evaluation_seeds
from quantrl.quant.ptq import collect_calibration_inputs, quantize_actor_post_training
from quantrl.viz.video import (
    check_same_environment,
    compose_side_by_side,
    render_policy_episodes,
    write_gif,
    write_video,
)

METHOD_LABELS = {
    "fp32": "fp32",
    "ptq": "PTQ",
    "qat_finetune": "QAT fine-tune",
    "qat_scratch": "QAT from scratch",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render evaluation episodes of a policy, or two policies side by side."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--ptq-bits",
        type=int,
        default=None,
        help="Show the checkpoint quantized with PTQ at this bit width instead of as trained",
    )
    parser.add_argument("--compare-checkpoint", default=None, help="Right-hand policy")
    parser.add_argument(
        "--compare-ptq-bits",
        type=int,
        default=None,
        help="Right-hand policy is the left checkpoint quantized with PTQ at this bit width",
    )
    parser.add_argument("--per-tensor", action="store_true", help="PTQ with per-tensor weights")
    parser.add_argument(
        "--relu-grid",
        default="unsigned",
        choices=["unsigned", "signed"],
        help="PTQ grid for activations that come out of a ReLU",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--first-eval-seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit env seeds")
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--gif", default=None, help="Optional output gif path")
    parser.add_argument("--gif-width", type=int, default=320)
    return parser.parse_args()


def policy_title(config, method: str | None = None, bits: int | None = None) -> str:
    method = method or config.quant.method
    bits = bits if bits is not None else config.quant.weight_bits
    if bits >= 32:
        return "fp32"
    return f"{bits}-bit {METHOD_LABELS.get(method, method)}"


def ptq_actor(loaded, bits: int, per_channel: bool, relu_grid: str):
    config = loaded.config
    env = make_env(config.env)
    try:
        calibration_inputs = collect_calibration_inputs(
            env,
            deterministic_policy_function(loaded.actor),
            loaded.normalizer,
            evaluation_seeds(config.eval.calibration_seed_offset, config.eval.calibration_episodes),
        )
    finally:
        env.close()
    quant_config = QuantConfig(
        method="ptq",
        weight_bits=bits,
        activation_bits=bits,
        per_channel_weights=per_channel,
        relu_activation_grid=relu_grid,
    )
    return quantize_actor_post_training(loaded.actor, config.sac, quant_config, calibration_inputs)


def main() -> None:
    arguments = parse_arguments()
    loaded = load_policy(arguments.checkpoint)
    config = loaded.config
    if arguments.seeds:
        seeds = arguments.seeds
    else:
        first_seed = arguments.first_eval_seed
        if first_seed is None:
            first_seed = config.eval.seed_offset
        seeds = evaluation_seeds(first_seed, arguments.episodes)

    left_actor = loaded.actor
    left_title = arguments.title or policy_title(config)
    if arguments.ptq_bits:
        left_actor = ptq_actor(
            loaded, arguments.ptq_bits, not arguments.per_tensor, arguments.relu_grid
        )
        left_title = arguments.title or policy_title(config, "ptq", arguments.ptq_bits)
    left_frames, left_outcomes = render_policy_episodes(
        config.env,
        deterministic_policy_function(left_actor),
        loaded.normalizer,
        seeds,
        left_title,
        size=arguments.size,
    )
    frames = left_frames
    message = f"left {left_title}: successes {left_outcomes}"

    if arguments.compare_checkpoint or arguments.compare_ptq_bits:
        if arguments.compare_checkpoint:
            compared = load_policy(arguments.compare_checkpoint)
            check_same_environment(config.env, compared.config.env)
            right_actor, right_normalizer = compared.actor, compared.normalizer
            right_title = policy_title(compared.config)
        else:
            right_actor = ptq_actor(
                loaded, arguments.compare_ptq_bits, not arguments.per_tensor, arguments.relu_grid
            )
            right_normalizer = loaded.normalizer
            right_title = policy_title(config, "ptq", arguments.compare_ptq_bits)
        right_frames, right_outcomes = render_policy_episodes(
            config.env,
            deterministic_policy_function(right_actor),
            right_normalizer,
            seeds,
            right_title,
            size=arguments.size,
        )
        frames = compose_side_by_side(left_frames, right_frames)
        message += f"; right {right_title}: successes {right_outcomes}"

    write_video(frames, arguments.out, fps=arguments.fps)
    print(f"wrote {arguments.out} ({len(frames)} frames, {np.asarray(frames[0]).shape})")
    print(message)
    if arguments.gif:
        write_gif(frames, arguments.gif, fps=arguments.fps, width=arguments.gif_width)
        print(f"wrote {arguments.gif}")


if __name__ == "__main__":
    main()
