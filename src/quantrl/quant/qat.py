from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from quantrl.algos.sac import SACAgent
from quantrl.checkpoint import build_actor_from_checkpoint, deterministic_policy_function
from quantrl.config import RunConfig, saved_config_from_dict
from quantrl.envs.normalizer import GoalObservationNormalizer
from quantrl.eval.evaluate import evaluation_seeds
from quantrl.quant.layers import activation_quantizers
from quantrl.quant.ptq import (
    calibrate_activation_ranges,
    collect_calibration_inputs,
    load_full_precision_weights,
)


class CheckpointMismatchError(ValueError):
    pass


def check_dimensions(checkpoint: dict[str, Any], dimensions: dict[str, int]) -> None:
    if checkpoint["dimensions"] != dimensions:
        raise CheckpointMismatchError(
            f"Checkpoint dimensions {checkpoint['dimensions']} do not match the "
            f"environment {dimensions}"
        )


def check_checkpoint_matches(
    checkpoint: dict[str, Any], config: RunConfig, dimensions: dict[str, int]
) -> None:
    saved = saved_config_from_dict(checkpoint["config"])
    problems = []
    if saved.env.env_id != config.env.env_id:
        problems.append(
            f"env.env_id is {saved.env.env_id!r} in the checkpoint, {config.env.env_id!r} now"
        )
    if saved.sac.hidden_sizes != config.sac.hidden_sizes:
        problems.append(
            f"sac.hidden_sizes is {saved.sac.hidden_sizes} in the checkpoint, "
            f"{config.sac.hidden_sizes} now"
        )
    if checkpoint["dimensions"] != dimensions:
        problems.append(
            f"dimensions are {checkpoint['dimensions']} in the checkpoint, {dimensions} now"
        )
    if problems:
        raise CheckpointMismatchError(
            "Checkpoint does not match the config:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def apply_configured_learning_rates(agent: SACAgent) -> None:
    configured = (
        (agent.actor_optimizer, agent.config.actor_lr),
        (agent.critic_optimizer, agent.config.critic_lr),
        (agent.alpha_optimizer, agent.config.alpha_lr),
    )
    for optimizer, learning_rate in configured:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def initialize_agent_from_checkpoint(
    agent: SACAgent,
    normalizer: GoalObservationNormalizer,
    checkpoint: dict[str, Any],
    config: RunConfig,
    env: gym.Env,
) -> int:
    agent_state = checkpoint["agent"]
    load_full_precision_weights(agent.actor, agent_state["actor"])
    load_full_precision_weights(agent.critic, agent_state["critic"])
    load_full_precision_weights(agent.critic_target, agent_state["critic_target"])
    with torch.no_grad():
        agent.log_alpha.copy_(agent_state["log_alpha"].to(agent.device))
    agent.actor_optimizer.load_state_dict(agent_state["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(agent_state["critic_optimizer"])
    agent.alpha_optimizer.load_state_dict(agent_state["alpha_optimizer"])
    apply_configured_learning_rates(agent)
    normalizer.load_state_dict(checkpoint["normalizer"])

    quantizers = [
        quantizer
        for module in (agent.actor, agent.critic, agent.critic_target)
        for quantizer in activation_quantizers(module)
        if quantizer.enabled
    ]
    if not quantizers:
        return 0
    source_actor = build_actor_from_checkpoint(checkpoint)
    calibration_inputs = collect_calibration_inputs(
        env,
        deterministic_policy_function(source_actor),
        normalizer,
        evaluation_seeds(config.eval.calibration_seed_offset, config.eval.calibration_episodes),
    )
    calibrate_activation_ranges(agent.actor, calibration_inputs)
    for quantizer in activation_quantizers(agent.actor):
        quantizer.unfreeze()
    return int(calibration_inputs.shape[0])
