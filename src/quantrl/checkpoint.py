from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantrl.algos.networks import GaussianActor, LinearFactory
from quantrl.config import RunConfig, saved_config_from_dict
from quantrl.envs.normalizer import GoalObservationNormalizer
from quantrl.quant.layers import actor_linear_factory


@dataclass
class LoadedPolicy:
    config: RunConfig
    actor: GaussianActor
    normalizer: GoalObservationNormalizer
    dimensions: dict[str, int]
    metadata: dict[str, Any]


def save_checkpoint(
    path: Path,
    config: RunConfig,
    agent_state: dict[str, Any],
    normalizer: GoalObservationNormalizer,
    dimensions: dict[str, int],
    metadata: dict[str, Any],
    total_steps: int,
) -> None:
    torch.save(
        {
            "config": config.to_dict(),
            "agent": agent_state,
            "normalizer": normalizer.state_dict(),
            "dimensions": dimensions,
            "metadata": metadata,
            "total_steps": total_steps,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_actor_from_checkpoint(
    checkpoint: dict[str, Any], linear_factory: LinearFactory | None = None
) -> GaussianActor:
    config = saved_config_from_dict(checkpoint["config"])
    dimensions = checkpoint["dimensions"]
    if linear_factory is None:
        linear_factory = actor_linear_factory(config.quant)
    actor = GaussianActor(
        dimensions["observation_dim"] + dimensions["goal_dim"],
        dimensions["action_dim"],
        config.sac.hidden_sizes,
        config.sac.log_std_min,
        config.sac.log_std_max,
        linear_factory,
    )
    actor.load_state_dict(checkpoint["agent"]["actor"])
    actor.eval()
    return actor


def load_policy(path: str | Path) -> LoadedPolicy:
    checkpoint = load_checkpoint(path)
    config = saved_config_from_dict(checkpoint["config"])
    dimensions = checkpoint["dimensions"]
    normalizer = GoalObservationNormalizer(
        dimensions["observation_dim"],
        dimensions["goal_dim"],
        config.env.obs_clip,
        config.env.normalized_clip,
        config.env.normalizer_eps,
    )
    normalizer.load_state_dict(checkpoint["normalizer"])
    return LoadedPolicy(
        config=config,
        actor=build_actor_from_checkpoint(checkpoint),
        normalizer=normalizer,
        dimensions=dimensions,
        metadata=checkpoint["metadata"],
    )


def deterministic_policy_function(actor: GaussianActor) -> Callable[[np.ndarray], np.ndarray]:
    reference_parameter = next(actor.parameters())

    @torch.no_grad()
    def policy(inputs: np.ndarray) -> np.ndarray:
        tensor_inputs = torch.as_tensor(
            inputs, dtype=reference_parameter.dtype, device=reference_parameter.device
        )
        return actor.deterministic_action(tensor_inputs).float().cpu().numpy()

    return policy
