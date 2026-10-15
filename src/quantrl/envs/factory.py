from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import gymnasium_robotics
import numpy as np

from quantrl.config import EnvConfig

gym.register_envs(gymnasium_robotics)


@dataclass(frozen=True)
class EnvDimensions:
    observation_dim: int
    goal_dim: int
    action_dim: int
    episode_length: int


def make_env(
    env_config: EnvConfig,
    render_mode: str | None = None,
    render_width: int = 480,
    render_height: int = 480,
) -> gym.Env:
    kwargs: dict[str, Any] = {}
    if render_mode is not None:
        kwargs.update(render_mode=render_mode, width=render_width, height=render_height)
    if env_config.max_episode_steps is not None:
        kwargs["max_episode_steps"] = env_config.max_episode_steps
    return gym.make(env_config.env_id, **kwargs)


GOAL_KEYS = ("observation", "achieved_goal", "desired_goal")


class EnvironmentContractError(RuntimeError):
    pass


def check_goal_env(env: gym.Env) -> None:
    observation_space = env.observation_space
    problems = []
    if not isinstance(observation_space, gym.spaces.Dict):
        problems.append(f"observation space must be a Dict, got {type(observation_space).__name__}")
    else:
        missing = [key for key in GOAL_KEYS if key not in observation_space.spaces]
        if missing:
            problems.append(f"observation space is missing keys {missing}")
        for key in GOAL_KEYS:
            space = observation_space.spaces.get(key)
            if space is not None and (
                not isinstance(space, gym.spaces.Box) or len(space.shape) != 1
            ):
                problems.append(f"observation[{key!r}] must be a 1-D Box, got {space}")
        achieved = observation_space.spaces.get("achieved_goal")
        desired = observation_space.spaces.get("desired_goal")
        if achieved is not None and desired is not None and achieved.shape != desired.shape:
            problems.append(
                f"achieved_goal {achieved.shape} and desired_goal {desired.shape} differ in shape"
            )
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1:
        problems.append(f"action space must be a 1-D Box, got {action_space}")
    elif not (np.all(action_space.low == -1.0) and np.all(action_space.high == 1.0)):
        problems.append(
            "action space must be bounded by -1 and 1 because the actor squashes with tanh, "
            f"got low {action_space.low} and high {action_space.high}"
        )
    if not callable(getattr(env.unwrapped, "compute_reward", None)):
        problems.append("environment has no compute_reward method, which HER needs")
    if env.spec is None or env.spec.max_episode_steps is None:
        problems.append("environment must have a time limit (max_episode_steps)")
    if problems:
        raise EnvironmentContractError(
            "Environment does not match the goal-conditioned contract:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def check_goal_observation(observation: dict, dimensions: EnvDimensions) -> None:
    expected = {
        "observation": (dimensions.observation_dim,),
        "achieved_goal": (dimensions.goal_dim,),
        "desired_goal": (dimensions.goal_dim,),
    }
    for key, shape in expected.items():
        value = np.asarray(observation[key])
        if value.shape != shape:
            raise EnvironmentContractError(
                f"observation[{key!r}] has shape {value.shape}, expected {shape}"
            )
        if not np.all(np.isfinite(value)):
            raise EnvironmentContractError(f"observation[{key!r}] contains NaN or inf: {value}")


def env_dimensions(env: gym.Env) -> EnvDimensions:
    observation_space = cast(gym.spaces.Dict, env.observation_space)
    action_space = cast(gym.spaces.Box, env.action_space)
    if env.spec is None or env.spec.max_episode_steps is None:
        raise EnvironmentContractError("environment must have a time limit (max_episode_steps)")
    observation_shape = observation_space["observation"].shape
    goal_shape = observation_space["desired_goal"].shape
    if observation_shape is None or goal_shape is None:
        raise EnvironmentContractError("observation and goal spaces must have fixed shapes")
    return EnvDimensions(
        observation_dim=int(observation_shape[0]),
        goal_dim=int(goal_shape[0]),
        action_dim=int(action_space.shape[0]),
        episode_length=int(env.spec.max_episode_steps),
    )
