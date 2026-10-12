from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics

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
    kwargs = {}
    if render_mode is not None:
        kwargs.update(render_mode=render_mode, width=render_width, height=render_height)
    if env_config.max_episode_steps is not None:
        kwargs["max_episode_steps"] = env_config.max_episode_steps
    return gym.make(env_config.env_id, **kwargs)


def env_dimensions(env: gym.Env) -> EnvDimensions:
    observation_space = env.observation_space
    return EnvDimensions(
        observation_dim=observation_space["observation"].shape[0],
        goal_dim=observation_space["desired_goal"].shape[0],
        action_dim=env.action_space.shape[0],
        episode_length=env.spec.max_episode_steps,
    )
