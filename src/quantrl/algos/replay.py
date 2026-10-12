from __future__ import annotations

from collections.abc import Callable

import numpy as np

RewardFunction = Callable[[np.ndarray, np.ndarray, dict], np.ndarray]


def future_goal_probability(replay_k: int) -> float:
    return 1.0 - 1.0 / (1.0 + replay_k)


class HindsightReplayBuffer:
    def __init__(
        self,
        capacity_episodes: int,
        episode_length: int,
        observation_dim: int,
        goal_dim: int,
        action_dim: int,
        reward_function: RewardFunction,
        replay_k: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self.capacity = capacity_episodes
        self.episode_length = episode_length
        self.reward_function = reward_function
        self.future_probability = future_goal_probability(replay_k)
        self.rng = rng if rng is not None else np.random.default_rng()
        steps = episode_length
        self.observations = np.zeros((capacity_episodes, steps + 1, observation_dim), np.float32)
        self.achieved_goals = np.zeros((capacity_episodes, steps + 1, goal_dim), np.float32)
        self.desired_goals = np.zeros((capacity_episodes, steps, goal_dim), np.float32)
        self.actions = np.zeros((capacity_episodes, steps, action_dim), np.float32)
        self.next_index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def store_episode(
        self,
        observations: np.ndarray,
        achieved_goals: np.ndarray,
        desired_goals: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        index = self.next_index
        self.observations[index] = observations
        self.achieved_goals[index] = achieved_goals
        self.desired_goals[index] = desired_goals
        self.actions[index] = actions
        self.next_index = (self.next_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode_indices = self.rng.integers(0, self.size, size=batch_size)
        time_indices = self.rng.integers(0, self.episode_length, size=batch_size)
        relabel_mask = self.rng.random(batch_size) < self.future_probability
        steps_remaining = self.episode_length - time_indices
        future_offsets = np.floor(self.rng.random(batch_size) * steps_remaining).astype(np.int64)
        future_indices = np.where(relabel_mask, time_indices + 1 + future_offsets, -1)
        return episode_indices, time_indices, future_indices

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        episode_indices, time_indices, future_indices = self.sample_indices(batch_size)
        goals = self.desired_goals[episode_indices, time_indices].copy()
        relabeled = future_indices >= 0
        goals[relabeled] = self.achieved_goals[
            episode_indices[relabeled], future_indices[relabeled]
        ]
        next_achieved_goals = self.achieved_goals[episode_indices, time_indices + 1]
        rewards = self.reward_function(next_achieved_goals, goals, {})
        return {
            "observations": self.observations[episode_indices, time_indices],
            "next_observations": self.observations[episode_indices, time_indices + 1],
            "goals": goals,
            "actions": self.actions[episode_indices, time_indices],
            "rewards": np.asarray(rewards, dtype=np.float32),
            "relabeled": relabeled,
        }
