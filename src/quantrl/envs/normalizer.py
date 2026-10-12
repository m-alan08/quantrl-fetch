from __future__ import annotations

from typing import Any

import numpy as np


class RunningNormalizer:
    def __init__(self, size: int, eps: float = 0.01, clip: float = 5.0):
        self.size = size
        self.eps = eps
        self.clip = clip
        self.total = np.zeros(size, dtype=np.float64)
        self.total_squared = np.zeros(size, dtype=np.float64)
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.std = np.ones(size, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1, self.size)
        self.total += flat.sum(axis=0)
        self.total_squared += np.square(flat).sum(axis=0)
        self.count += flat.shape[0]
        self.mean = self.total / self.count
        variance = self.total_squared / self.count - np.square(self.mean)
        self.std = np.sqrt(np.maximum(np.square(self.eps), variance))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(values, dtype=np.float64) - self.mean) / self.std
        return np.clip(normalized, -self.clip, self.clip)

    def state_dict(self) -> dict[str, Any]:
        return {
            "total": self.total.copy(),
            "total_squared": self.total_squared.copy(),
            "count": self.count,
            "mean": self.mean.copy(),
            "std": self.std.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total = np.asarray(state["total"], dtype=np.float64).copy()
        self.total_squared = np.asarray(state["total_squared"], dtype=np.float64).copy()
        self.count = int(state["count"])
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.std = np.asarray(state["std"], dtype=np.float64).copy()


class GoalObservationNormalizer:
    def __init__(
        self,
        observation_dim: int,
        goal_dim: int,
        obs_clip: float = 200.0,
        normalized_clip: float = 5.0,
        eps: float = 0.01,
    ):
        self.obs_clip = obs_clip
        self.observation = RunningNormalizer(observation_dim, eps, normalized_clip)
        self.goal = RunningNormalizer(goal_dim, eps, normalized_clip)

    @property
    def input_dim(self) -> int:
        return self.observation.size + self.goal.size

    def _clip_raw(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, -self.obs_clip, self.obs_clip)

    def update(self, observations: np.ndarray, goals: np.ndarray) -> None:
        self.observation.update(self._clip_raw(observations))
        self.goal.update(self._clip_raw(goals))

    def __call__(self, observations: np.ndarray, goals: np.ndarray) -> np.ndarray:
        normalized_observations = self.observation.normalize(self._clip_raw(observations))
        normalized_goals = self.goal.normalize(self._clip_raw(goals))
        return np.concatenate([normalized_observations, normalized_goals], axis=-1).astype(
            np.float32
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "obs_clip": self.obs_clip,
            "observation": self.observation.state_dict(),
            "goal": self.goal.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.obs_clip = float(state["obs_clip"])
        self.observation.load_state_dict(state["observation"])
        self.goal.load_state_dict(state["goal"])
