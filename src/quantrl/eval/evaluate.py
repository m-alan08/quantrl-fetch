from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np

from quantrl.envs.normalizer import GoalObservationNormalizer
from quantrl.eval.stats import bootstrap_confidence_interval

PolicyFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class EvaluationResult:
    episodes: int
    success_rate: float
    success_ci_low: float
    success_ci_high: float
    mean_return: float
    return_ci_low: float
    return_ci_high: float
    mean_final_distance: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class EpisodeOutcome:
    seed: int
    success: float
    episode_return: float
    final_distance: float


def evaluation_seeds(seed_offset: int, episodes: int) -> list[int]:
    return [seed_offset + index for index in range(episodes)]


def run_episode(
    env: gym.Env,
    policy: PolicyFunction,
    normalizer: GoalObservationNormalizer,
    seed: int,
) -> EpisodeOutcome:
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    info: dict = {}
    terminated = truncated = False
    while not (terminated or truncated):
        inputs = normalizer(observation["observation"], observation["desired_goal"])
        action = policy(inputs[None])[0]
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
    final_distance = float(
        np.linalg.norm(observation["achieved_goal"] - observation["desired_goal"])
    )
    return EpisodeOutcome(seed, float(info["is_success"]), episode_return, final_distance)


def evaluate_policy(
    env: gym.Env,
    policy: PolicyFunction,
    normalizer: GoalObservationNormalizer,
    seeds: Sequence[int],
    bootstrap_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[EvaluationResult, list[EpisodeOutcome]]:
    outcomes = [run_episode(env, policy, normalizer, seed) for seed in seeds]
    successes = np.array([outcome.success for outcome in outcomes])
    returns = np.array([outcome.episode_return for outcome in outcomes])
    distances = np.array([outcome.final_distance for outcome in outcomes])
    success_low, success_high = bootstrap_confidence_interval(
        successes, bootstrap_resamples, confidence
    )
    return_low, return_high = bootstrap_confidence_interval(
        returns, bootstrap_resamples, confidence
    )
    result = EvaluationResult(
        episodes=len(outcomes),
        success_rate=float(successes.mean()),
        success_ci_low=success_low,
        success_ci_high=success_high,
        mean_return=float(returns.mean()),
        return_ci_low=return_low,
        return_ci_high=return_high,
        mean_final_distance=float(distances.mean()),
    )
    return result, outcomes
