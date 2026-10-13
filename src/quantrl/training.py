from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantrl.algos.networks import LinearFactory
from quantrl.algos.replay import HindsightReplayBuffer
from quantrl.algos.sac import UPDATE_METRIC_NAMES, SACAgent
from quantrl.checkpoint import load_checkpoint, save_checkpoint
from quantrl.config import RunConfig, save_config
from quantrl.envs.factory import env_dimensions, make_env
from quantrl.envs.normalizer import GoalObservationNormalizer
from quantrl.eval.evaluate import evaluate_policy, evaluation_seeds
from quantrl.quant.layers import actor_linear_factory, critic_linear_factory
from quantrl.quant.qat import check_dimensions, initialize_agent_from_checkpoint
from quantrl.utils.run_info import collect_run_metadata
from quantrl.utils.seeding import resolve_device, seed_everything


class ProgressLog:
    def __init__(self, path: Path):
        self.path = path
        self.fieldnames: list[str] | None = None
        self.rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        new_columns = [key for key in row if self.fieldnames is None or key not in self.fieldnames]
        if self.fieldnames is None or new_columns:
            self.fieldnames = [*(self.fieldnames or []), *new_columns]
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, restval="")
                writer.writeheader()
                writer.writerows(self.rows)
            return
        with self.path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames, restval="").writerow(row)


class UpdateScheduler:
    def __init__(self, updates_per_step: float):
        self.updates_per_step = updates_per_step
        self.carry = 0.0

    def updates_for(self, environment_steps: int) -> int:
        self.carry += self.updates_per_step * environment_steps
        count = int(self.carry + 1e-9)
        self.carry -= count
        return count


def collect_episode(env, agent, normalizer, episode_length, use_random_actions):
    observation_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    observations = np.zeros((episode_length + 1, observation_dim), np.float32)
    achieved_goals = np.zeros((episode_length + 1, goal_dim), np.float32)
    desired_goals = np.zeros((episode_length, goal_dim), np.float32)
    actions = np.zeros((episode_length, action_dim), np.float32)
    observation, _ = env.reset()
    successes = 0.0
    for step in range(episode_length):
        observations[step] = observation["observation"]
        achieved_goals[step] = observation["achieved_goal"]
        desired_goals[step] = observation["desired_goal"]
        if use_random_actions:
            action = env.action_space.sample()
        else:
            inputs = normalizer(observation["observation"], observation["desired_goal"])
            action = agent.act(inputs[None])[0]
        actions[step] = action
        observation, _, _, _, info = env.step(action)
        successes = float(info["is_success"])
    observations[episode_length] = observation["observation"]
    achieved_goals[episode_length] = observation["achieved_goal"]
    return observations, achieved_goals, desired_goals, actions, successes


def build_update_batch(buffer, normalizer, batch_size):
    sample = buffer.sample(batch_size)
    return {
        "inputs": normalizer(sample["observations"], sample["goals"]),
        "next_inputs": normalizer(sample["next_observations"], sample["goals"]),
        "actions": sample["actions"],
        "rewards": sample["rewards"],
    }


def train(
    config: RunConfig,
    actor_factory: LinearFactory | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = seed_everything(config.seed)
    torch.set_num_threads(config.torch_threads)
    device = resolve_device(config.device)

    metadata = collect_run_metadata(config.seed)
    metadata["device"] = str(device)
    metadata["torch_threads"] = config.torch_threads
    save_config(config, run_dir / "config.json")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    env = make_env(config.env)
    eval_env = make_env(config.env)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)
    dimensions = env_dimensions(env)
    episode_length = dimensions.episode_length

    normalizer = GoalObservationNormalizer(
        dimensions.observation_dim,
        dimensions.goal_dim,
        config.env.obs_clip,
        config.env.normalized_clip,
        config.env.normalizer_eps,
    )
    buffer = HindsightReplayBuffer(
        capacity_episodes=config.her.buffer_episodes,
        episode_length=episode_length,
        observation_dim=dimensions.observation_dim,
        goal_dim=dimensions.goal_dim,
        action_dim=dimensions.action_dim,
        reward_function=env.unwrapped.compute_reward,
        replay_k=config.her.replay_k,
        rng=rng,
    )
    agent = SACAgent(
        normalizer.input_dim,
        dimensions.action_dim,
        config.sac,
        device,
        actor_linear_factory=actor_factory or actor_linear_factory(config.quant),
        critic_linear_factory=critic_linear_factory(config.quant),
    )
    calibration_inputs = 0
    if config.train.init_checkpoint:
        initial_checkpoint = load_checkpoint(config.train.init_checkpoint)
        check_dimensions(initial_checkpoint, asdict(dimensions))
        calibration_inputs = initialize_agent_from_checkpoint(
            agent, normalizer, initial_checkpoint, config, eval_env
        )
    metadata["init_checkpoint"] = config.train.init_checkpoint
    metadata["calibration_inputs"] = calibration_inputs
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def policy(inputs: np.ndarray) -> np.ndarray:
        return agent.act(inputs, deterministic=True)

    progress = ProgressLog(run_dir / "progress.csv")
    training_eval_seeds = evaluation_seeds(config.eval.seed_offset, config.train.eval_episodes)
    update_scheduler = UpdateScheduler(config.train.updates_per_step)
    total_steps = 0
    episodes = 0
    next_eval_step = config.train.eval_interval
    next_checkpoint_step = config.train.checkpoint_interval or None
    recent_metrics: dict[str, list[float]] = defaultdict(list)
    recent_train_success: list[float] = []
    start_time = time.perf_counter()
    eval_seconds = 0.0

    def checkpoint(path: Path) -> None:
        save_checkpoint(
            path,
            config,
            agent.state_dict(),
            normalizer,
            asdict(dimensions),
            metadata,
            total_steps,
        )

    def evaluate_and_log() -> None:
        nonlocal eval_seconds
        eval_start = time.perf_counter()
        result, _ = evaluate_policy(
            eval_env,
            policy,
            normalizer,
            training_eval_seeds,
            config.eval.bootstrap_resamples,
            config.eval.confidence,
        )
        eval_seconds += time.perf_counter() - eval_start
        row = {
            "step": total_steps,
            "episodes": episodes,
            "wall_seconds": round(time.perf_counter() - start_time, 2),
            "train_success": float(np.mean(recent_train_success))
            if recent_train_success
            else float("nan"),
            **{f"eval_{key}": value for key, value in result.to_dict().items()},
            **{
                key: float(np.mean(recent_metrics[key])) if recent_metrics[key] else float("nan")
                for key in UPDATE_METRIC_NAMES
            },
        }
        progress.write(row)
        if verbose:
            print(
                f"[{config.name} seed {config.seed}] step {total_steps:>7d} "
                f"eval success {result.success_rate:.2f} "
                f"return {result.mean_return:7.2f} "
                f"alpha {row['alpha']:.4f} "
                f"time {row['wall_seconds']:.0f}s",
                flush=True,
            )
        recent_metrics.clear()
        recent_train_success.clear()

    if config.train.eval_at_start:
        evaluate_and_log()

    while total_steps < config.train.total_steps:
        use_random_actions = total_steps < config.train.random_steps
        observations, achieved, desired, actions, success = collect_episode(
            env, agent, normalizer, episode_length, use_random_actions
        )
        buffer.store_episode(observations, achieved, desired, actions)
        normalizer.update(observations, np.concatenate([desired, achieved]))
        total_steps += episode_length
        episodes += 1
        recent_train_success.append(success)

        if total_steps >= config.train.update_after:
            for _ in range(update_scheduler.updates_for(episode_length)):
                batch = build_update_batch(buffer, normalizer, config.sac.batch_size)
                for key, value in agent.update(batch).items():
                    recent_metrics[key].append(value)

        if next_checkpoint_step is not None and total_steps >= next_checkpoint_step:
            checkpoint(run_dir / f"checkpoint_{total_steps}.pt")
            next_checkpoint_step += config.train.checkpoint_interval

        if total_steps >= next_eval_step or total_steps >= config.train.total_steps:
            evaluate_and_log()
            next_eval_step += config.train.eval_interval

    train_seconds = time.perf_counter() - start_time
    checkpoint(run_dir / "checkpoint.pt")

    final_seeds = evaluation_seeds(config.eval.seed_offset, config.eval.episodes)
    final_result, _ = evaluate_policy(
        eval_env,
        policy,
        normalizer,
        final_seeds,
        config.eval.bootstrap_resamples,
        config.eval.confidence,
    )
    summary = {
        "name": config.name,
        "env_id": config.env.env_id,
        "seed": config.seed,
        "total_steps": total_steps,
        "episodes": episodes,
        "train_wall_seconds": round(train_seconds, 2),
        "periodic_eval_seconds": round(eval_seconds, 2),
        "final_eval": final_result.to_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    env.close()
    eval_env.close()
    return summary
