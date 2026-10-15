import gymnasium as gym
import numpy as np
import pytest
import torch

from quantrl.algos.replay import HindsightReplayBuffer
from quantrl.algos.sac import NonFiniteTrainingError, SACAgent
from quantrl.checkpoint import load_checkpoint
from quantrl.config import SACConfig, load_config
from quantrl.envs.factory import (
    EnvDimensions,
    EnvironmentContractError,
    check_goal_env,
    check_goal_observation,
    make_env,
)
from quantrl.quant.qat import CheckpointMismatchError, check_checkpoint_matches
from quantrl.training import check_action, train

SMOKE = [
    "device=cpu",
    "sac.hidden_sizes=[16, 16]",
    "sac.batch_size=16",
    "train.total_steps=150",
    "train.random_steps=100",
    "train.update_after=100",
    "train.updates_per_step=0.2",
    "train.eval_interval=150",
    "train.eval_episodes=1",
    "eval.episodes=1",
    "eval.bootstrap_resamples=20",
    "her.buffer_episodes=5",
]
DIMENSIONS = EnvDimensions(observation_dim=10, goal_dim=3, action_dim=4, episode_length=50)


def update_batch(rows=16, dim=8, action_dim=2):
    rng = np.random.default_rng(0)
    return {
        "inputs": rng.normal(size=(rows, dim)).astype(np.float32),
        "next_inputs": rng.normal(size=(rows, dim)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(rows, action_dim)).astype(np.float32),
        "rewards": -np.ones(rows, dtype=np.float32),
    }


def test_nan_weights_stop_training_with_a_clear_error():
    agent = SACAgent(8, 2, SACConfig(hidden_sizes=[16]))
    with torch.no_grad():
        agent.critic.q1[0].weight[0, 0] = float("nan")
    with pytest.raises(NonFiniteTrainingError, match="Non-finite values in SAC update 1"):
        agent.update(update_batch())


def test_nan_inputs_stop_training_with_a_clear_error():
    agent = SACAgent(8, 2, SACConfig(hidden_sizes=[16]))
    batch = update_batch()
    batch["rewards"][3] = float("inf")
    with pytest.raises(NonFiniteTrainingError, match="rewards"):
        agent.update(batch)


def test_healthy_update_reports_finite_gradient_norms():
    metrics = SACAgent(8, 2, SACConfig(hidden_sizes=[16])).update(update_batch())
    assert metrics["critic_grad_norm"] > 0 and np.isfinite(metrics["critic_grad_norm"])
    assert metrics["actor_grad_norm"] > 0 and np.isfinite(metrics["actor_grad_norm"])


def make_buffer():
    return HindsightReplayBuffer(3, 5, 2, 3, 1, lambda a, g, i: np.zeros(len(a)))


def test_sampling_an_empty_buffer_explains_itself():
    with pytest.raises(RuntimeError, match="empty replay buffer"):
        make_buffer().sample(4)


@pytest.mark.parametrize(
    "field, value",
    [
        ("observations", np.zeros((5, 2))),
        ("actions", np.zeros((5, 2))),
        ("observations", np.full((6, 2), np.nan)),
    ],
)
def test_storing_a_malformed_episode_is_rejected(field, value):
    episode = {
        "observations": np.zeros((6, 2)),
        "achieved_goals": np.zeros((6, 3)),
        "desired_goals": np.zeros((5, 3)),
        "actions": np.zeros((5, 1)),
    }
    episode[field] = value
    with pytest.raises(ValueError, match=field):
        make_buffer().store_episode(**episode)


@pytest.mark.parametrize(
    "action, message",
    [
        (np.zeros(3), "shape"),
        (np.array([0.0, np.nan, 0.0, 0.0]), "NaN"),
        (np.array([0.0, 1.5, 0.0, 0.0]), "outside"),
    ],
)
def test_bad_actions_are_rejected(action, message):
    with pytest.raises(EnvironmentContractError, match=message):
        check_action(action, DIMENSIONS)


def test_bad_observations_are_rejected():
    observation = {
        "observation": np.zeros(10),
        "achieved_goal": np.zeros(3),
        "desired_goal": np.zeros(2),
    }
    with pytest.raises(EnvironmentContractError, match="desired_goal"):
        check_goal_observation(observation, DIMENSIONS)
    observation["desired_goal"] = np.array([0.0, np.inf, 0.0])
    with pytest.raises(EnvironmentContractError, match="NaN or inf"):
        check_goal_observation(observation, DIMENSIONS)


def test_fetch_environments_meet_the_contract():
    for env_id in ("FetchReach-v4", "FetchPush-v4"):
        env = make_env(load_config(None, [f"env.env_id={env_id}"]).env)
        check_goal_env(env)
        env.close()


def test_a_non_goal_environment_is_rejected():
    env = gym.make("Pendulum-v1")
    with pytest.raises(EnvironmentContractError, match="Dict"):
        check_goal_env(env)
    env.close()


def test_a_checkpoint_from_another_architecture_is_rejected(tmp_path):
    config = load_config("configs/fetch_reach.json", [*SMOKE, f"output_dir={tmp_path}"])
    train(config, verbose=False)
    checkpoint = load_checkpoint(config.run_dir / "checkpoint.pt")
    other = load_config(
        "configs/fetch_reach.json", [*SMOKE, "sac.hidden_sizes=[32, 32]", "env.env_id=FetchPush-v4"]
    )
    dimensions = dict(checkpoint["dimensions"], observation_dim=25)
    with pytest.raises(CheckpointMismatchError) as error:
        check_checkpoint_matches(checkpoint, other, dimensions)
    text = str(error.value)
    assert "env.env_id" in text and "sac.hidden_sizes" in text and "dimensions" in text
