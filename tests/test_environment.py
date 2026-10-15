import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pytest

gym.register_envs(gymnasium_robotics)


def test_fetch_reach_has_goal_conditioned_dict_observations():
    env = gym.make("FetchReach-v4")
    observation, _ = env.reset(seed=0)
    assert set(observation) == {"observation", "achieved_goal", "desired_goal"}
    assert observation["observation"].shape == (10,)
    assert env.action_space.shape == (4,)
    env.close()


def test_compute_reward_is_vectorized():
    env = gym.make("FetchReach-v4")
    achieved = np.zeros((5, 3))
    desired = np.zeros((5, 3))
    desired[0] = 1.0
    rewards = env.unwrapped.compute_reward(achieved, desired, {})
    assert rewards.shape == (5,)
    assert rewards[0] == -1.0
    assert np.all(rewards[1:] == 0.0)
    env.close()


ENV_IDS = ["FetchReach-v4", "FetchPush-v4"]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reset_and_step_return_the_documented_types(env_id):
    env = gym.make(env_id)
    observation, info = env.reset(seed=0)
    assert isinstance(info, dict)
    for key in ("observation", "achieved_goal", "desired_goal"):
        assert observation[key].dtype == np.float64
        assert observation[key].ndim == 1
        assert env.observation_space[key].contains(
            observation[key].astype(env.observation_space[key].dtype)
        )
    action = env.action_space.sample()
    assert action.dtype == np.float32 and action.shape == (4,)
    observation, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, (float, np.floating)) and np.ndim(reward) == 0
    assert terminated is False and truncated is False
    assert set(info) >= {"is_success"}
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_goal_handling_and_reward_range(env_id):
    env = gym.make(env_id)
    observation, _ = env.reset(seed=3)
    goal = env.unwrapped.goal.copy()
    np.testing.assert_allclose(observation["desired_goal"], goal)
    rewards = []
    truncated = False
    steps = 0
    while not truncated:
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert not terminated
        np.testing.assert_allclose(observation["desired_goal"], goal)
        distance = np.linalg.norm(observation["achieved_goal"] - observation["desired_goal"])
        assert reward == (0.0 if distance < 0.05 else -1.0)
        assert info["is_success"] == float(distance < 0.05)
        rewards.append(reward)
    assert steps == env.spec.max_episode_steps == 50
    assert set(rewards) <= {0.0, -1.0}
    env.close()


def test_reach_achieved_goal_is_the_gripper_position():
    env = gym.make("FetchReach-v4")
    observation, _ = env.reset(seed=1)
    np.testing.assert_allclose(observation["achieved_goal"], observation["observation"][:3])
    env.close()


def test_push_achieved_goal_is_the_block_position():
    env = gym.make("FetchPush-v4")
    observation, _ = env.reset(seed=1)
    np.testing.assert_allclose(observation["achieved_goal"], observation["observation"][3:6])
    env.close()


def test_same_seed_gives_the_same_goal_and_start():
    first = gym.make("FetchPush-v4")
    second = gym.make("FetchPush-v4")
    a, _ = first.reset(seed=11)
    b, _ = second.reset(seed=11)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    first.close()
    second.close()
