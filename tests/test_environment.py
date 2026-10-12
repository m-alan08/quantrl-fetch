import gymnasium as gym
import gymnasium_robotics
import numpy as np

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
