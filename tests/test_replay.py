import numpy as np

from quantrl.algos.replay import HindsightReplayBuffer, future_goal_probability

EPISODE_LENGTH = 5
GOAL_DIM = 3


def distance_reward(achieved, desired, info):
    return -(np.linalg.norm(achieved - desired, axis=-1) > 0.05).astype(np.float32)


def make_buffer(replay_k=4, capacity=10, seed=0):
    return HindsightReplayBuffer(
        capacity_episodes=capacity,
        episode_length=EPISODE_LENGTH,
        observation_dim=2,
        goal_dim=GOAL_DIM,
        action_dim=1,
        reward_function=distance_reward,
        replay_k=replay_k,
        rng=np.random.default_rng(seed),
    )


def store_labeled_episode(buffer, episode_id):
    steps = np.arange(EPISODE_LENGTH + 1, dtype=np.float32)
    observations = np.stack([np.full_like(steps, episode_id), steps], axis=1)
    achieved = np.stack([np.full_like(steps, episode_id), steps, np.zeros_like(steps)], axis=1)
    desired = np.tile(np.array([[-1.0, -1.0, -1.0]], np.float32), (EPISODE_LENGTH, 1))
    actions = steps[:-1, None].copy()
    buffer.store_episode(observations, achieved, desired, actions)


def test_future_probability_matches_replay_k():
    assert future_goal_probability(0) == 0.0
    assert future_goal_probability(4) == 0.8


def test_without_relabeling_goals_and_rewards_are_original():
    buffer = make_buffer(replay_k=0)
    store_labeled_episode(buffer, episode_id=3)
    batch = buffer.sample(256)
    assert not batch["relabeled"].any()
    assert np.all(batch["goals"] == -1.0)
    assert np.all(batch["rewards"] == -1.0)


def test_relabeled_goals_are_future_achieved_goals_from_the_same_episode():
    buffer = make_buffer(replay_k=4)
    for episode_id in range(4):
        store_labeled_episode(buffer, episode_id)
    batch = buffer.sample(4096)
    relabeled = batch["relabeled"]
    assert 0.75 < relabeled.mean() < 0.85
    goal_episode = batch["goals"][relabeled, 0]
    goal_step = batch["goals"][relabeled, 1]
    transition_episode = batch["observations"][relabeled, 0]
    transition_step = batch["observations"][relabeled, 1]
    assert np.all(goal_episode == transition_episode)
    assert np.all(goal_step >= transition_step + 1)
    assert np.all(goal_step <= EPISODE_LENGTH)
    assert set(np.unique(goal_step - transition_step)) == set(range(1, EPISODE_LENGTH + 1))


def test_rewards_are_recomputed_for_relabeled_goals():
    buffer = make_buffer(replay_k=4)
    store_labeled_episode(buffer, episode_id=0)
    batch = buffer.sample(2048)
    next_step = batch["next_observations"][:, 1]
    goal_step = batch["goals"][:, 1]
    reached = batch["relabeled"] & (goal_step == next_step)
    assert np.all(batch["rewards"][reached] == 0.0)
    assert np.all(batch["rewards"][~reached] == -1.0)
    assert np.all(batch["actions"][:, 0] == batch["observations"][:, 1])


def test_buffer_overwrites_oldest_episode_when_full():
    buffer = make_buffer(capacity=2)
    for episode_id in range(3):
        store_labeled_episode(buffer, episode_id)
    assert len(buffer) == 2
    assert set(np.unique(buffer.observations[:, 0, 0])) == {1.0, 2.0}
