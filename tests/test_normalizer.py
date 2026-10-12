import numpy as np

from quantrl.envs.normalizer import GoalObservationNormalizer, RunningNormalizer


def test_running_normalizer_matches_numpy_statistics():
    rng = np.random.default_rng(0)
    values = rng.normal(3.0, 2.0, size=(1000, 4))
    normalizer = RunningNormalizer(4, eps=1e-6, clip=100.0)
    normalizer.update(values[:400])
    normalizer.update(values[400:])
    np.testing.assert_allclose(normalizer.mean, values.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(normalizer.std, values.std(axis=0), rtol=1e-8)


def test_normalized_values_are_clipped():
    normalizer = RunningNormalizer(1, eps=1.0, clip=5.0)
    normalizer.update(np.zeros((10, 1)))
    assert normalizer.normalize(np.array([[100.0]]))[0, 0] == 5.0


def test_goal_normalizer_state_round_trip():
    rng = np.random.default_rng(1)
    first = GoalObservationNormalizer(5, 3)
    first.update(rng.normal(size=(50, 5)), rng.normal(size=(70, 3)))
    second = GoalObservationNormalizer(5, 3)
    second.load_state_dict(first.state_dict())
    observations, goals = rng.normal(size=(4, 5)), rng.normal(size=(4, 3))
    np.testing.assert_array_equal(first(observations, goals), second(observations, goals))
    assert first(observations, goals).shape == (4, 8)
