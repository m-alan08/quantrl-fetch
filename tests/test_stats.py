import numpy as np
import pandas as pd

from quantrl.eval.stats import aggregate_across_seeds, bootstrap_confidence_interval


def test_bootstrap_interval_contains_mean_and_is_ordered():
    values = np.random.default_rng(0).normal(1.0, 1.0, size=200)
    low, high = bootstrap_confidence_interval(values, resamples=2000)
    assert low < values.mean() < high
    assert high - low < 0.5


def test_bootstrap_interval_of_constant_is_degenerate():
    assert bootstrap_confidence_interval(np.ones(10)) == (1.0, 1.0)


def test_aggregate_across_seeds_groups_rows():
    frame = pd.DataFrame(
        {"method": ["a", "a", "a", "b", "b", "b"], "success": [1.0, 0.8, 0.9, 0.1, 0.2, 0.3]}
    )
    summary = aggregate_across_seeds(frame, ["method"], "success", resamples=500)
    assert list(summary["method"]) == ["a", "b"]
    assert list(summary["n_seeds"]) == [3, 3]
    np.testing.assert_allclose(summary["mean"], [0.9, 0.2])
