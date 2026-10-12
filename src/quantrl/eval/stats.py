from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def bootstrap_confidence_interval(
    values: Sequence[float] | np.ndarray,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, data.size, size=(resamples, data.size))
    resample_means = data[resample_indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(resample_means, [tail, 1.0 - tail])
    return float(low), float(high)


def aggregate_across_seeds(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric: str,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(list(group_columns), sort=True) if group_columns else [((), frame)]
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        values = group[metric].to_numpy(dtype=np.float64)
        low, high = bootstrap_confidence_interval(values, resamples, confidence)
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "metric": metric,
                "n_seeds": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "ci_low": low,
                "ci_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
