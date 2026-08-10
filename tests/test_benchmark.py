from datetime import datetime, timedelta

import numpy as np
import polars as pl

from nyc_demand.features.builder import build_features
from nyc_demand.models.benchmark import compare_lightgbm_to_seasonal_baseline, summarize_benchmark


def _feature_frame(hours: int = 24 * 35) -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    index = list(range(hours))
    demand = [25 + (i % 24) // 3 + 2 * ((i // 24) % 7) for i in index]
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in index],
            "zone_id": [161] * hours,
            "demand": demand,
        }
    )
    return build_features(frame)


def test_benchmark_uses_paired_validation_rows_and_finite_scores() -> None:
    results = compare_lightgbm_to_seasonal_baseline(
        _feature_frame(),
        horizon_hours=6,
        train_days=10,
        validation_days=3,
        step_days=3,
        params={
            "objective": "poisson",
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "random_state": 42,
        },
    )
    summary = summarize_benchmark(results)

    assert len(results) >= 2
    assert all(result.validation_rows > 0 for result in results)
    assert all(np.isfinite(value) for value in summary.values())
