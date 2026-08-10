from datetime import datetime, timedelta

import numpy as np
import polars as pl

from nyc_demand.models.backtest import backtest_lightgbm, summarize_folds


def _feature_frame(hours: int = 240) -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    index = list(range(hours))
    demand = [30 + (i % 24) // 4 + (i // 24) % 3 for i in index]
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in index],
            "zone_id": [161] * hours,
            "demand": demand,
            "hour": [i % 24 for i in index],
            "day_of_week": [(i // 24) % 7 for i in index],
            "demand_lag_1h": [float(demand[max(0, i - 1)]) for i in index],
            "demand_lag_24h": [float(demand[max(0, i - 24)]) for i in index],
        }
    )


def test_backtest_returns_finite_fold_metrics() -> None:
    results = backtest_lightgbm(
        _feature_frame(),
        horizon_hours=6,
        train_days=4,
        validation_days=2,
        step_days=2,
        params={
            "objective": "poisson",
            "n_estimators": 15,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "random_state": 42,
        },
    )
    summary = summarize_folds(results)

    assert len(results) >= 2
    assert all(result.horizon_hours == 6 for result in results)
    assert all(result.train_rows > 0 for result in results)
    assert all(result.validation_rows > 0 for result in results)
    assert all(np.isfinite(value) for value in summary.values())
