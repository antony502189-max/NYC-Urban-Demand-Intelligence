from datetime import datetime, timedelta

import numpy as np
import polars as pl

from nyc_demand.models.baseline_backtest import (
    attach_direct_seasonal_baseline,
    backtest_seasonal_baseline,
    summarize_baseline_folds,
)


def _weekly_series(hours: int = 24 * 28) -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    index = list(range(hours))
    demand = [20 + (i % 24) + 3 * ((i // 24) % 7) for i in index]
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in index],
            "zone_id": [161] * hours,
            "demand": demand,
        }
    )


def test_direct_seasonal_baseline_uses_target_hour_from_previous_week() -> None:
    frame = _weekly_series()
    result = attach_direct_seasonal_baseline(frame, horizon_hours=24)

    first = result.row(0, named=True)
    origin = first["timestamp"]
    expected_target_time = origin + timedelta(hours=24)
    previous_week_time = expected_target_time - timedelta(hours=168)

    source = frame.filter(pl.col("timestamp") == previous_week_time)
    assert source.height == 1
    assert first["prediction"] == float(source["demand"][0])


def test_seasonal_baseline_backtest_returns_finite_metrics() -> None:
    results = backtest_seasonal_baseline(
        _weekly_series(),
        horizon_hours=6,
        train_days=10,
        validation_days=4,
        step_days=4,
    )
    summary = summarize_baseline_folds(results)

    assert len(results) >= 2
    assert all(result.validation_rows > 0 for result in results)
    assert all(np.isfinite(value) for value in summary.values())
