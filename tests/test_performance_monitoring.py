from datetime import datetime, timedelta

import numpy as np
import polars as pl

from nyc_demand.monitoring.performance import (
    classify_performance_regression,
    performance_by_day,
    performance_snapshot,
)


def test_performance_snapshot_exposes_operational_metrics() -> None:
    frame = pl.DataFrame(
        {
            "actual": [10.0, 20.0, 30.0, 40.0],
            "prediction": [9.0, 18.0, 33.0, 35.0],
        }
    )

    snapshot = performance_snapshot(frame)

    assert snapshot.observations == 4
    assert snapshot.mae == 2.75
    assert np.isfinite(snapshot.wape)
    assert np.isfinite(snapshot.p90_absolute_error)


def test_performance_by_day_returns_daily_wape_and_bias() -> None:
    start = datetime(2026, 1, 1)
    frame = pl.DataFrame(
        {
            "timestamp": [
                start,
                start + timedelta(hours=1),
                start + timedelta(days=1),
                start + timedelta(days=1, hours=1),
            ],
            "actual": [10.0, 20.0, 30.0, 40.0],
            "prediction": [9.0, 18.0, 27.0, 36.0],
        }
    )

    result = performance_by_day(frame)

    assert result.height == 2
    assert result["observations"].to_list() == [2, 2]
    assert all(value > 0 for value in result["wape"].to_list())


def test_performance_regression_severity_bands() -> None:
    assert classify_performance_regression(0.10, 0.11) == "stable"
    assert classify_performance_regression(0.10, 0.12) == "watch"
    assert classify_performance_regression(0.10, 0.14) == "critical"
