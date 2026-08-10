from datetime import datetime, timedelta

import polars as pl

from nyc_demand.features.builder import build_features


def test_features_use_only_prior_zone_history() -> None:
    start = datetime(2024, 1, 1, 0, 0)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(3)],
            "zone_id": [1, 1, 1],
            "demand": [10, 20, 30],
        }
    )

    result = build_features(frame, lags=[1], windows=[2])

    assert result["demand_lag_1h"].to_list() == [None, 10, 20]
    assert result["demand_roll_mean_2h"].to_list() == [None, 10.0, 15.0]
    assert result["hour"].to_list() == [0, 1, 2]


def test_lags_do_not_cross_zone_boundaries() -> None:
    start = datetime(2024, 1, 1, 0, 0)
    frame = pl.DataFrame(
        {
            "timestamp": [start, start, start + timedelta(hours=1), start + timedelta(hours=1)],
            "zone_id": [1, 2, 1, 2],
            "demand": [10, 100, 20, 200],
        }
    )

    result = build_features(frame, lags=[1], windows=[]).sort(["zone_id", "timestamp"])

    assert result["demand_lag_1h"].to_list() == [None, 10, None, 100]
