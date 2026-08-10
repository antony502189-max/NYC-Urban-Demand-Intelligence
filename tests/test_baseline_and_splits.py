from datetime import datetime, timedelta

import polars as pl

from nyc_demand.models.baseline import seasonal_naive
from nyc_demand.models.splits import expanding_window_folds, split_frame


def test_seasonal_naive_uses_same_zone_history() -> None:
    start = datetime(2024, 1, 1)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(4)],
            "zone_id": [1, 1, 1, 1],
            "demand": [5, 6, 7, 8],
        }
    )

    result = seasonal_naive(frame, period_hours=2)
    assert result["prediction"].to_list() == [None, None, 5.0, 6.0]


def test_expanding_window_folds_are_chronological() -> None:
    start = datetime(2024, 1, 1)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(24 * 40)],
            "zone_id": [1] * (24 * 40),
            "demand": [1] * (24 * 40),
        }
    )

    folds = expanding_window_folds(
        frame,
        train_days=20,
        validation_days=5,
        step_days=5,
    )

    assert len(folds) == 4
    train, validation = split_frame(frame, folds[0])
    assert train["timestamp"].max() < validation["timestamp"].min()
