from datetime import datetime, timedelta

import polars as pl

from nyc_demand.models.dataset import build_training_matrix
from nyc_demand.models.horizons import make_horizon_dataset


def test_horizon_dataset_aligns_future_demand_without_target_leakage() -> None:
    start = datetime(2026, 1, 1)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(5)],
            "zone_id": [161] * 5,
            "demand": [10, 11, 12, 13, 14],
            "demand_lag_1h": [9.0, 10.0, 11.0, 12.0, 13.0],
        }
    )

    supervised = make_horizon_dataset(frame, horizon_hours=2)
    matrix = build_training_matrix(supervised, target_column="target_demand")

    assert supervised["target_demand"].to_list() == [12, 13, 14]
    assert supervised["forecast_timestamp"].to_list() == [
        start + timedelta(hours=2),
        start + timedelta(hours=3),
        start + timedelta(hours=4),
    ]
    assert "target_demand" not in matrix.feature_names
    assert "demand" not in matrix.feature_names
