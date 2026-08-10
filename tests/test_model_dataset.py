from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nyc_demand.models.dataset import build_feature_matrix, build_training_matrix


def _frame() -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(4)],
            "zone_id": [161, 161, 162, 162],
            "hour": [0, 1, 2, 3],
            "demand_lag_1h": [10.0, 11.0, 5.0, 7.0],
            "demand": [11, 12, 7, 9],
        }
    )


def test_training_matrix_excludes_timestamp_and_target() -> None:
    matrix = build_training_matrix(_frame())

    assert matrix.feature_names == ("zone_id", "hour", "demand_lag_1h")
    assert matrix.features.shape == (4, 3)
    assert matrix.target.tolist() == [11.0, 12.0, 7.0, 9.0]


def test_inference_matrix_rejects_nulls_to_preserve_row_alignment() -> None:
    frame = _frame().with_columns(
        pl.when(pl.col("hour") == 2)
        .then(None)
        .otherwise(pl.col("demand_lag_1h"))
        .alias("demand_lag_1h")
    )

    with pytest.raises(ValueError, match="null"):
        build_feature_matrix(frame, feature_columns=("hour", "demand_lag_1h"))


def test_training_matrix_rejects_non_finite_target() -> None:
    frame = _frame().with_columns(
        pl.Series("demand", [11.0, 12.0, np.inf, 9.0])
    )

    with pytest.raises(ValueError, match="Target contains"):
        build_training_matrix(frame)
