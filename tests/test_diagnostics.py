from datetime import datetime, timedelta

import numpy as np
import polars as pl

from nyc_demand.models.diagnostics import (
    demand_band_error_table,
    residual_summary,
    zone_error_table,
)


def test_residual_summary_reports_bias_and_tail_error() -> None:
    result = residual_summary([10, 20, 30, 40], [8, 21, 25, 50])

    assert result["bias"] == -1.0
    assert result["median_absolute_error"] == 3.5
    assert np.isclose(result["underprediction_share"], 0.5)
    assert np.isclose(result["overprediction_share"], 0.5)
    assert result["p90_absolute_error"] > result["median_absolute_error"]


def test_zone_error_table_ranks_worst_zone_first() -> None:
    start = datetime(2026, 1, 1)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(4)],
            "zone_id": [1, 1, 2, 2],
            "target_demand": [10.0, 10.0, 10.0, 10.0],
            "prediction": [9.0, 11.0, 2.0, 18.0],
        }
    )

    result = zone_error_table(frame)

    assert result["zone_id"].to_list() == [2, 1]
    assert result["mae"].to_list() == [8.0, 1.0]


def test_demand_band_error_table_keeps_business_friendly_order() -> None:
    start = datetime(2026, 1, 1)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(4)],
            "zone_id": [1, 1, 1, 1],
            "target_demand": [0.0, 5.0, 15.0, 30.0],
            "prediction": [1.0, 4.0, 14.0, 29.0],
        }
    )

    result = demand_band_error_table(frame)

    assert result["demand_band"].to_list() == ["zero", "low", "medium", "high"]
    assert result["observations"].to_list() == [1, 1, 1, 1]
