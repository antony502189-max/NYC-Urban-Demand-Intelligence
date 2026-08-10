from datetime import datetime, timedelta

import polars as pl
import pytest

from nyc_demand.data.range_pipeline import YearMonth, merge_hourly_demand, month_range


def test_month_range_is_inclusive_across_year_boundary() -> None:
    result = month_range(YearMonth(2025, 11), YearMonth(2026, 2))

    assert [item.label for item in result] == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]


def test_month_range_rejects_reverse_window() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        month_range(YearMonth(2026, 2), YearMonth(2026, 1))


def test_merge_hourly_demand_collapses_overlapping_timestamp_zone_keys() -> None:
    start = datetime(2026, 1, 31, 23, 0)
    left = pl.DataFrame(
        {
            "timestamp": [start, start + timedelta(hours=1)],
            "zone_id": [161, 161],
            "demand": [2, 4],
        }
    )
    right = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=1), start + timedelta(hours=2)],
            "zone_id": [161, 161],
            "demand": [3, 5],
        }
    )

    merged = merge_hourly_demand([left, right])

    assert merged["timestamp"].to_list() == [
        start,
        start + timedelta(hours=1),
        start + timedelta(hours=2),
    ]
    assert merged["demand"].to_list() == [2, 7, 5]
