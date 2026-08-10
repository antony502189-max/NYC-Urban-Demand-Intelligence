from datetime import datetime, timedelta

import polars as pl

from nyc_demand.analysis.eda import demand_overview, hourly_profile, top_zones


def _demand_frame() -> pl.DataFrame:
    start = datetime(2026, 1, 5, 0, 0)
    return pl.DataFrame(
        {
            "timestamp": [start, start, start + timedelta(hours=1), start + timedelta(hours=1)],
            "zone_id": [1, 2, 1, 2],
            "demand": [10, 0, 20, 5],
        }
    )


def test_demand_overview_reports_coverage_and_zero_share() -> None:
    overview = demand_overview(_demand_frame())

    assert overview["rows"] == 4
    assert overview["zones"] == 2
    assert overview["total_demand"] == 35
    assert overview["zero_demand_share"] == 0.25


def test_hourly_profile_aggregates_all_zones() -> None:
    result = hourly_profile(_demand_frame())

    assert result["hour"].to_list() == [0, 1]
    assert result["total_demand"].to_list() == [10, 25]


def test_top_zones_can_attach_human_readable_labels() -> None:
    lookup = pl.DataFrame(
        {
            "zone_id": [1, 2],
            "borough": ["Manhattan", "Queens"],
            "zone_name": ["Alpha", "Beta"],
        }
    )

    result = top_zones(_demand_frame(), n=1, zone_lookup=lookup)

    assert result["zone_id"].to_list() == [1]
    assert result["borough"].to_list() == ["Manhattan"]
    assert result["zone_name"].to_list() == ["Alpha"]
