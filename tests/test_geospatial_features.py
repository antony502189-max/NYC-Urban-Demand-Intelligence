import polars as pl

from nyc_demand.features.geospatial import add_zone_features


def test_zone_features_are_one_hot_and_unknown_safe() -> None:
    frame = pl.DataFrame({"zone_id": [1, 2, 999], "demand_lag_1h": [10.0, 20.0, 30.0]})
    lookup = pl.DataFrame(
        {
            "zone_id": [1, 2],
            "borough": ["Manhattan", "Queens"],
            "service_zone": ["Yellow Zone", "Boro Zone"],
        }
    )

    result = add_zone_features(frame, lookup)

    assert result["borough_manhattan"].to_list() == [1, 0, 0]
    assert result["borough_queens"].to_list() == [0, 1, 0]
    assert result["borough_unknown"].to_list() == [0, 0, 1]
    assert result["service_zone_yellow_zone"].to_list() == [1, 0, 0]
    assert result["service_zone_unknown"].to_list() == [0, 0, 1]
    assert "borough" not in result.columns
    assert "service_zone" not in result.columns
