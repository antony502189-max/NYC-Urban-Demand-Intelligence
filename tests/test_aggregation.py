from datetime import datetime, timedelta

import polars as pl

from nyc_demand.data.aggregate import aggregate_hourly_demand


def test_hourly_aggregation_counts_pickups_and_fills_missing_hours() -> None:
    start = datetime(2024, 1, 1, 10, 0)
    frame = pl.DataFrame(
        {
            "tpep_pickup_datetime": [start, start + timedelta(minutes=20), start + timedelta(hours=2)],
            "tpep_dropoff_datetime": [
                start + timedelta(minutes=10),
                start + timedelta(minutes=35),
                start + timedelta(hours=2, minutes=15),
            ],
            "PULocationID": [161, 161, 161],
            "DOLocationID": [162, 163, 164],
            "trip_distance": [1.0, 2.0, 3.0],
            "total_amount": [10.0, 15.0, 20.0],
        }
    )

    demand, report = aggregate_hourly_demand(frame)

    assert demand["demand"].to_list() == [2, 0, 1]
    assert demand["zone_id"].to_list() == [161, 161, 161]
    assert report["valid_rows"] == 3
