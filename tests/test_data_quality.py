from datetime import datetime, timedelta

import polars as pl

from nyc_demand.data.quality import clean_trips


def _trip_frame() -> pl.DataFrame:
    pickup = datetime(2024, 1, 1, 12, 0)
    return pl.DataFrame(
        {
            "tpep_pickup_datetime": [pickup, pickup, pickup],
            "tpep_dropoff_datetime": [
                pickup + timedelta(minutes=20),
                pickup + timedelta(minutes=10),
                pickup + timedelta(minutes=30),
            ],
            "PULocationID": [161, 999, 90],
            "DOLocationID": [162, 91, 91],
            "trip_distance": [2.4, 1.0, 500.0],
            "total_amount": [18.0, 12.0, 20.0],
        }
    )


def test_clean_trips_filters_invalid_domain_rows() -> None:
    cleaned, report = clean_trips(_trip_frame())

    assert cleaned.height == 1
    assert cleaned["PULocationID"].to_list() == [161]
    assert report.input_rows == 3
    assert report.valid_rows == 1
    assert report.dropped_rows == 2
    assert report.invalid_zone_rows == 1
    assert report.invalid_distance_rows == 1
    assert report.valid_share == 1 / 3
