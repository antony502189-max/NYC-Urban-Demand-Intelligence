from datetime import datetime

import polars as pl

from nyc_demand.data.aggregate import aggregate_file


def test_aggregate_file_filters_pickups_to_requested_window(tmp_path) -> None:
    source = tmp_path / "yellow_tripdata_2024-01.parquet"
    output = tmp_path / "demand.parquet"

    frame = pl.DataFrame(
        {
            "tpep_pickup_datetime": [
                datetime(2002, 1, 1, 12, 0),
                datetime(2024, 1, 15, 12, 10),
                datetime(2024, 1, 15, 12, 40),
                datetime(2024, 2, 1, 0, 5),
            ],
            "tpep_dropoff_datetime": [
                datetime(2002, 1, 1, 12, 15),
                datetime(2024, 1, 15, 12, 25),
                datetime(2024, 1, 15, 12, 55),
                datetime(2024, 2, 1, 0, 20),
            ],
            "PULocationID": [161, 161, 161, 161],
            "DOLocationID": [162, 162, 162, 162],
            "trip_distance": [1.0, 1.2, 2.0, 1.1],
            "total_amount": [12.0, 15.0, 20.0, 13.0],
        }
    )
    frame.write_parquet(source)

    aggregate_file(
        source,
        output,
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 2, 1),
    )

    result = pl.read_parquet(output)

    assert result["timestamp"].min() == datetime(2024, 1, 15, 12, 0)
    assert result["timestamp"].max() == datetime(2024, 1, 15, 12, 0)
    assert result["demand"].sum() == 2
