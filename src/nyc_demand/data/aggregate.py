from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nyc_demand.data.quality import DEFAULT_CONFIG, clean_trips


READ_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "total_amount",
]


def aggregate_hourly_demand(
    frame: pl.DataFrame,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> tuple[pl.DataFrame, dict[str, int | float]]:
    cleaned, report = clean_trips(frame, config_path)
    if cleaned.is_empty():
        raise ValueError("No valid trip rows remain after quality checks")

    demand = (
        cleaned.with_columns(
            pl.col("tpep_pickup_datetime").dt.truncate("1h").alias("timestamp"),
            pl.col("PULocationID").cast(pl.Int16).alias("zone_id"),
        )
        .group_by(["timestamp", "zone_id"])
        .agg(pl.len().cast(pl.Int32).alias("demand"))
        .sort(["timestamp", "zone_id"])
    )
    return demand, report.to_dict()


def aggregate_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)

    frame = pl.read_parquet(source, columns=READ_COLUMNS)
    demand, report = aggregate_hourly_demand(frame, config_path=config_path)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    demand.write_parquet(destination, compression="zstd", statistics=True)
    destination.with_suffix(".quality.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TLC trips to hourly zone demand")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(aggregate_file(args.input, args.output, config_path=args.config))


if __name__ == "__main__":
    main()
