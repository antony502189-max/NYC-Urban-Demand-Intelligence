from __future__ import annotations

import argparse
import json
from datetime import datetime
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


def _complete_hourly_grid(counts: pl.DataFrame) -> pl.DataFrame:
    start = counts["timestamp"].min()
    end = counts["timestamp"].max()
    if start is None or end is None:
        raise ValueError("Unable to determine aggregation time range")

    hours = pl.DataFrame(
        {"timestamp": pl.datetime_range(start, end, interval="1h", eager=True)}
    )
    zones = counts.select("zone_id").unique().sort("zone_id")
    grid = hours.join(zones, how="cross")

    return (
        grid.join(counts, on=["timestamp", "zone_id"], how="left")
        .with_columns(pl.col("demand").fill_null(0).cast(pl.Int32))
        .sort(["timestamp", "zone_id"])
    )


def aggregate_hourly_demand(
    frame: pl.DataFrame,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    complete_grid: bool = True,
) -> tuple[pl.DataFrame, dict[str, int | float]]:
    """Build an hourly pickup-demand mart with explicit zero-demand hours."""
    cleaned, report = clean_trips(frame, config_path)
    if cleaned.is_empty():
        raise ValueError("No valid trip rows remain after quality checks")

    counts = (
        cleaned.with_columns(
            pl.col("tpep_pickup_datetime").dt.truncate("1h").alias("timestamp"),
            pl.col("PULocationID").cast(pl.Int16).alias("zone_id"),
        )
        .group_by(["timestamp", "zone_id"])
        .agg(pl.len().cast(pl.Int32).alias("demand"))
        .sort(["timestamp", "zone_id"])
    )

    demand = _complete_hourly_grid(counts) if complete_grid else counts
    return demand, report.to_dict()


def _read_trip_window(
    source: Path,
    *,
    start_time: datetime | None,
    end_time: datetime | None,
) -> pl.DataFrame:
    """Read only the requested pickup-time window using Parquet predicate pushdown."""
    scan = pl.scan_parquet(source).select(READ_COLUMNS)
    pickup = pl.col("tpep_pickup_datetime")

    if start_time is not None:
        scan = scan.filter(pickup >= pl.lit(start_time))
    if end_time is not None:
        scan = scan.filter(pickup < pl.lit(end_time))

    frame = scan.collect()
    if frame.is_empty():
        window = f"[{start_time!s}, {end_time!s})"
        raise ValueError(f"No trip rows found in requested pickup window {window}")
    return frame


def aggregate_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> Path:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise ValueError("end_time must be later than start_time")

    frame = _read_trip_window(source, start_time=start_time, end_time=end_time)
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
    parser.add_argument("--start-time", type=datetime.fromisoformat)
    parser.add_argument("--end-time", type=datetime.fromisoformat)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        aggregate_file(
            args.input,
            args.output,
            config_path=args.config,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    )


if __name__ == "__main__":
    main()
