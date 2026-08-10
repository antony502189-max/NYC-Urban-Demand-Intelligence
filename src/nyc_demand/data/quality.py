from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from nyc_demand.config import load_yaml

DEFAULT_CONFIG = "configs/data.yaml"


@dataclass(frozen=True)
class DataQualityReport:
    input_rows: int
    valid_rows: int
    dropped_rows: int
    null_pickup_rows: int
    invalid_zone_rows: int
    invalid_duration_rows: int
    invalid_distance_rows: int

    @property
    def valid_share(self) -> float:
        return self.valid_rows / self.input_rows if self.input_rows else 0.0

    def to_dict(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = asdict(self)
        payload["valid_share"] = self.valid_share
        return payload


def assert_required_columns(
    frame: pl.DataFrame,
    config_path: str | Path = DEFAULT_CONFIG,
) -> None:
    config = load_yaml(config_path)
    required = set(config["validation"]["required_columns"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required TLC columns: {', '.join(missing)}")


def clean_trips(
    frame: pl.DataFrame,
    config_path: str | Path = DEFAULT_CONFIG,
) -> tuple[pl.DataFrame, DataQualityReport]:
    """Apply explicit domain checks and return clean rows plus an audit report."""
    assert_required_columns(frame, config_path)
    config = load_yaml(config_path)
    rules = config["validation"]

    pickup = pl.col("tpep_pickup_datetime")
    dropoff = pl.col("tpep_dropoff_datetime")
    zone = pl.col("PULocationID")
    distance = pl.col("trip_distance")
    duration_hours = (dropoff - pickup).dt.total_seconds() / 3600

    null_pickup = pickup.is_null() | dropoff.is_null()
    invalid_zone = zone.is_null() | ~zone.is_between(
        rules["min_zone_id"], rules["max_zone_id"], closed="both"
    )
    invalid_duration = (
        null_pickup
        | (duration_hours < 0)
        | (duration_hours > float(rules["max_trip_duration_hours"]))
    )
    invalid_distance = distance.is_null() | ~distance.is_between(
        float(rules["min_trip_distance"]),
        float(rules["max_trip_distance"]),
        closed="both",
    )

    flags = frame.select(
        null_pickup.alias("null_pickup"),
        invalid_zone.alias("invalid_zone"),
        invalid_duration.alias("invalid_duration"),
        invalid_distance.alias("invalid_distance"),
    )

    valid_mask = ~(
        null_pickup | invalid_zone | invalid_duration | invalid_distance
    )
    cleaned = frame.filter(valid_mask)

    report = DataQualityReport(
        input_rows=frame.height,
        valid_rows=cleaned.height,
        dropped_rows=frame.height - cleaned.height,
        null_pickup_rows=flags["null_pickup"].sum(),
        invalid_zone_rows=flags["invalid_zone"].sum(),
        invalid_duration_rows=flags["invalid_duration"].sum(),
        invalid_distance_rows=flags["invalid_distance"].sum(),
    )
    return cleaned, report
