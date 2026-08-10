from __future__ import annotations

from datetime import timedelta

import polars as pl


def make_horizon_dataset(
    frame: pl.DataFrame,
    *,
    horizon_hours: int,
    target_column: str = "target_demand",
) -> pl.DataFrame:
    """Attach future demand targets to origin-time features for a direct forecast horizon."""
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")

    required = {"timestamp", "zone_id", "demand"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    ordered = frame.sort(["zone_id", "timestamp"])
    horizon = timedelta(hours=horizon_hours)
    return (
        ordered.with_columns(
            pl.col("demand")
            .shift(-horizon_hours)
            .over("zone_id")
            .alias(target_column),
            (pl.col("timestamp") + horizon).alias("forecast_timestamp"),
        )
        .drop_nulls([target_column])
        .sort(["timestamp", "zone_id"])
    )
