from __future__ import annotations

import polars as pl


def seasonal_naive(
    frame: pl.DataFrame,
    *,
    period_hours: int = 168,
    prediction_column: str = "prediction",
) -> pl.DataFrame:
    """Predict each zone from its demand at the same hour one season earlier."""
    if period_hours <= 0:
        raise ValueError("period_hours must be positive")

    required = {"timestamp", "zone_id", "demand"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing demand columns: {', '.join(sorted(missing))}")

    ordered = frame.sort(["zone_id", "timestamp"])
    return (
        ordered.with_columns(
            pl.col("demand")
            .shift(period_hours)
            .over("zone_id")
            .cast(pl.Float64)
            .alias(prediction_column)
        )
        .sort(["timestamp", "zone_id"])
    )
