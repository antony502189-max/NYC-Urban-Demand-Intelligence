from __future__ import annotations

from collections.abc import Iterable

import polars as pl

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
DEFAULT_WINDOWS = (3, 6, 12, 24, 72, 168)


def add_calendar_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("timestamp").dt.weekday().cast(pl.Int8).alias("day_of_week"),
        pl.col("timestamp").dt.day().cast(pl.Int8).alias("day_of_month"),
        pl.col("timestamp").dt.month().cast(pl.Int8).alias("month"),
        (pl.col("timestamp").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
    )


def add_lag_features(
    frame: pl.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pl.DataFrame:
    expressions = [
        pl.col("demand").shift(lag).over("zone_id").alias(f"demand_lag_{lag}h")
        for lag in lags
    ]
    return frame.with_columns(expressions)


def add_rolling_features(
    frame: pl.DataFrame,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for window in windows:
        expressions.extend(
            [
                pl.col("demand")
                .shift(1)
                .rolling_mean(window_size=window, min_samples=1)
                .over("zone_id")
                .alias(f"demand_roll_mean_{window}h"),
                pl.col("demand")
                .shift(1)
                .rolling_std(window_size=window, min_samples=2)
                .over("zone_id")
                .alias(f"demand_roll_std_{window}h"),
            ]
        )
    return frame.with_columns(expressions)


def build_features(
    demand: pl.DataFrame,
    *,
    lags: Iterable[int] = DEFAULT_LAGS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pl.DataFrame:
    """Create forecasting features using only historical observations per zone."""
    required = {"timestamp", "zone_id", "demand"}
    missing = required.difference(demand.columns)
    if missing:
        raise ValueError(f"Missing demand columns: {', '.join(sorted(missing))}")

    ordered = demand.sort(["zone_id", "timestamp"])
    featured = add_calendar_features(ordered)
    featured = add_lag_features(featured, lags)
    featured = add_rolling_features(featured, windows)
    return featured.sort(["timestamp", "zone_id"])
