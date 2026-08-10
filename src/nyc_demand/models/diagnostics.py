from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

ArrayLike = Sequence[float] | np.ndarray


def residual_summary(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Residual arrays must be non-empty and have identical shapes")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Residual arrays must contain only finite values")

    residual = actual - predicted
    absolute = np.abs(residual)
    return {
        "bias": float(np.mean(residual)),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "underprediction_share": float(np.mean(residual > 0)),
        "overprediction_share": float(np.mean(residual < 0)),
    }


def error_frame(
    frame: pl.DataFrame,
    *,
    actual_column: str = "target_demand",
    prediction_column: str = "prediction",
) -> pl.DataFrame:
    required = {"timestamp", "zone_id", actual_column, prediction_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing diagnostic columns: {', '.join(missing)}")

    return frame.with_columns(
        (pl.col(actual_column) - pl.col(prediction_column)).alias("residual"),
        (pl.col(actual_column) - pl.col(prediction_column)).abs().alias("absolute_error"),
    )


def zone_error_table(
    frame: pl.DataFrame,
    *,
    actual_column: str = "target_demand",
    prediction_column: str = "prediction",
) -> pl.DataFrame:
    errors = error_frame(
        frame,
        actual_column=actual_column,
        prediction_column=prediction_column,
    )
    return (
        errors.group_by("zone_id")
        .agg(
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("residual").mean().alias("bias"),
            pl.col(actual_column).sum().alias("actual_demand"),
            pl.len().alias("observations"),
        )
        .sort("mae", descending=True)
    )


def demand_band_error_table(
    frame: pl.DataFrame,
    *,
    actual_column: str = "target_demand",
    prediction_column: str = "prediction",
) -> pl.DataFrame:
    errors = error_frame(
        frame,
        actual_column=actual_column,
        prediction_column=prediction_column,
    ).with_columns(
        pl.when(pl.col(actual_column) == 0)
        .then(pl.lit("zero"))
        .when(pl.col(actual_column) <= 5)
        .then(pl.lit("low"))
        .when(pl.col(actual_column) <= 20)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
        .alias("demand_band")
    )

    order = pl.when(pl.col("demand_band") == "zero").then(0).when(
        pl.col("demand_band") == "low"
    ).then(1).when(pl.col("demand_band") == "medium").then(2).otherwise(3)

    return (
        errors.group_by("demand_band")
        .agg(
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("residual").mean().alias("bias"),
            pl.len().alias("observations"),
        )
        .with_columns(order.alias("_order"))
        .sort("_order")
        .drop("_order")
    )
