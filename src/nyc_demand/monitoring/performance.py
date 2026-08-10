from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from nyc_demand.models.diagnostics import residual_summary
from nyc_demand.models.metrics import evaluate


@dataclass(frozen=True)
class PerformanceSnapshot:
    observations: int
    mae: float
    rmse: float
    wape: float
    smape: float
    bias: float
    p90_absolute_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def performance_snapshot(
    frame: pl.DataFrame,
    *,
    actual_column: str = "actual",
    prediction_column: str = "prediction",
) -> PerformanceSnapshot:
    required = {actual_column, prediction_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing performance columns: {', '.join(missing)}")
    if frame.is_empty():
        raise ValueError("Performance frame must not be empty")

    clean = frame.drop_nulls([actual_column, prediction_column])
    if clean.is_empty():
        raise ValueError("No scored rows remain after dropping null values")

    actual = clean[actual_column].to_numpy().astype(float, copy=False)
    prediction = clean[prediction_column].to_numpy().astype(float, copy=False)
    scores = evaluate(actual, prediction)
    residuals = residual_summary(actual, prediction)
    return PerformanceSnapshot(
        observations=clean.height,
        mae=scores["mae"],
        rmse=scores["rmse"],
        wape=scores["wape"],
        smape=scores["smape"],
        bias=residuals["bias"],
        p90_absolute_error=residuals["p90_absolute_error"],
    )


def performance_by_day(
    frame: pl.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    actual_column: str = "actual",
    prediction_column: str = "prediction",
) -> pl.DataFrame:
    required = {timestamp_column, actual_column, prediction_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing daily-performance columns: {', '.join(missing)}")
    if frame.is_empty():
        raise ValueError("Performance frame must not be empty")

    clean = frame.drop_nulls([timestamp_column, actual_column, prediction_column]).with_columns(
        (pl.col(actual_column) - pl.col(prediction_column)).alias("_residual"),
        (pl.col(actual_column) - pl.col(prediction_column)).abs().alias("_absolute_error"),
    )
    return (
        clean.with_columns(pl.col(timestamp_column).dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.len().alias("observations"),
            pl.col("_absolute_error").mean().alias("mae"),
            pl.col("_residual").mean().alias("bias"),
            pl.col("_absolute_error").sum().alias("_absolute_error_sum"),
            pl.col(actual_column).abs().sum().alias("_absolute_actual_sum"),
        )
        .with_columns(
            pl.when(pl.col("_absolute_actual_sum") == 0)
            .then(pl.lit(float("inf")))
            .otherwise(pl.col("_absolute_error_sum") / pl.col("_absolute_actual_sum"))
            .alias("wape")
        )
        .drop("_absolute_error_sum", "_absolute_actual_sum")
        .sort("date")
    )


def classify_performance_regression(
    reference_wape: float,
    current_wape: float,
    *,
    warning_ratio: float = 1.15,
    critical_ratio: float = 1.30,
) -> str:
    if not np.isfinite(reference_wape) or reference_wape < 0:
        raise ValueError("reference_wape must be finite and non-negative")
    if not np.isfinite(current_wape) or current_wape < 0:
        raise ValueError("current_wape must be finite and non-negative")
    if not 1.0 <= warning_ratio < critical_ratio:
        raise ValueError("threshold ratios must satisfy 1 <= warning < critical")

    if reference_wape == 0:
        return "stable" if current_wape == 0 else "critical"

    ratio = current_wape / reference_wape
    if ratio >= critical_ratio:
        return "critical"
    if ratio >= warning_ratio:
        return "watch"
    return "stable"
