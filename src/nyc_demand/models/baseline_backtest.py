from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from nyc_demand.models.metrics import evaluate
from nyc_demand.models.splits import expanding_window_folds, split_frame


@dataclass(frozen=True)
class BaselineFoldMetrics:
    fold: int
    horizon_hours: int
    train_rows: int
    validation_rows: int
    mae: float
    rmse: float
    wape: float
    smape: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def attach_direct_seasonal_baseline(
    frame: pl.DataFrame,
    *,
    horizon_hours: int,
    seasonal_period_hours: int = 168,
    target_column: str = "target_demand",
    prediction_column: str = "prediction",
) -> pl.DataFrame:
    """Attach a direct-horizon weekly baseline using only information known at origin time."""
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if seasonal_period_hours <= horizon_hours:
        raise ValueError("seasonal_period_hours must be greater than horizon_hours")

    required = {"timestamp", "zone_id", "demand"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing demand columns: {', '.join(missing)}")

    ordered = frame.sort(["zone_id", "timestamp"])
    history_lag = seasonal_period_hours - horizon_hours
    return (
        ordered.with_columns(
            pl.col("demand")
            .shift(-horizon_hours)
            .over("zone_id")
            .alias(target_column),
            pl.col("demand")
            .shift(history_lag)
            .over("zone_id")
            .cast(pl.Float64)
            .alias(prediction_column),
            (pl.col("timestamp") + pl.duration(hours=horizon_hours)).alias(
                "forecast_timestamp"
            ),
        )
        .drop_nulls([target_column, prediction_column])
        .sort(["timestamp", "zone_id"])
    )


def backtest_seasonal_baseline(
    frame: pl.DataFrame,
    *,
    horizon_hours: int,
    train_days: int,
    validation_days: int,
    step_days: int,
    seasonal_period_hours: int = 168,
) -> list[BaselineFoldMetrics]:
    supervised = attach_direct_seasonal_baseline(
        frame,
        horizon_hours=horizon_hours,
        seasonal_period_hours=seasonal_period_hours,
    )
    folds = expanding_window_folds(
        supervised,
        train_days=train_days,
        validation_days=validation_days,
        step_days=step_days,
    )
    if not folds:
        raise ValueError("Not enough temporal coverage to create a validation fold")

    results: list[BaselineFoldMetrics] = []
    for fold in folds:
        train, validation = split_frame(supervised, fold)
        validation = validation.filter(pl.col("forecast_timestamp") < fold.validation_end)
        if train.is_empty() or validation.is_empty():
            continue

        actual = validation["target_demand"].to_numpy().astype(float, copy=False)
        prediction = validation["prediction"].to_numpy().astype(float, copy=False)
        scores = evaluate(actual, prediction)
        results.append(
            BaselineFoldMetrics(
                fold=fold.fold,
                horizon_hours=horizon_hours,
                train_rows=train.height,
                validation_rows=validation.height,
                mae=scores["mae"],
                rmse=scores["rmse"],
                wape=scores["wape"],
                smape=scores["smape"],
            )
        )

    if not results:
        raise ValueError("All baseline validation folds became empty")
    return results


def summarize_baseline_folds(results: list[BaselineFoldMetrics]) -> dict[str, float]:
    if not results:
        raise ValueError("At least one fold result is required")
    return {
        metric: float(np.mean([getattr(result, metric) for result in results]))
        for metric in ("mae", "rmse", "wape", "smape")
    }
