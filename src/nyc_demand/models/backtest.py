from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from nyc_demand.models.horizons import make_horizon_dataset
from nyc_demand.models.lightgbm_model import train_lightgbm
from nyc_demand.models.metrics import evaluate
from nyc_demand.models.splits import expanding_window_folds, split_frame


@dataclass(frozen=True)
class FoldMetrics:
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


def backtest_lightgbm(
    feature_frame: pl.DataFrame,
    *,
    horizon_hours: int,
    train_days: int,
    validation_days: int,
    step_days: int,
    params: dict[str, Any] | None = None,
) -> list[FoldMetrics]:
    """Run leakage-safe expanding-window validation for one direct forecast horizon."""
    supervised = make_horizon_dataset(feature_frame, horizon_hours=horizon_hours)
    folds = expanding_window_folds(
        supervised,
        train_days=train_days,
        validation_days=validation_days,
        step_days=step_days,
    )
    if not folds:
        raise ValueError("Not enough temporal coverage to create a validation fold")

    results: list[FoldMetrics] = []
    for fold in folds:
        train, validation = split_frame(supervised, fold)

        # The label attached to a training origin must also predate the validation boundary.
        train = train.filter(pl.col("forecast_timestamp") < fold.validation_start)
        validation = validation.filter(pl.col("forecast_timestamp") < fold.validation_end)
        if train.is_empty() or validation.is_empty():
            continue

        fitted = train_lightgbm(
            train,
            target_column="target_demand",
            params=params,
        )
        validation = validation.drop_nulls([*fitted.feature_names, "target_demand"])
        if validation.is_empty():
            continue

        prediction = fitted.predict(validation)
        actual = validation["target_demand"].to_numpy().astype(float, copy=False)
        scores = evaluate(actual, prediction)
        results.append(
            FoldMetrics(
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
        raise ValueError("All validation folds became empty after leakage safeguards")
    return results


def summarize_folds(results: list[FoldMetrics]) -> dict[str, float]:
    if not results:
        raise ValueError("At least one fold result is required")

    return {
        metric: float(np.mean([getattr(result, metric) for result in results]))
        for metric in ("mae", "rmse", "wape", "smape")
    }
