from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from nyc_demand.models.baseline_backtest import attach_direct_seasonal_baseline
from nyc_demand.models.dataset import infer_feature_columns
from nyc_demand.models.horizons import make_horizon_dataset
from nyc_demand.models.lightgbm_model import train_lightgbm
from nyc_demand.models.metrics import evaluate
from nyc_demand.models.splits import expanding_window_folds, split_frame


@dataclass(frozen=True)
class BenchmarkFold:
    fold: int
    horizon_hours: int
    validation_rows: int
    model_mae: float
    baseline_mae: float
    model_wape: float
    baseline_wape: float

    @property
    def mae_improvement(self) -> float:
        if self.baseline_mae == 0:
            return 0.0
        return 1.0 - self.model_mae / self.baseline_mae

    @property
    def wape_improvement(self) -> float:
        if self.baseline_wape == 0:
            return 0.0
        return 1.0 - self.model_wape / self.baseline_wape

    def to_dict(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = asdict(self)
        payload["mae_improvement"] = self.mae_improvement
        payload["wape_improvement"] = self.wape_improvement
        return payload


def compare_lightgbm_to_seasonal_baseline(
    feature_frame: pl.DataFrame,
    *,
    horizon_hours: int,
    train_days: int,
    validation_days: int,
    step_days: int,
    seasonal_period_hours: int = 168,
    params: dict[str, Any] | None = None,
) -> list[BenchmarkFold]:
    """Compare LightGBM and seasonal naive on identical validation rows and folds."""
    model_supervised = make_horizon_dataset(feature_frame, horizon_hours=horizon_hours)
    feature_columns = infer_feature_columns(model_supervised)
    benchmark = attach_direct_seasonal_baseline(
        feature_frame,
        horizon_hours=horizon_hours,
        seasonal_period_hours=seasonal_period_hours,
    ).drop_nulls([*feature_columns, "target_demand", "prediction"])

    folds = expanding_window_folds(
        benchmark,
        train_days=train_days,
        validation_days=validation_days,
        step_days=step_days,
    )
    if not folds:
        raise ValueError("Not enough temporal coverage to benchmark models")

    results: list[BenchmarkFold] = []
    for fold in folds:
        train, validation = split_frame(benchmark, fold)
        train = train.filter(pl.col("forecast_timestamp") < fold.validation_start)
        validation = validation.filter(pl.col("forecast_timestamp") < fold.validation_end)
        if train.is_empty() or validation.is_empty():
            continue

        fitted = train_lightgbm(
            train,
            feature_columns=feature_columns,
            target_column="target_demand",
            params=params,
        )
        actual = validation["target_demand"].to_numpy().astype(float, copy=False)
        model_prediction = fitted.predict(validation)
        baseline_prediction = validation["prediction"].to_numpy().astype(float, copy=False)
        model_scores = evaluate(actual, model_prediction)
        baseline_scores = evaluate(actual, baseline_prediction)

        results.append(
            BenchmarkFold(
                fold=fold.fold,
                horizon_hours=horizon_hours,
                validation_rows=validation.height,
                model_mae=model_scores["mae"],
                baseline_mae=baseline_scores["mae"],
                model_wape=model_scores["wape"],
                baseline_wape=baseline_scores["wape"],
            )
        )

    if not results:
        raise ValueError("All benchmark folds became empty")
    return results


def summarize_benchmark(results: list[BenchmarkFold]) -> dict[str, float]:
    if not results:
        raise ValueError("At least one benchmark fold is required")
    return {
        "model_mae": float(np.mean([fold.model_mae for fold in results])),
        "baseline_mae": float(np.mean([fold.baseline_mae for fold in results])),
        "mae_improvement": float(np.mean([fold.mae_improvement for fold in results])),
        "model_wape": float(np.mean([fold.model_wape for fold in results])),
        "baseline_wape": float(np.mean([fold.baseline_wape for fold in results])),
        "wape_improvement": float(np.mean([fold.wape_improvement for fold in results])),
    }
