from __future__ import annotations

from dataclasses import dataclass

import optuna
import polars as pl

from nyc_demand.models.backtest import backtest_lightgbm, summarize_folds


@dataclass(frozen=True)
class TuningResult:
    best_value: float
    best_params: dict[str, int | float | str]
    trials: int


def tune_lightgbm(
    feature_frame: pl.DataFrame,
    *,
    horizon_hours: int,
    train_days: int,
    validation_days: int,
    step_days: int,
    n_trials: int = 30,
    seed: int = 42,
) -> TuningResult:
    """Tune LightGBM against mean expanding-window WAPE."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, int | float | str] = {
            "objective": "poisson",
            "n_estimators": trial.suggest_int("n_estimators", 250, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", -1, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
        }
        folds = backtest_lightgbm(
            feature_frame,
            horizon_hours=horizon_hours,
            train_days=train_days,
            validation_days=validation_days,
            step_days=step_days,
            params=params,
        )
        return summarize_folds(folds)["wape"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return TuningResult(
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        trials=len(study.trials),
    )
