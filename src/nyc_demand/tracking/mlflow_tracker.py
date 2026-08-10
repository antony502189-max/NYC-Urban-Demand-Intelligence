from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nyc_demand.models.backtest import FoldMetrics, summarize_folds


def log_backtest_run(
    results: list[FoldMetrics],
    *,
    params: Mapping[str, Any],
    experiment_name: str = "nyc-demand-forecasting",
    tags: Mapping[str, str] | None = None,
) -> str:
    """Log cross-validation parameters, aggregate metrics, and fold diagnostics to MLflow."""
    if not results:
        raise ValueError("At least one backtest result is required")

    import mlflow

    mlflow.set_experiment(experiment_name)
    summary = summarize_folds(results)
    horizon = results[0].horizon_hours

    with mlflow.start_run() as run:
        mlflow.log_params({**dict(params), "horizon_hours": horizon})
        mlflow.log_metrics({f"cv_{name}": value for name, value in summary.items()})
        mlflow.log_dict(
            {"folds": [result.to_dict() for result in results]},
            "backtest/folds.json",
        )
        if tags:
            mlflow.set_tags(dict(tags))
        return run.info.run_id
