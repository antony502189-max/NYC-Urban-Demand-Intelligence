from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nyc_demand.config import load_yaml
from nyc_demand.features.builder import build_features
from nyc_demand.models.backtest import backtest_lightgbm, summarize_folds
from nyc_demand.models.horizons import make_horizon_dataset
from nyc_demand.models.lightgbm_model import load_lightgbm_params, train_lightgbm
from nyc_demand.models.persistence import save_model_bundle
from nyc_demand.tracking.mlflow_tracker import log_backtest_run


def train_pipeline(
    input_path: str | Path,
    *,
    horizon_hours: int,
    model_dir: str | Path,
    report_path: str | Path,
    use_mlflow: bool = False,
) -> dict[str, object]:
    demand = pl.read_parquet(input_path)
    features = build_features(demand)

    config = load_yaml("configs/model.yaml")
    validation = config["validation"]
    params = load_lightgbm_params()

    folds = backtest_lightgbm(
        features,
        horizon_hours=horizon_hours,
        train_days=int(validation["train_days"]),
        validation_days=int(validation["validation_days"]),
        step_days=int(validation["step_days"]),
        params=params,
    )
    summary = summarize_folds(folds)

    supervised = make_horizon_dataset(features, horizon_hours=horizon_hours)
    fitted = train_lightgbm(
        supervised,
        target_column="target_demand",
        params=params,
    )
    bundle = save_model_bundle(
        fitted,
        model_dir,
        horizon_hours=horizon_hours,
    )

    payload: dict[str, object] = {
        "horizon_hours": horizon_hours,
        "model_dir": str(bundle),
        "summary": summary,
        "folds": [fold.to_dict() for fold in folds],
        "feature_names": list(fitted.feature_names),
    }

    if use_mlflow:
        payload["mlflow_run_id"] = log_backtest_run(
            folds,
            params=params,
            tags={"model": "lightgbm", "task": "taxi-zone-demand"},
        )

    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate an NYC demand model")
    parser.add_argument("--input", required=True, help="Hourly demand Parquet file")
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_pipeline(
        args.input,
        horizon_hours=args.horizon,
        model_dir=args.model_dir,
        report_path=args.report,
        use_mlflow=args.mlflow,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
