from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nyc_demand.config import load_yaml
from nyc_demand.features.builder import build_features
from nyc_demand.models.benchmark import compare_lightgbm_to_seasonal_baseline, summarize_benchmark
from nyc_demand.models.lightgbm_model import load_lightgbm_params


def benchmark_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    horizon_hours: int,
) -> dict[str, object]:
    demand = pl.read_parquet(input_path)
    features = build_features(demand)
    config = load_yaml("configs/model.yaml")
    validation = config["validation"]
    baseline = config["baseline"]

    folds = compare_lightgbm_to_seasonal_baseline(
        features,
        horizon_hours=horizon_hours,
        train_days=int(validation["train_days"]),
        validation_days=int(validation["validation_days"]),
        step_days=int(validation["step_days"]),
        seasonal_period_hours=int(baseline["seasonal_period_hours"]),
        params=load_lightgbm_params(),
    )
    payload: dict[str, object] = {
        "horizon_hours": horizon_hours,
        "summary": summarize_benchmark(folds),
        "folds": [fold.to_dict() for fold in folds],
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LightGBM against weekly seasonal naive")
    parser.add_argument("--input", required=True, help="Hourly demand Parquet file")
    parser.add_argument("--output", required=True, help="Destination JSON report")
    parser.add_argument("--horizon", type=int, choices=[1, 6, 24], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = benchmark_pipeline(args.input, args.output, horizon_hours=args.horizon)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
