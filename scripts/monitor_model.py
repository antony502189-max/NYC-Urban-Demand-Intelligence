from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nyc_demand.monitoring.report import monitoring_report


def build_monitoring_report(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path,
    *,
    feature_names: list[str],
    actual_column: str = "actual",
    prediction_column: str = "prediction",
) -> dict[str, object]:
    reference = pl.read_parquet(reference_path)
    current = pl.read_parquet(current_path)
    report = monitoring_report(
        reference,
        current,
        feature_names=feature_names,
        actual_column=actual_column,
        prediction_column=prediction_column,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reference and current model-scoring windows")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--features",
        required=True,
        help="Comma-separated numeric feature names to monitor with PSI",
    )
    parser.add_argument("--actual-column", default="actual")
    parser.add_argument("--prediction-column", default="prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = [name.strip() for name in args.features.split(",") if name.strip()]
    report = build_monitoring_report(
        args.reference,
        args.current,
        args.output,
        feature_names=features,
        actual_column=args.actual_column,
        prediction_column=args.prediction_column,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
