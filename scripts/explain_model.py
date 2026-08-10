from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nyc_demand.models.explainability import global_shap_importance
from nyc_demand.models.persistence import load_model_bundle


def explain_model(
    model_dir: str | Path,
    feature_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = 5000,
    top_n: int = 30,
) -> dict[str, object]:
    model = load_model_bundle(model_dir)
    frame = pl.read_parquet(feature_path).drop_nulls(list(model.feature_names))
    importance = global_shap_importance(
        model.booster,
        frame,
        feature_names=model.feature_names,
        max_rows=max_rows,
    )
    payload: dict[str, object] = {
        "horizon_hours": model.horizon_hours,
        "rows_explained": min(frame.height, max_rows),
        "features": importance.head(top_n).to_dicts(),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP importance for a stored model bundle")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--features", required=True, help="Feature Parquet file")
    parser.add_argument("--output", required=True, help="Destination JSON report")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = explain_model(
        args.model_dir,
        args.features,
        args.output,
        max_rows=args.max_rows,
        top_n=args.top_n,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
