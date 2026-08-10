from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from nyc_demand.features.builder import build_features


def build_feature_file(input_path: str | Path, output_path: str | Path) -> Path:
    demand = pl.read_parquet(input_path)
    features = build_features(demand)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(destination, compression="zstd", statistics=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe feature Parquet")
    parser.add_argument("--input", required=True, help="Hourly demand Parquet file")
    parser.add_argument("--output", required=True, help="Feature Parquet destination")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(build_feature_file(args.input, args.output))


if __name__ == "__main__":
    main()
