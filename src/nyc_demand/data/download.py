from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from nyc_demand.config import ensure_directory, load_yaml

DEFAULT_CONFIG = "configs/data.yaml"
CHUNK_SIZE = 1024 * 1024


def build_url(year: int, month: int, config_path: str | Path = DEFAULT_CONFIG) -> str:
    if year < 2009:
        raise ValueError("NYC TLC Yellow Taxi trip records are not expected before 2009")
    if month not in range(1, 13):
        raise ValueError("month must be between 1 and 12")

    config = load_yaml(config_path)
    template = config["dataset"]["url_template"]
    return str(template).format(year=year, month=month)


def target_path(year: int, month: int, config_path: str | Path = DEFAULT_CONFIG) -> Path:
    config = load_yaml(config_path)
    raw_dir = ensure_directory(config["paths"]["raw_dir"])
    return raw_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"


def download_month(
    year: int,
    month: int,
    *,
    overwrite: bool = False,
    config_path: str | Path = DEFAULT_CONFIG,
    timeout_seconds: float = 120.0,
) -> Path:
    """Download one official TLC Yellow Taxi Parquet file atomically."""
    destination = target_path(year, month, config_path)
    if destination.exists() and not overwrite:
        return destination

    url = build_url(year, month, config_path)
    partial = destination.with_suffix(destination.suffix + ".part")

    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=timeout_seconds,
            headers={"User-Agent": "NYC-Urban-Demand-Intelligence/0.1"},
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_bytes(CHUNK_SIZE):
                    output.write(chunk)

        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NYC TLC Yellow Taxi trip data")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = download_month(
        args.year,
        args.month,
        overwrite=args.overwrite,
        config_path=args.config,
    )
    print(path)


if __name__ == "__main__":
    main()
