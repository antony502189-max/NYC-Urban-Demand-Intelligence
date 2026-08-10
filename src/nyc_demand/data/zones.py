from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl

from nyc_demand.config import ensure_directory

TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
REQUIRED_ZONE_COLUMNS = ("LocationID", "Borough", "Zone", "service_zone")


def validate_zone_lookup(frame: pl.DataFrame) -> None:
    missing = sorted(set(REQUIRED_ZONE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing taxi-zone columns: {', '.join(missing)}")

    if frame.is_empty():
        raise ValueError("Taxi-zone lookup must not be empty")

    duplicated = frame.select(pl.col("LocationID").is_duplicated().any()).item()
    if duplicated:
        raise ValueError("Taxi-zone lookup contains duplicate LocationID values")

    invalid_ids = frame.filter(
        pl.col("LocationID").is_null() | ~pl.col("LocationID").is_between(1, 265)
    )
    if invalid_ids.height:
        raise ValueError("Taxi-zone lookup contains invalid LocationID values")


def normalize_zone_lookup(frame: pl.DataFrame) -> pl.DataFrame:
    validate_zone_lookup(frame)
    return (
        frame.select(REQUIRED_ZONE_COLUMNS)
        .with_columns(
            pl.col("LocationID").cast(pl.Int16).alias("zone_id"),
            pl.col("Borough").cast(pl.String).str.strip_chars().alias("borough"),
            pl.col("Zone").cast(pl.String).str.strip_chars().alias("zone_name"),
            pl.col("service_zone")
            .cast(pl.String)
            .fill_null("Unknown")
            .str.strip_chars()
            .alias("service_zone"),
        )
        .select("zone_id", "borough", "zone_name", "service_zone")
        .sort("zone_id")
    )


def download_zone_lookup(
    destination: str | Path = "data/reference/taxi_zone_lookup.csv",
    *,
    url: str = TAXI_ZONE_LOOKUP_URL,
    timeout_seconds: float = 30.0,
) -> Path:
    target = Path(destination)
    if not target.is_absolute():
        target = ensure_directory(target.parent) / target.name
    else:
        target.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        target.write_bytes(response.content)

    frame = pl.read_csv(target)
    normalize_zone_lookup(frame)
    return target


def load_zone_lookup(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return normalize_zone_lookup(pl.read_csv(source))
