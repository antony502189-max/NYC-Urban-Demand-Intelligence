from __future__ import annotations

import polars as pl


BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island", "EWR", "Unknown")
SERVICE_ZONES = ("Boro Zone", "Yellow Zone", "Airports", "Unknown")


def add_zone_features(
    frame: pl.DataFrame,
    zone_lookup: pl.DataFrame,
) -> pl.DataFrame:
    """Attach stable one-hot taxi-zone context without imposing ordinal category semantics."""
    if "zone_id" not in frame.columns:
        raise ValueError("Feature frame must contain zone_id")

    required = {"zone_id", "borough", "service_zone"}
    missing = sorted(required.difference(zone_lookup.columns))
    if missing:
        raise ValueError(f"Missing zone lookup columns: {', '.join(missing)}")

    lookup = zone_lookup.select("zone_id", "borough", "service_zone").unique("zone_id")
    enriched = frame.join(lookup, on="zone_id", how="left").with_columns(
        pl.col("borough").fill_null("Unknown"),
        pl.col("service_zone").fill_null("Unknown"),
    )

    expressions: list[pl.Expr] = []
    for borough in BOROUGHS:
        safe_name = borough.lower().replace(" ", "_")
        expressions.append(
            (pl.col("borough") == borough).cast(pl.Int8).alias(f"borough_{safe_name}")
        )
    for service_zone in SERVICE_ZONES:
        safe_name = service_zone.lower().replace(" ", "_")
        expressions.append(
            (pl.col("service_zone") == service_zone)
            .cast(pl.Int8)
            .alias(f"service_zone_{safe_name}")
        )

    return enriched.with_columns(expressions).drop("borough", "service_zone")
