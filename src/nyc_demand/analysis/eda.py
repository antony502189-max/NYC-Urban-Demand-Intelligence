from __future__ import annotations

import polars as pl


REQUIRED_DEMAND_COLUMNS = frozenset({"timestamp", "zone_id", "demand"})


def _validate_demand_frame(frame: pl.DataFrame) -> None:
    missing = sorted(REQUIRED_DEMAND_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing demand columns: {', '.join(missing)}")
    if frame.is_empty():
        raise ValueError("Demand frame must not be empty")


def demand_overview(frame: pl.DataFrame) -> dict[str, int | float | str]:
    """Return compact dataset-level coverage and demand statistics."""
    _validate_demand_frame(frame)
    start = frame["timestamp"].min()
    end = frame["timestamp"].max()
    return {
        "rows": frame.height,
        "zones": frame["zone_id"].n_unique(),
        "start": str(start),
        "end": str(end),
        "total_demand": int(frame["demand"].sum()),
        "mean_hourly_zone_demand": float(frame["demand"].mean()),
        "zero_demand_share": float((frame["demand"] == 0).mean()),
    }


def hourly_profile(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate demand by hour-of-day across all zones and dates."""
    _validate_demand_frame(frame)
    return (
        frame.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(
            pl.col("demand").sum().alias("total_demand"),
            pl.col("demand").mean().alias("mean_zone_demand"),
            pl.len().alias("observations"),
        )
        .sort("hour")
    )


def weekday_profile(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate demand by ISO weekday where Monday=1 and Sunday=7."""
    _validate_demand_frame(frame)
    return (
        frame.with_columns(pl.col("timestamp").dt.weekday().alias("weekday"))
        .group_by("weekday")
        .agg(
            pl.col("demand").sum().alias("total_demand"),
            pl.col("demand").mean().alias("mean_zone_demand"),
            pl.len().alias("observations"),
        )
        .sort("weekday")
    )


def top_zones(
    frame: pl.DataFrame,
    *,
    n: int = 20,
    zone_lookup: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Rank pickup zones by total observed demand, optionally adding human-readable labels."""
    _validate_demand_frame(frame)
    if n <= 0:
        raise ValueError("n must be positive")

    ranked = (
        frame.group_by("zone_id")
        .agg(
            pl.col("demand").sum().alias("total_demand"),
            pl.col("demand").mean().alias("mean_hourly_demand"),
        )
        .sort("total_demand", descending=True)
        .head(n)
    )
    if zone_lookup is None:
        return ranked

    required = {"zone_id", "borough", "zone_name"}
    missing = sorted(required.difference(zone_lookup.columns))
    if missing:
        raise ValueError(f"Missing zone lookup columns: {', '.join(missing)}")

    return ranked.join(
        zone_lookup.select("zone_id", "borough", "zone_name"),
        on="zone_id",
        how="left",
    )
