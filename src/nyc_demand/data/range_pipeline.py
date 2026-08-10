from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True, order=True)
class YearMonth:
    year: int
    month: int

    def __post_init__(self) -> None:
        if self.year < 2009:
            raise ValueError("year must be 2009 or later")
        if self.month not in range(1, 13):
            raise ValueError("month must be between 1 and 12")

    def next(self) -> YearMonth:
        if self.month == 12:
            return YearMonth(self.year + 1, 1)
        return YearMonth(self.year, self.month + 1)

    @property
    def label(self) -> str:
        return f"{self.year}-{self.month:02d}"


def month_range(start: YearMonth, end: YearMonth) -> list[YearMonth]:
    """Return an inclusive chronological month range."""
    if end < start:
        raise ValueError("end month must not precede start month")

    months: list[YearMonth] = []
    current = start
    while current <= end:
        months.append(current)
        current = current.next()
    return months


def merge_hourly_demand(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Merge monthly marts and restore a complete timestamp x zone grid.

    A zone can be absent from an individual monthly source when it has no pickups
    during that month. Downstream lag features are row-based within each zone, so
    leaving those gaps would silently make non-consecutive hours look adjacent.
    The merged mart therefore reindexes the union of observed zones over the full
    hourly range and fills missing demand with zero.
    """
    if not frames:
        raise ValueError("At least one monthly demand frame is required")

    required = {"timestamp", "zone_id", "demand"}
    for index, frame in enumerate(frames):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Monthly demand frame {index} is missing columns: {', '.join(missing)}"
            )

    counts = (
        pl.concat(frames, how="vertical_relaxed")
        .group_by(["timestamp", "zone_id"])
        .agg(pl.col("demand").sum().cast(pl.Int32).alias("demand"))
    )
    if counts.is_empty():
        raise ValueError("Merged demand frame must not be empty")

    start = counts["timestamp"].min()
    end = counts["timestamp"].max()
    if start is None or end is None:
        raise ValueError("Unable to determine merged demand time range")

    hours = pl.DataFrame(
        {"timestamp": pl.datetime_range(start, end, interval="1h", eager=True)}
    )
    zones = counts.select("zone_id").unique().sort("zone_id")
    grid = hours.join(zones, how="cross")

    return (
        grid.join(counts, on=["timestamp", "zone_id"], how="left")
        .with_columns(pl.col("demand").fill_null(0).cast(pl.Int32))
        .sort(["timestamp", "zone_id"])
    )


def write_merged_demand(frames: list[pl.DataFrame], output_path: str | Path) -> Path:
    merged = merge_hourly_demand(frames)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(destination, compression="zstd", statistics=True)
    return destination
