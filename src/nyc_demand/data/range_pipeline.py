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
    """Merge monthly hourly-demand marts into one unique timestamp-zone table."""
    if not frames:
        raise ValueError("At least one monthly demand frame is required")

    required = {"timestamp", "zone_id", "demand"}
    for index, frame in enumerate(frames):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Monthly demand frame {index} is missing columns: {', '.join(missing)}"
            )

    merged = (
        pl.concat(frames, how="vertical_relaxed")
        .group_by(["timestamp", "zone_id"])
        .agg(pl.col("demand").sum().cast(pl.Int32).alias("demand"))
        .sort(["timestamp", "zone_id"])
    )
    if merged.is_empty():
        raise ValueError("Merged demand frame must not be empty")
    return merged


def write_merged_demand(frames: list[pl.DataFrame], output_path: str | Path) -> Path:
    merged = merge_hourly_demand(frames)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(destination, compression="zstd", statistics=True)
    return destination
