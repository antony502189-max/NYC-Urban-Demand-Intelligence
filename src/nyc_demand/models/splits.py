from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass(frozen=True)
class TimeFold:
    fold: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime


def expanding_window_folds(
    frame: pl.DataFrame,
    *,
    train_days: int,
    validation_days: int,
    step_days: int,
) -> list[TimeFold]:
    """Create chronological expanding-window validation folds."""
    if min(train_days, validation_days, step_days) <= 0:
        raise ValueError("window sizes must be positive")
    if "timestamp" not in frame.columns or frame.is_empty():
        raise ValueError("frame must contain non-empty timestamp data")

    start = frame["timestamp"].min()
    end = frame["timestamp"].max()
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise TypeError("timestamp must contain Python-compatible datetime values")

    train_delta = timedelta(days=train_days)
    validation_delta = timedelta(days=validation_days)
    step_delta = timedelta(days=step_days)

    validation_start = start + train_delta
    folds: list[TimeFold] = []
    fold_number = 1

    while validation_start + validation_delta <= end + timedelta(hours=1):
        validation_end = validation_start + validation_delta
        folds.append(
            TimeFold(
                fold=fold_number,
                train_start=start,
                train_end=validation_start,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
        fold_number += 1
        validation_start += step_delta

    return folds


def split_frame(frame: pl.DataFrame, fold: TimeFold) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = frame.filter(
        (pl.col("timestamp") >= fold.train_start)
        & (pl.col("timestamp") < fold.train_end)
    )
    validation = frame.filter(
        (pl.col("timestamp") >= fold.validation_start)
        & (pl.col("timestamp") < fold.validation_end)
    )
    return train, validation
