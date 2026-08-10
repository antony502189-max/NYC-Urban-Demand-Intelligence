from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


NON_FEATURE_COLUMNS = frozenset({"timestamp", "demand"})


@dataclass(frozen=True)
class TrainingMatrix:
    features: np.ndarray
    target: np.ndarray
    feature_names: tuple[str, ...]


def infer_feature_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    """Infer numeric model features while excluding timestamp and target columns."""
    columns: list[str] = []
    for name, dtype in frame.schema.items():
        if name in NON_FEATURE_COLUMNS:
            continue
        if dtype.is_numeric():
            columns.append(name)

    if not columns:
        raise ValueError("No numeric feature columns are available")
    return tuple(columns)


def build_training_matrix(
    frame: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    target_column: str = "demand",
) -> TrainingMatrix:
    """Convert a feature frame into a finite NumPy matrix for model training."""
    if frame.is_empty():
        raise ValueError("Training frame must not be empty")
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    selected = feature_columns or infer_feature_columns(frame)
    missing = set(selected).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")

    clean = frame.drop_nulls([*selected, target_column])
    if clean.is_empty():
        raise ValueError("No complete rows remain after dropping null model inputs")

    features = clean.select(selected).to_numpy().astype(np.float64, copy=False)
    target = clean[target_column].to_numpy().astype(np.float64, copy=False)

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values")
    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or infinite values")
    if np.any(target < 0):
        raise ValueError("Demand target must be non-negative")

    return TrainingMatrix(
        features=features,
        target=target,
        feature_names=tuple(selected),
    )
