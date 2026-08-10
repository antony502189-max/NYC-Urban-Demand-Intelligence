from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl


NON_FEATURE_COLUMNS = frozenset({"timestamp", "forecast_timestamp", "demand", "target_demand"})


@dataclass(frozen=True)
class FeatureMatrix:
    features: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class TrainingMatrix(FeatureMatrix):
    target: np.ndarray


def infer_feature_columns(
    frame: pl.DataFrame,
    *,
    excluded_columns: Iterable[str] = (),
) -> tuple[str, ...]:
    """Infer numeric model features while excluding identifiers and target columns."""
    excluded = NON_FEATURE_COLUMNS.union(excluded_columns)
    columns: list[str] = []
    for name, dtype in frame.schema.items():
        if name in excluded:
            continue
        if dtype.is_numeric():
            columns.append(name)

    if not columns:
        raise ValueError("No numeric feature columns are available")
    return tuple(columns)


def build_feature_matrix(
    frame: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
) -> FeatureMatrix:
    """Convert feature rows into a finite NumPy matrix without changing row order."""
    if frame.is_empty():
        raise ValueError("Feature frame must not be empty")

    selected = feature_columns or infer_feature_columns(frame)
    missing = set(selected).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")

    if frame.select(selected).null_count().row(0) != tuple(0 for _ in selected):
        raise ValueError("Feature rows contain null values")

    features = frame.select(selected).to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values")

    return FeatureMatrix(features=features, feature_names=tuple(selected))


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

    selected = feature_columns or infer_feature_columns(
        frame,
        excluded_columns=(target_column,),
    )
    missing = set(selected).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")

    clean = frame.drop_nulls([*selected, target_column])
    if clean.is_empty():
        raise ValueError("No complete rows remain after dropping null model inputs")

    matrix = build_feature_matrix(clean, feature_columns=tuple(selected))
    target = clean[target_column].to_numpy().astype(np.float64, copy=False)

    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or infinite values")
    if np.any(target < 0):
        raise ValueError("Demand target must be non-negative")

    return TrainingMatrix(
        features=matrix.features,
        target=target,
        feature_names=matrix.feature_names,
    )
