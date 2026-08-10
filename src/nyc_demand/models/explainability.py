from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import shap

from nyc_demand.models.dataset import build_feature_matrix


def _deterministic_sample(features: np.ndarray, max_rows: int) -> np.ndarray:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if features.shape[0] <= max_rows:
        return features

    indices = np.linspace(0, features.shape[0] - 1, num=max_rows, dtype=int)
    return features[indices]


def global_shap_importance(
    model: Any,
    frame: pl.DataFrame,
    *,
    feature_names: tuple[str, ...],
    max_rows: int = 5000,
) -> pl.DataFrame:
    """Return deterministic mean absolute SHAP importance for a tree model."""
    matrix = build_feature_matrix(frame, feature_columns=feature_names)
    sample = _deterministic_sample(matrix.features, max_rows)

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(sample), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("Unexpected SHAP output shape for regression model")

    mean_absolute = np.mean(np.abs(values), axis=0)
    total = float(mean_absolute.sum())
    share = mean_absolute / total if total > 0 else np.zeros_like(mean_absolute)

    return pl.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": mean_absolute,
            "importance_share": share,
        }
    ).sort("mean_abs_shap", descending=True)


def local_shap_explanation(
    model: Any,
    frame: pl.DataFrame,
    *,
    feature_names: tuple[str, ...],
    row_index: int = 0,
) -> dict[str, object]:
    """Explain one prediction as base value plus per-feature SHAP contributions."""
    matrix = build_feature_matrix(frame, feature_columns=feature_names)
    if not 0 <= row_index < matrix.features.shape[0]:
        raise IndexError("row_index is outside the feature matrix")

    row = matrix.features[row_index : row_index + 1]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(row), dtype=float)
    if values.shape != (1, len(feature_names)):
        raise ValueError("Unexpected SHAP output shape for local explanation")

    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    base_value = float(expected[0])
    contributions = {
        name: float(value)
        for name, value in zip(feature_names, values[0], strict=True)
    }
    return {
        "base_value": base_value,
        "contributions": contributions,
        "explained_value": base_value + float(values[0].sum()),
    }
