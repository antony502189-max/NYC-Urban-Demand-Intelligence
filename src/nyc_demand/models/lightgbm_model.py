from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor

from nyc_demand.config import load_yaml
from nyc_demand.models.dataset import build_feature_matrix, build_training_matrix


DEFAULT_MODEL_CONFIG = "configs/model.yaml"


@dataclass
class FittedDemandModel:
    model: LGBMRegressor
    feature_names: tuple[str, ...]

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        """Generate non-negative demand predictions without altering row order."""
        matrix = build_feature_matrix(frame, feature_columns=self.feature_names)
        prediction = np.asarray(self.model.predict(matrix.features), dtype=float)
        return np.maximum(prediction, 0.0)

    def feature_importance(self) -> dict[str, float]:
        importance = np.asarray(self.model.feature_importances_, dtype=float)
        return dict(zip(self.feature_names, importance, strict=True))


def load_lightgbm_params(config_path: str = DEFAULT_MODEL_CONFIG) -> dict[str, Any]:
    config = load_yaml(config_path)
    params = config.get("lightgbm")
    if not isinstance(params, dict):
        raise ValueError("Model configuration must contain a lightgbm mapping")
    return dict(params)


def train_lightgbm(
    train: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    params: dict[str, Any] | None = None,
) -> FittedDemandModel:
    """Train a LightGBM Poisson demand model from a leakage-safe feature frame."""
    matrix = build_training_matrix(train, feature_columns=feature_columns)
    model_params = load_lightgbm_params() if params is None else dict(params)
    model_params.setdefault("verbosity", -1)

    model = LGBMRegressor(**model_params)
    model.fit(matrix.features, matrix.target)
    return FittedDemandModel(model=model, feature_names=matrix.feature_names)
