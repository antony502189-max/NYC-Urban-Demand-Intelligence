from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import Booster

from nyc_demand.models.dataset import build_feature_matrix
from nyc_demand.models.lightgbm_model import FittedDemandModel


@dataclass(frozen=True)
class StoredDemandModel:
    booster: Booster
    feature_names: tuple[str, ...]
    horizon_hours: int

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = build_feature_matrix(frame, feature_columns=self.feature_names)
        prediction = np.asarray(self.booster.predict(matrix.features), dtype=float)
        return np.maximum(prediction, 0.0)


def save_model_bundle(
    fitted: FittedDemandModel,
    directory: str | Path,
    *,
    horizon_hours: int,
) -> Path:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if not hasattr(fitted.model, "booster_"):
        raise ValueError("LightGBM model must be fitted before it can be saved")

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.txt"
    metadata_path = target / "metadata.json"

    fitted.model.booster_.save_model(str(model_path))
    metadata_path.write_text(
        json.dumps(
            {
                "horizon_hours": horizon_hours,
                "feature_names": list(fitted.feature_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def load_model_bundle(directory: str | Path) -> StoredDemandModel:
    source = Path(directory)
    metadata_path = source / "metadata.json"
    model_path = source / "model.txt"
    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Incomplete model bundle: {source}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names")
    horizon_hours = metadata.get("horizon_hours")
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError("Model metadata contains invalid feature_names")
    if not isinstance(horizon_hours, int) or horizon_hours <= 0:
        raise ValueError("Model metadata contains invalid horizon_hours")

    return StoredDemandModel(
        booster=Booster(model_file=str(model_path)),
        feature_names=tuple(feature_names),
        horizon_hours=horizon_hours,
    )
