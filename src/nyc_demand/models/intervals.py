from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class SymmetricConformalInterval:
    coverage: float
    radius: float

    def bounds(self, predictions: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(predictions, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Predictions must be finite")
        lower = np.maximum(values - self.radius, 0.0)
        upper = values + self.radius
        return lower, upper


def fit_conformal_interval(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    coverage: float = 0.9,
) -> SymmetricConformalInterval:
    """Calibrate a finite-sample split-conformal interval from absolute residuals."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Calibration arrays must be non-empty and have equal shapes")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Calibration arrays must be finite")

    residuals = np.abs(actual - predicted)
    n = residuals.size
    quantile_level = min(np.ceil((n + 1) * coverage) / n, 1.0)
    radius = float(np.quantile(residuals, quantile_level, method="higher"))
    return SymmetricConformalInterval(coverage=coverage, radius=radius)
