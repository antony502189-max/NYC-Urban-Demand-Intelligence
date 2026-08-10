from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _as_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if actual.size == 0:
        raise ValueError("metrics require at least one observation")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("metrics do not accept NaN or infinite values")
    return actual, predicted


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    actual, predicted = _as_arrays(y_true, y_pred)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return 0.0 if np.allclose(actual, predicted) else float("inf")
    return float(np.abs(actual - predicted).sum() / denominator)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    actual, predicted = _as_arrays(y_true, y_pred)
    denominator = np.abs(actual) + np.abs(predicted)
    numerator = 2.0 * np.abs(actual - predicted)
    terms = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    return float(np.mean(terms))


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
