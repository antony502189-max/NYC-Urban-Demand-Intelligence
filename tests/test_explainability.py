import numpy as np
import polars as pl

from nyc_demand.models.explainability import global_shap_importance, local_shap_explanation
from nyc_demand.models.lightgbm_model import train_lightgbm


def _training_frame() -> pl.DataFrame:
    x1 = np.arange(60, dtype=float)
    x2 = np.tile(np.arange(6, dtype=float), 10)
    demand = 5.0 + 0.8 * x1 + 2.5 * x2
    return pl.DataFrame({"x1": x1, "x2": x2, "demand": demand})


def test_global_shap_importance_is_normalized_and_sorted() -> None:
    frame = _training_frame()
    fitted = train_lightgbm(
        frame,
        params={
            "objective": "regression",
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "random_state": 42,
        },
    )

    result = global_shap_importance(
        fitted.model,
        frame,
        feature_names=fitted.feature_names,
        max_rows=30,
    )

    assert result.height == 2
    assert np.isclose(result["importance_share"].sum(), 1.0)
    assert result["mean_abs_shap"].to_list() == sorted(
        result["mean_abs_shap"].to_list(), reverse=True
    )


def test_local_shap_explanation_returns_one_contribution_per_feature() -> None:
    frame = _training_frame()
    fitted = train_lightgbm(
        frame,
        params={
            "objective": "regression",
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "random_state": 42,
        },
    )

    result = local_shap_explanation(
        fitted.model,
        frame,
        feature_names=fitted.feature_names,
        row_index=5,
    )

    assert set(result["contributions"]) == set(fitted.feature_names)
    assert np.isfinite(result["base_value"])
    assert np.isfinite(result["explained_value"])
