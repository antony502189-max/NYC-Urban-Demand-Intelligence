from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from nyc_demand.monitoring.drift import classify_psi, population_stability_index
from nyc_demand.monitoring.performance import (
    classify_performance_regression,
    performance_snapshot,
)


def feature_drift_report(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    feature_names: Sequence[str],
    bins: int = 10,
) -> list[dict[str, object]]:
    if not feature_names:
        raise ValueError("feature_names must not be empty")

    rows: list[dict[str, object]] = []
    for feature in feature_names:
        if feature not in reference.columns or feature not in current.columns:
            raise ValueError(f"Missing monitored feature: {feature}")

        reference_values = reference[feature].drop_nulls().to_numpy()
        current_values = current[feature].drop_nulls().to_numpy()
        psi = population_stability_index(reference_values, current_values, bins=bins)
        rows.append({"feature": feature, "psi": psi, "status": classify_psi(psi)})

    return sorted(rows, key=lambda item: float(item["psi"]), reverse=True)


def monitoring_report(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    feature_names: Sequence[str],
    actual_column: str = "actual",
    prediction_column: str = "prediction",
) -> dict[str, object]:
    reference_performance = performance_snapshot(
        reference,
        actual_column=actual_column,
        prediction_column=prediction_column,
    )
    current_performance = performance_snapshot(
        current,
        actual_column=actual_column,
        prediction_column=prediction_column,
    )
    performance_status = classify_performance_regression(
        reference_performance.wape,
        current_performance.wape,
    )
    drift = feature_drift_report(
        reference,
        current,
        feature_names=feature_names,
    )

    drift_statuses = {str(item["status"]) for item in drift}
    if performance_status == "critical" or "drift" in drift_statuses:
        overall_status = "critical"
    elif performance_status == "watch" or "watch" in drift_statuses:
        overall_status = "watch"
    else:
        overall_status = "stable"

    return {
        "overall_status": overall_status,
        "performance_status": performance_status,
        "reference_performance": reference_performance.to_dict(),
        "current_performance": current_performance.to_dict(),
        "feature_drift": drift,
    }
