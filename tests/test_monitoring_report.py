import polars as pl

from nyc_demand.monitoring.report import feature_drift_report, monitoring_report


def _reference() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature_a": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "actual": [10.0] * 10,
            "prediction": [9.5] * 10,
        }
    )


def test_feature_drift_report_ranks_shifted_feature() -> None:
    reference = _reference()
    current = reference.with_columns((pl.col("feature_a") + 10.0).alias("feature_a"))

    report = feature_drift_report(reference, current, feature_names=["feature_a"], bins=5)

    assert report[0]["feature"] == "feature_a"
    assert report[0]["status"] == "drift"


def test_monitoring_report_escalates_performance_regression() -> None:
    reference = _reference()
    current = reference.with_columns(
        pl.lit(5.0).alias("prediction"),
    )

    report = monitoring_report(reference, current, feature_names=["feature_a"])

    assert report["performance_status"] == "critical"
    assert report["overall_status"] == "critical"
