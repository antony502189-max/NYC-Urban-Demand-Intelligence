from datetime import datetime, timedelta

import polars as pl

from nyc_demand.models.lightgbm_model import train_lightgbm


def _training_frame() -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    rows = 96
    hours = list(range(rows))
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in hours],
            "zone_id": [161 if i % 2 == 0 else 162 for i in hours],
            "hour": [i % 24 for i in hours],
            "day_of_week": [(i // 24) % 7 for i in hours],
            "demand_lag_1h": [20.0 + (i % 7) for i in hours],
            "demand_lag_24h": [18.0 + (i % 5) for i in hours],
            "demand": [22 + (i % 8) for i in hours],
        }
    )


def test_lightgbm_training_produces_non_negative_predictions() -> None:
    frame = _training_frame()
    model = train_lightgbm(
        frame,
        params={
            "objective": "poisson",
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "random_state": 42,
        },
    )

    predictions = model.predict(frame)

    assert len(predictions) == frame.height
    assert (predictions >= 0).all()
    assert set(model.feature_importance()) == {
        "zone_id",
        "hour",
        "day_of_week",
        "demand_lag_1h",
        "demand_lag_24h",
    }
