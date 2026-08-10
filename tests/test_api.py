from dataclasses import dataclass

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from nyc_demand.api.app import create_app


@dataclass
class FakeModel:
    feature_names: tuple[str, ...] = ("zone_id", "hour", "demand_lag_1h")
    horizon_hours: int = 6

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return np.asarray([float(frame["demand_lag_1h"][0]) + 2.0])


def test_forecast_endpoint_returns_model_prediction() -> None:
    client = TestClient(create_app(FakeModel()))

    response = client.post(
        "/v1/forecast",
        json={
            "features": {
                "zone_id": 161,
                "hour": 18,
                "demand_lag_1h": 42.0,
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 44.0
    assert response.json()["horizon_hours"] == 6


def test_forecast_endpoint_rejects_feature_contract_mismatch() -> None:
    client = TestClient(create_app(FakeModel()))

    response = client.post(
        "/v1/forecast",
        json={"features": {"zone_id": 161, "hour": 18}},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"] == ["demand_lag_1h"]


def test_health_is_degraded_without_model() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False}
