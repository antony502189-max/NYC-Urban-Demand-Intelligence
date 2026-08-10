from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, status

from nyc_demand.api.schemas import ForecastRequest, ForecastResponse, HealthResponse
from nyc_demand.models.persistence import StoredDemandModel, load_model_bundle


class ServingModel(Protocol):
    feature_names: tuple[str, ...]
    horizon_hours: int

    def predict(self, frame: pl.DataFrame) -> np.ndarray: ...


def create_app(model: ServingModel | None = None) -> FastAPI:
    app = FastAPI(
        title="NYC Urban Demand Intelligence",
        version="0.1.0",
        description="Inference API for hourly NYC taxi-zone demand forecasts.",
    )
    app.state.model = model

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.model is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
        )

    @app.post("/v1/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastRequest) -> ForecastResponse:
        serving_model: ServingModel | None = app.state.model
        if serving_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model bundle is loaded",
            )

        expected = set(serving_model.feature_names)
        received = set(payload.features)
        missing = expected.difference(received)
        extra = received.difference(expected)
        if missing or extra:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "missing_features": sorted(missing),
                    "unexpected_features": sorted(extra),
                },
            )

        row = pl.DataFrame(
            [{name: payload.features[name] for name in serving_model.feature_names}]
        )
        prediction = float(serving_model.predict(row)[0])
        return ForecastResponse(
            prediction=prediction,
            horizon_hours=serving_model.horizon_hours,
            model_features=list(serving_model.feature_names),
        )

    return app


def _load_from_environment() -> StoredDemandModel | None:
    model_dir = os.getenv("NYC_DEMAND_MODEL_DIR")
    if not model_dir:
        return None
    return load_model_bundle(Path(model_dir))


app = create_app(_load_from_environment())
