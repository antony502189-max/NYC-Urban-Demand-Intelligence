from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(min_length=1)


class ForecastResponse(BaseModel):
    prediction: float = Field(ge=0)
    horizon_hours: int = Field(gt=0)
    model_features: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
