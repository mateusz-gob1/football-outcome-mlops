"""Pydantic request/response models for the prediction API."""

from datetime import date

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    match_date: date
    b365h: float = Field(..., gt=1.0, description="Bet365 home win odds")
    b365d: float = Field(..., gt=1.0, description="Bet365 draw odds")
    b365a: float = Field(..., gt=1.0, description="Bet365 away win odds")

    @model_validator(mode="after")
    def teams_must_differ(self) -> "PredictRequest":
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must be different")
        return self


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    match_date: date
    prob_home: float
    prob_draw: float
    prob_away: float
    model_version: str
    insufficient_history: bool
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
