"""Tests for the FastAPI serving layer.

These exercise the real loaded model and historical data (not mocked), so they
require the Phase 1 pipeline to have already been run once:
ingest -> validate -> train -> registry (see src/models/registry.py), so that
a "production"-aliased model exists in the local MLflow registry.
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app

VALID_REQUEST = {
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2026-08-15",
    "b365h": 2.1,
    "b365d": 3.4,
    "b365a": 3.2,
    "bwh": 2.05,
    "bwd": 3.3,
    "bwa": 3.3,
}


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_name"] == "football_outcome_predictor"
        assert "model_version" in body


def test_predict_valid_request_returns_probabilities_summing_to_one():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200
        body = response.json()
        total = body["prob_home"] + body["prob_draw"] + body["prob_away"]
        assert total == pytest.approx(1.0, abs=1e-6)
        assert body["home_team"] == "Arsenal"
        assert body["away_team"] == "Chelsea"
        assert body["insufficient_history"] is False


def test_predict_unknown_team_returns_422_not_500():
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={**VALID_REQUEST, "home_team": "Totally Made Up FC"}
        )
        assert response.status_code == 422
        assert "Unknown team" in response.json()["detail"]


def test_predict_zero_history_date_returns_422():
    with TestClient(app) as client:
        # Opening day of the entire dataset - no team has any prior match yet.
        response = client.post(
            "/predict", json={**VALID_REQUEST, "match_date": "2010-08-14"}
        )
        assert response.status_code == 422


def test_predict_bad_date_format_returns_422():
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={**VALID_REQUEST, "match_date": "not-a-date"}
        )
        assert response.status_code == 422


def test_predict_bad_odds_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={**VALID_REQUEST, "b365h": 0.5})
        assert response.status_code == 422


def test_predict_same_team_both_sides_returns_422():
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={**VALID_REQUEST, "away_team": "Arsenal"}
        )
        assert response.status_code == 422
