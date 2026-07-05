"""FastAPI serving layer for the football match outcome predictor.

Reuses src/features/build_features.py directly at request time (append the
requested fixture to the historical dataset, recompute features, read back
the one new row) instead of re-implementing feature logic for serving. This
avoids train/serve skew: the exact same leak-free pipeline that produced the
training data also produces the live prediction's features.
"""

import logging
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient

from src.data.ingest import season_start_year_for_date
from src.features.build_features import build_features
from src.models.registry import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME
from src.models.train import FEATURE_COLUMNS, PROCESSED_DATA_PATH, impute_missing_features
from src.serving.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}")
    historical = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])

    _state["model"] = model
    _state["model_version"] = str(version.version)
    _state["historical"] = historical
    _state["known_teams"] = set(historical["HomeTeam"]) | set(historical["AwayTeam"])
    logger.info("Loaded %s v%s (alias=%s)", REGISTERED_MODEL_NAME, version.version, PRODUCTION_ALIAS)

    yield
    _state.clear()


app = FastAPI(title="Football Match Outcome Predictor", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=REGISTERED_MODEL_NAME,
        model_version=_state["model_version"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    logger.info("predict request: %s vs %s on %s", request.home_team, request.away_team, request.match_date)

    for team in (request.home_team, request.away_team):
        if team not in _state["known_teams"]:
            logger.warning("predict rejected: unknown team '%s'", team)
            raise HTTPException(status_code=422, detail=f"Unknown team: '{team}' has no historical data")

    new_row = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(request.match_date),
                "season_start_year": season_start_year_for_date(request.match_date),
                "HomeTeam": request.home_team,
                "AwayTeam": request.away_team,
                "FTHG": 0,
                "FTAG": 0,
                "FTR": "H",
                "B365H": request.b365h,
                "B365D": request.b365d,
                "B365A": request.b365a,
                "_is_query": True,
            }
        ]
    )
    historical = _state["historical"].assign(_is_query=False)
    combined = pd.concat([historical, new_row], ignore_index=True)

    features = build_features(combined)
    query_row = features[features["_is_query"]]

    no_history = (
        query_row[["home_matches_played_prior", "away_matches_played_prior"]].eq(0).any(axis=None)
    )
    if no_history:
        logger.warning(
            "predict rejected: zero prior matches for %s or %s before %s",
            request.home_team, request.away_team, request.match_date,
        )
        raise HTTPException(
            status_code=422,
            detail="At least one team has zero matches before this date in our historical data - cannot compute features",
        )

    query_row = impute_missing_features(query_row)
    X = query_row[FEATURE_COLUMNS]

    model = _state["model"]
    proba = model.predict_proba(X)[0]
    prob_by_class = dict(zip(model.classes_, proba))

    insufficient_history = not bool(query_row["has_min_history"].iloc[0])
    warnings: list[str] = []
    if insufficient_history:
        warnings.append(
            "One or both teams have fewer than the minimum history threshold used in "
            "evaluation - this prediction is lower-confidence than the validated numbers "
            "in vault/Evaluation-Log.md."
        )

    logger.info(
        "predict result: %s vs %s -> H=%.3f D=%.3f A=%.3f",
        request.home_team, request.away_team, prob_by_class["H"], prob_by_class["D"], prob_by_class["A"],
    )

    return PredictResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        match_date=request.match_date,
        prob_home=float(prob_by_class["H"]),
        prob_draw=float(prob_by_class["D"]),
        prob_away=float(prob_by_class["A"]),
        model_version=_state["model_version"],
        insufficient_history=insufficient_history,
        warnings=warnings,
    )
