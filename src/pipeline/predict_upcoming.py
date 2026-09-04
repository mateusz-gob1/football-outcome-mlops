"""Generate predictions for the next Premier League gameweek from all trained models.

Fits each ML model (logistic regression, random forest, XGBoost, and their
average ensemble) on the *entire* available match history via
src.models.registry.fit_full_model -
there's no held-out test season for a fixture that hasn't been played yet.
Features are built the same leak-free way as every other prediction path in
this project (src.features.query_features.build_query_features). The
bookmaker's own pick is attached for display only (never fed into any of our
models - see vault/Architecture-Decisions.md ADR-022) when football-data.co.uk
has published pre-match odds for that fixture yet; otherwise it's left
unavailable rather than guessed.

Output: data/processed/reports/upcoming_predictions.csv, one row per fixture
in the next unplayed gameweek - joined by frontend/prepare_data.py into the
static demo's "Next Matchday" export.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data.fixtures_odds import attach_odds, fetch_fixtures_odds
from src.data.fixtures_openfootball import next_gameweek_fixtures
from src.features.query_features import build_query_features
from src.models.registry import fit_full_model
from src.models.train import (
    BOOKMAKER_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    prepare_dataset,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "reports"
ML_MODEL_PREFIXES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "xgboost": "xgb",
    "ensemble": "ens",
}
CLASSES = ["A", "D", "H"]


def predict_upcoming_gameweek(
    fixtures_fetcher=next_gameweek_fixtures,
    odds_fetcher=fetch_fixtures_odds,
) -> pd.DataFrame:
    """Build the upcoming-gameweek predictions table.

    `fixtures_fetcher`/`odds_fetcher` are injectable for testing without
    hitting the network - default to the real openfootball/football-data.co.uk
    fetchers.
    """
    historical = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
    known_teams = set(historical["HomeTeam"]) | set(historical["AwayTeam"])

    fixtures = fixtures_fetcher(known_teams)
    if not fixtures:
        logger.warning("No upcoming gameweek fixtures found - nothing to predict")
        return pd.DataFrame()

    fixtures_df = pd.DataFrame(fixtures)
    fixtures_df = attach_odds(fixtures_df, odds_fetcher())

    query_features = build_query_features(historical, fixtures_df)
    training_data = prepare_dataset()

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(fixtures_df["Date"]).dt.strftime("%Y-%m-%d"),
            "home": fixtures_df["HomeTeam"],
            "away": fixtures_df["AwayTeam"],
            "matchday": fixtures_df["Matchday"],
            "insufficient_history": ~query_features["has_min_history"].to_numpy(),
        }
    )

    for model_name, prefix in ML_MODEL_PREFIXES.items():
        model = fit_full_model(model_name, None, training_data)
        proba = model.predict_proba(query_features[FEATURE_COLUMNS])
        class_order = list(model.classes_)
        for cls in CLASSES:
            result[f"{prefix}_{cls}"] = proba[:, class_order.index(cls)]

    home_col, draw_col, away_col = BOOKMAKER_FEATURE_COLUMNS
    has_odds = query_features[BOOKMAKER_FEATURE_COLUMNS].notna().all(axis=1)
    result["odds_available"] = has_odds.to_numpy()
    result["book_H"] = query_features[home_col].where(has_odds).to_numpy()
    result["book_D"] = query_features[draw_col].where(has_odds).to_numpy()
    result["book_A"] = query_features[away_col].where(has_odds).to_numpy()

    return result.reset_index(drop=True)


def run() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = predict_upcoming_gameweek()
    path = REPORTS_DIR / "upcoming_predictions.csv"
    predictions.to_csv(path, index=False)
    logger.info("Saved %s (%d fixtures)", path, len(predictions))
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run()
