"""Build model-ready features for not-yet-played fixtures.

Shared by src/serving/api.py (one fixture per request) and
src/pipeline/predict_upcoming.py (a whole gameweek at once): both need
features for a fixture that hasn't been played yet, computed the exact same
leak-free way training data is (src.features.build_features), by appending a
synthetic row with a placeholder result and recomputing - rather than
re-implementing feature logic for live/batch use. This is what avoids
train/serve skew.

Each fixture is featurized against *only* real historical data, one at a
time - never combined with any other not-yet-played fixture, even when
predicting a whole gameweek in one call. Table position and rolling form
rank/aggregate across all teams, so a placeholder result for one Friday
fixture would otherwise leak into a Sunday fixture's features in the same
gameweek (inflating one team's points/table position before that Friday
match has actually been played). Looping per-fixture costs nothing
meaningful at gameweek scale (~10 fixtures) and removes that leak entirely.
"""

import pandas as pd

from src.data.ingest import season_start_year_for_date
from src.features.build_features import build_features
from src.models.train import impute_missing_features

ODDS_COLUMNS = ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA"]


def build_query_features(historical: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Compute features for one or more not-yet-played fixtures.

    `fixtures` needs columns Date, HomeTeam, AwayTeam, and optionally (NaN
    where unknown) B365H/B365D/B365A/BWH/BWD/BWA. Returns one feature row per
    input fixture, in the same order, with missing table-position/h2h values
    already imputed - callers still pick whichever feature-column list
    applies to the model they're using (FEATURE_COLUMNS or
    BOOKMAKER_FEATURE_COLUMNS from src.models.train).
    """
    historical_flagged = historical.assign(_is_query=False)

    query_rows = []
    for _, fixture in fixtures.iterrows():
        match_date = pd.Timestamp(fixture["Date"])
        row = {
            "Date": match_date,
            "season_start_year": season_start_year_for_date(match_date.date()),
            "HomeTeam": fixture["HomeTeam"],
            "AwayTeam": fixture["AwayTeam"],
            "FTHG": 0,
            "FTAG": 0,
            "FTR": "H",
            "_is_query": True,
        }
        for col in ODDS_COLUMNS:
            value = fixture[col] if col in fixture.index else float("nan")
            row[col] = value if pd.notna(value) else float("nan")

        combined = pd.concat(
            [historical_flagged, pd.DataFrame([row])], ignore_index=True
        )
        features = build_features(combined)
        query_rows.append(features[features["_is_query"]])

    if not query_rows:
        return pd.DataFrame()

    result = pd.concat(query_rows, ignore_index=True)
    return impute_missing_features(result)
