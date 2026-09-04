"""Regression test for a real leak risk caught during design.

Table position and rolling form rank/aggregate across *all* teams. If two
not-yet-played fixtures from the same gameweek were featurized together
(one combined DataFrame with both placeholder rows), an earlier-dated
fixture's fake result could leak into a later-dated fixture's features -
inflating a team's table position before it has actually played. This test
proves build_query_features avoids that by construction: predicting several
fixtures together must give byte-for-byte the same result as predicting
each one alone.
"""

import pandas as pd

from src.features.query_features import build_query_features
from src.models.train import FEATURE_COLUMNS


def _historical_matches() -> pd.DataFrame:
    rows = []
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    pairings = [
        ("Alpha", "Beta", 2, 0),
        ("Gamma", "Delta", 1, 1),
        ("Beta", "Gamma", 0, 3),
        ("Delta", "Alpha", 1, 2),
        ("Alpha", "Gamma", 2, 2),
        ("Beta", "Delta", 1, 0),
        ("Delta", "Beta", 3, 1),
        ("Gamma", "Alpha", 0, 1),
    ]
    for i, (home, away, hg, ag) in enumerate(pairings):
        rows.append(
            {
                "Date": pd.Timestamp("2026-08-01") + pd.Timedelta(days=7 * i),
                "season_start_year": 2026,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
                "B365H": 2.0,
                "B365D": 3.3,
                "B365A": 3.6,
                "BWH": 2.0,
                "BWD": 3.3,
                "BWA": 3.6,
                "HS": 10,
                "AS": 10,
                "HST": 5,
                "AST": 5,
                "HC": 5,
                "AC": 5,
                "HY": 1,
                "AY": 1,
                "HR": 0,
                "AR": 0,
            }
        )
    return pd.DataFrame(rows)


def test_batch_prediction_matches_predicting_each_fixture_alone():
    historical = _historical_matches()
    last_date = historical["Date"].max()

    # Two fixtures in the "same gameweek" (a few days apart), different
    # team pairs - the exact scenario where a naive combined-DataFrame
    # approach would let Friday's fake result leak into Sunday's features.
    friday = pd.DataFrame(
        [{"Date": last_date + pd.Timedelta(days=5), "HomeTeam": "Alpha", "AwayTeam": "Delta"}]
    )
    sunday = pd.DataFrame(
        [{"Date": last_date + pd.Timedelta(days=7), "HomeTeam": "Beta", "AwayTeam": "Gamma"}]
    )
    both = pd.concat([friday, sunday], ignore_index=True)

    batch_result = build_query_features(historical, both)
    individually = pd.concat(
        [
            build_query_features(historical, friday),
            build_query_features(historical, sunday),
        ],
        ignore_index=True,
    )

    for col in FEATURE_COLUMNS:
        assert (
            batch_result[col].to_numpy() == individually[col].to_numpy()
        ).all(), f"feature '{col}' differs between batch and individual prediction"


def test_missing_odds_do_not_break_feature_building():
    historical = _historical_matches()
    last_date = historical["Date"].max()
    fixture = pd.DataFrame(
        [
            {
                "Date": last_date + pd.Timedelta(days=5),
                "HomeTeam": "Alpha",
                "AwayTeam": "Delta",
            }
        ]
    )
    result = build_query_features(historical, fixture)
    assert len(result) == 1
    assert result["odds_implied_home_prob"].isna().all()
