"""Regression test for a real bug: NaN isn't valid JSON.

pandas' to_dict() leaves missing bookmaker odds as float('nan'); a naive
json.dumps() would emit a literal `NaN` token that browsers' JSON.parse
rejects outright. frontend/prepare_data.py::build_upcoming must convert
those to None (-> JSON null) instead.
"""

import json

import pandas as pd

from frontend.prepare_data import build_predictions, build_table, build_upcoming


def test_build_upcoming_converts_nan_odds_to_json_null(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {
                "date": "2026-09-04",
                "home": "Ipswich",
                "away": "Liverpool",
                "insufficient_history": False,
                "logreg_H": 0.35,
                "logreg_D": 0.21,
                "logreg_A": 0.44,
                "rf_H": 0.41,
                "rf_D": 0.21,
                "rf_A": 0.38,
                "xgb_H": 0.57,
                "xgb_D": 0.20,
                "xgb_A": 0.23,
                "odds_available": False,
                "book_H": float("nan"),
                "book_D": float("nan"),
                "book_A": float("nan"),
            }
        ]
    )
    # build_upcoming() reads from PROJECT_ROOT / "data/processed/reports/...",
    # so point it at a fake project root with that exact layout.
    import frontend.prepare_data as prepare_data

    reports_dir = tmp_path / "data" / "processed" / "reports"
    reports_dir.mkdir(parents=True)
    df.to_csv(reports_dir / "upcoming_predictions.csv", index=False)
    monkeypatch.setattr(prepare_data, "PROJECT_ROOT", tmp_path)

    records = build_upcoming()
    assert records[0]["book_H"] is None

    serialized = json.dumps(records)
    assert "NaN" not in serialized
    # round-trip through strict JSON (no NaN/Infinity extensions allowed)
    reparsed = json.loads(serialized, parse_constant=_reject_constants)
    assert reparsed[0]["book_H"] is None


def _reject_constants(value):
    raise ValueError(f"non-standard JSON constant encountered: {value}")


def test_build_table_computes_standings_only_from_the_latest_season(tmp_path, monkeypatch):
    import frontend.prepare_data as prepare_data

    matches = pd.DataFrame(
        [
            # Older season - must be excluded entirely from the table.
            {"season_start_year": 2024, "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 5, "FTAG": 0},
            # Current season (2025): Arsenal 2 games (W, D), Chelsea 2 games (L, D).
            {"season_start_year": 2025, "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 0},
            {"season_start_year": 2025, "HomeTeam": "Chelsea", "AwayTeam": "Arsenal", "FTHG": 1, "FTAG": 1},
        ]
    )
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    matches.to_csv(processed_dir / "matches_validated.csv", index=False)
    monkeypatch.setattr(prepare_data, "PROJECT_ROOT", tmp_path)

    table = build_table()

    assert len(table) == 2
    arsenal = next(r for r in table if r["team"] == "Arsenal")
    chelsea = next(r for r in table if r["team"] == "Chelsea")

    assert arsenal["played"] == 2
    assert arsenal["won"] == 1 and arsenal["drawn"] == 1 and arsenal["lost"] == 0
    assert arsenal["points"] == 4
    assert arsenal["gf"] == 3 and arsenal["ga"] == 1 and arsenal["gd"] == 2

    assert chelsea["played"] == 2
    assert chelsea["points"] == 1

    # Arsenal finished above Chelsea (more points) -> position 1.
    assert arsenal["position"] == 1
    assert chelsea["position"] == 2


def test_build_predictions_keeps_cold_start_matches_with_a_null_evaluated_flag(
    tmp_path, monkeypatch
):
    """Regression test for a real gap a user found: a brand-new team's first
    fixture (has_min_history == False) never gets a walk-forward OOF
    prediction and used to vanish from predictions.json entirely (inner
    join). It should still show up - with null model picks and
    evaluated=False - not disappear as if the match never happened.
    """
    import frontend.prepare_data as prepare_data

    base_cols = {
        "B365H": 2.0, "B365D": 3.3, "B365A": 3.6,
        "BWH": 2.0, "BWD": 3.3, "BWA": 3.6,
        "HS": 10, "AS": 10, "HST": 5, "AST": 5,
        "HC": 5, "AC": 5, "HY": 1, "AY": 1, "HR": 0, "AR": 0,
        "HxG": "", "AxG": "",
    }
    rows = []
    # A vs B alternate for 7 matches - by the 6th (index 5) both have 5 prior
    # matches each (MIN_HISTORY), same pattern as test_features.py.
    for i in range(7):
        home, away = ("A", "B") if i % 2 == 0 else ("B", "A")
        rows.append(
            {
                "Date": f"2015-08-{i + 1:02d}",
                "season_start_year": 2015,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                **base_cols,
            }
        )
    # Brand-new team C's very first match on record - 0 prior matches, so
    # has_min_history is False for this row regardless of A's own history.
    rows.append(
        {
            "Date": "2015-09-10",
            "season_start_year": 2015,
            "HomeTeam": "A",
            "AwayTeam": "C",
            "FTHG": 3,
            "FTAG": 0,
            "FTR": "H",
            **base_cols,
        }
    )
    matches = pd.DataFrame(rows)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    matches.to_csv(processed_dir / "matches_validated.csv", index=False)

    fold_results_dir = processed_dir / "fold_results"
    fold_results_dir.mkdir()
    # Only indices 5 and 6 (the two rows with has_min_history == True) get
    # an OOF row - index 7 (A vs C) is deliberately absent, exactly like the
    # real walk-forward pipeline would leave it out.
    oof = pd.DataFrame(
        {"proba_H": [0.6, 0.6], "proba_D": [0.2, 0.2], "proba_A": [0.2, 0.2], "test_season": [2015, 2015]},
        index=[5, 6],
    )
    for name in ["bookmaker_baseline", "logistic_regression", "random_forest", "xgboost", "ensemble"]:
        oof.to_csv(fold_results_dir / f"{name}_oof_predictions.csv")

    monkeypatch.setattr(prepare_data, "PROJECT_ROOT", tmp_path)

    records = build_predictions()
    by_key = {(r["home"], r["away"], r["date"]): r for r in records}

    cold_start = by_key[("A", "C", "2015-09-10")]
    assert cold_start["evaluated"] is False
    assert cold_start["rf_H"] is None
    assert cold_start["actual"] == "H"  # the real result is still shown

    evaluated_match = by_key[("B", "A", "2015-08-06")]  # index 5 (i=5 is odd -> home=B, away=A)
    assert evaluated_match["evaluated"] is True
    assert evaluated_match["rf_H"] == 0.6
