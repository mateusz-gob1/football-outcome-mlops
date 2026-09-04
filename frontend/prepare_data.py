"""One-off script: build the static JSON dataset the demo ships with.

Joins the out-of-fold predictions for all 4 models (logistic regression,
random forest, XGBoost, bookmaker baseline - all already computed by
src/models/train.py's walk-forward validation) back onto match metadata
(teams, date) by row index, plus the next gameweek's predictions from
src/pipeline/predict_upcoming.py - so the static frontend can just fetch
JSON and render, no server, no model, no feature engineering at runtime.
Run this once whenever the underlying model/data or upcoming fixtures change.
"""

import json
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "data"


# Model OOF CSVs -> the column prefix used in predictions.json/upcoming.json.
MODEL_PREFIXES = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "xgboost": "xgb",
    "ensemble": "ens",
}


def build_predictions() -> list[dict]:
    from src.features.build_features import build_features

    matches = pd.read_csv(
        PROJECT_ROOT / "data/processed/matches_validated.csv", parse_dates=["Date"]
    )
    features = build_features(matches)

    # Some matches within an otherwise-evaluated season are excluded from
    # walk-forward training/evaluation entirely (has_min_history == False -
    # one of the two teams had fewer than MIN_HISTORY prior matches, e.g. a
    # newly-promoted club's very first fixture in this dataset). Those rows
    # never got an OOF prediction from any model - not a bug, a deliberate
    # cold-start rule (see build_features.py). They're still included here
    # (left join, not inner) with null pick columns and evaluated=False,
    # rather than silently vanishing from a team's browse history - a match
    # you can see happened but the model was never asked to call.
    bookmaker_oof = pd.read_csv(
        PROJECT_ROOT
        / "data/processed/fold_results/bookmaker_baseline_oof_predictions.csv",
        index_col=0,
    )
    min_evaluated_season = int(bookmaker_oof["test_season"].min())

    # Raw match stats (not model features - see ADR-026) carried through for
    # the frontend's team-profile view: shots/corners/cards give a richer
    # "recent form" than just the scoreline; xG is display-only everywhere
    # (only available from 2026/27 onward, so it's null for older matches).
    meta = features[features["season_start_year"] >= min_evaluated_season][
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "season_start_year",
            "FTR",
            "FTHG",
            "FTAG",
            "HS",
            "AS",
            "HST",
            "AST",
            "HC",
            "AC",
            "HY",
            "AY",
            "HR",
            "AR",
            "HxG",
            "AxG",
        ]
    ].rename(
        columns={
            "HomeTeam": "home",
            "AwayTeam": "away",
            "season_start_year": "season",
            "FTR": "actual",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "HS": "home_shots",
            "AS": "away_shots",
            "HST": "home_shots_on_target",
            "AST": "away_shots_on_target",
            "HC": "home_corners",
            "AC": "away_corners",
            "HxG": "home_xg",
            "AxG": "away_xg",
        }
    )
    meta["home_cards"] = features["HY"] + features["HR"]
    meta["away_cards"] = features["AY"] + features["AR"]
    meta = meta.drop(columns=["HY", "AY", "HR", "AR"])

    combined = meta.join(
        bookmaker_oof[["proba_H", "proba_D", "proba_A"]].rename(
            columns={"proba_H": "book_H", "proba_D": "book_D", "proba_A": "book_A"}
        ),
        how="left",
    )

    for model_name, prefix in MODEL_PREFIXES.items():
        oof = pd.read_csv(
            PROJECT_ROOT
            / f"data/processed/fold_results/{model_name}_oof_predictions.csv",
            index_col=0,
        )
        combined = combined.join(
            oof[["proba_H", "proba_D", "proba_A"]].rename(
                columns={
                    "proba_H": f"{prefix}_H",
                    "proba_D": f"{prefix}_D",
                    "proba_A": f"{prefix}_A",
                }
            ),
            how="left",
        )

    combined["date"] = combined["Date"].dt.strftime("%Y-%m-%d")
    combined["evaluated"] = combined["rf_H"].notna()

    stat_cols = [
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
        "home_cards",
        "away_cards",
        "home_xg",
        "away_xg",
    ]
    model_cols = [f"{p}_{c}" for p in [*MODEL_PREFIXES.values(), "book"] for c in "HDA"]
    combined = combined[
        [
            "date",
            "home",
            "away",
            "season",
            "evaluated",
            *model_cols,
            "actual",
            *stat_cols,
        ]
    ]
    for col in model_cols:
        combined[col] = combined[col].round(4)
    for col in ("home_xg", "away_xg"):
        combined[col] = combined[col].round(2)

    records = combined.sort_values("date").to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records


def build_upcoming() -> list[dict]:
    path = PROJECT_ROOT / "data/processed/reports/upcoming_predictions.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)

    prob_cols = [f"{p}_{c}" for p in [*MODEL_PREFIXES.values(), "book"] for c in "HDA"]
    for col in prob_cols:
        if col in df.columns:
            df[col] = df[col].round(4)
    # NaN (e.g. book_H/D/A when odds aren't published yet) isn't valid JSON -
    # json.dumps would emit a literal `NaN` token that browsers' JSON.parse
    # rejects. Pandas re-coerces None back to NaN on a float column
    # (df.where(..., None) doesn't survive), so the fix has to happen on the
    # plain Python dicts after to_dict(), not on the DataFrame.
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records


def build_report(filename: str) -> list[dict]:
    df = pd.read_csv(PROJECT_ROOT / "data/processed/reports" / filename)
    return df.to_dict(orient="records")


def build_table() -> list[dict]:
    """Current Premier League table: full standings for the latest season on record.

    A plain end-of-latest-matchday standings computation (all played games
    counted), not the leak-free "as of the day before" position
    src/features/build_features.py computes for model training - that one is
    deliberately pre-match-only, which would make "today's table" look one
    matchday stale. This is a display artifact only, not a model input, so
    it's computed directly here rather than reusing that function.
    """
    matches = pd.read_csv(PROJECT_ROOT / "data/processed/matches_validated.csv")
    current_season = matches["season_start_year"].max()
    season_matches = matches[matches["season_start_year"] == current_season]

    rows = []
    for team in sorted(
        set(season_matches["HomeTeam"]) | set(season_matches["AwayTeam"])
    ):
        home = season_matches[season_matches["HomeTeam"] == team]
        away = season_matches[season_matches["AwayTeam"] == team]
        played = len(home) + len(away)
        won = int(
            (home["FTHG"] > home["FTAG"]).sum() + (away["FTAG"] > away["FTHG"]).sum()
        )
        drawn = int(
            (home["FTHG"] == home["FTAG"]).sum() + (away["FTHG"] == away["FTAG"]).sum()
        )
        lost = played - won - drawn
        gf = int(home["FTHG"].sum() + away["FTAG"].sum())
        ga = int(home["FTAG"].sum() + away["FTHG"].sum())
        rows.append(
            {
                "team": team,
                "played": played,
                "won": won,
                "drawn": drawn,
                "lost": lost,
                "gf": gf,
                "ga": ga,
                "gd": gf - ga,
                "points": won * 3 + drawn,
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values(["points", "gd", "gf"], ascending=False)
        .reset_index(drop=True)
    )
    table["position"] = table.index + 1
    return table.to_dict(orient="records")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "predictions.json": build_predictions(),
        "upcoming.json": build_upcoming(),
        "table.json": build_table(),
        "model_summary.json": build_report("model_summary.csv"),
        "bootstrap_ci.json": build_report("bootstrap_ci_vs_bookmaker.csv"),
        "calibration.json": build_report("random_forest_calibration_home.csv"),
        "feature_importance.json": build_report("random_forest_feature_importance.csv"),
    }
    for name, data in datasets.items():
        path = OUT_DIR / name
        path.write_text(json.dumps(data, indent=None))
        print(f"Saved {path} ({len(data)} records)")
