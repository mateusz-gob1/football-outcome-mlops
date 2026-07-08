"""One-off script: build the static JSON dataset the demo ships with.

Joins the out-of-fold predictions (random_forest, bookmaker_baseline - both
already computed by src/models/train.py's walk-forward validation) back onto
match metadata (teams, date) by row index, so the static frontend can just
fetch JSON and render - no server, no model, no feature engineering at
runtime. Run this once whenever the underlying model/data changes.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "data"


def build_predictions() -> list[dict]:
    from src.features.build_features import build_features

    matches = pd.read_csv(
        PROJECT_ROOT / "data/processed/matches_validated.csv", parse_dates=["Date"]
    )
    features = build_features(matches)
    meta = features[["Date", "HomeTeam", "AwayTeam"]]

    model_oof = pd.read_csv(
        PROJECT_ROOT / "data/processed/fold_results/random_forest_oof_predictions.csv",
        index_col=0,
    )
    bookmaker_oof = pd.read_csv(
        PROJECT_ROOT
        / "data/processed/fold_results/bookmaker_baseline_oof_predictions.csv",
        index_col=0,
    )

    combined = meta.join(
        model_oof.rename(
            columns={"proba_H": "model_H", "proba_D": "model_D", "proba_A": "model_A"}
        ),
        how="inner",
    ).join(
        bookmaker_oof[["proba_H", "proba_D", "proba_A"]].rename(
            columns={"proba_H": "book_H", "proba_D": "book_D", "proba_A": "book_A"}
        ),
        how="inner",
    )
    combined["date"] = combined["Date"].dt.strftime("%Y-%m-%d")
    combined["season"] = combined["test_season"].astype(int)
    combined["actual"] = combined["true_label"]
    combined = combined.rename(columns={"HomeTeam": "home", "AwayTeam": "away"})
    combined = combined[
        [
            "date",
            "home",
            "away",
            "season",
            "model_H",
            "model_D",
            "model_A",
            "book_H",
            "book_D",
            "book_A",
            "actual",
        ]
    ]
    for col in ["model_H", "model_D", "model_A", "book_H", "book_D", "book_A"]:
        combined[col] = combined[col].round(4)
    return combined.sort_values("date").to_dict(orient="records")


def build_report(filename: str) -> list[dict]:
    df = pd.read_csv(PROJECT_ROOT / "data/processed/reports" / filename)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "predictions.json": build_predictions(),
        "model_summary.json": build_report("model_summary.csv"),
        "bootstrap_ci.json": build_report("bootstrap_ci_vs_bookmaker.csv"),
        "calibration.json": build_report("random_forest_calibration_home.csv"),
        "feature_importance.json": build_report("random_forest_feature_importance.csv"),
    }
    for name, data in datasets.items():
        path = OUT_DIR / name
        path.write_text(json.dumps(data, indent=None))
        print(f"Saved {path} ({len(data)} records)")
