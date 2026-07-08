"""One-off script: build the static dataset the Streamlit demo ships with.

Joins the out-of-fold predictions (random_forest, bookmaker_baseline - both
already computed by src/models/train.py's walk-forward validation) back onto
match metadata (teams, date) by row index, so the demo can just load one CSV
and display it - no feature engineering or model loading needed at runtime.
Run this once whenever the underlying model/data changes; the demo itself
never re-runs it.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_demo_dataset() -> pd.DataFrame:
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
            columns={
                "proba_H": "model_prob_H",
                "proba_D": "model_prob_D",
                "proba_A": "model_prob_A",
            }
        ),
        how="inner",
    ).join(
        bookmaker_oof[["proba_H", "proba_D", "proba_A"]].rename(
            columns={
                "proba_H": "bookmaker_prob_H",
                "proba_D": "bookmaker_prob_D",
                "proba_A": "bookmaker_prob_A",
            }
        ),
        how="inner",
    )

    combined["season"] = combined["test_season"].astype(int)
    combined["actual_result"] = combined["true_label"]
    combined = combined.drop(columns=["test_season", "true_label"])
    return combined.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    dataset = build_demo_dataset()
    out_path = Path(__file__).resolve().parent / "data" / "predictions.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(dataset)} matches)")
