"""Train baseline and candidate models with walk-forward validation + MLflow logging."""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.build_features import build_features
from src.models.validation import walk_forward_validate

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_validated.csv"
FOLD_RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "fold_results"

FEATURE_COLUMNS = [
    "home_form_pts_5",
    "home_form_pts_10",
    "home_goals_scored_5",
    "home_goals_conceded_5",
    "home_streak_before",
    "home_rest_days",
    "home_table_position",
    "away_form_pts_5",
    "away_form_pts_10",
    "away_goals_scored_5",
    "away_goals_conceded_5",
    "away_streak_before",
    "away_rest_days",
    "away_table_position",
    "h2h_home_pts",
    "h2h_away_pts",
    "h2h_matches_count",
    "odds_implied_home_prob",
    "odds_implied_draw_prob",
    "odds_implied_away_prob",
]
TARGET_COLUMN = "FTR"
MIN_TRAIN_SEASONS = 5


class BookmakerBaseline:
    """No-fit baseline: the bookmaker's own overround-normalized implied probabilities."""

    classes_ = np.array(["A", "D", "H"])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return X[
            [
                "odds_implied_away_prob",
                "odds_implied_draw_prob",
                "odds_implied_home_prob",
            ]
        ].to_numpy()


class XGBoostWrapper:
    """Wraps XGBClassifier to accept string labels and expose sklearn-style .classes_.

    XGBoost's sklearn API requires integer class labels; A/D/H are encoded 0/1/2
    (alphabetical, matching sklearn's own default label ordering) so predict_proba
    columns line up with `classes_` the same way they do for the other models.
    """

    classes_ = np.array(["A", "D", "H"])
    _label_to_idx = {"A": 0, "D": 1, "H": 2}

    def __init__(self, params: dict):
        self.model = XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            objective="multi:softprob",
            num_class=3,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y.map(self._label_to_idx))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def make_logreg(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=params.get("C", 1.0))),
        ]
    )


def make_random_forest(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth"),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        random_state=42,
    )


MODEL_SPECS = {
    "bookmaker_baseline": {
        "factory": lambda params: BookmakerBaseline(),
        "param_grid": None,
    },
    "logistic_regression": {
        "factory": make_logreg,
        "param_grid": [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
    },
    "random_forest": {
        "factory": make_random_forest,
        "param_grid": [
            {"n_estimators": n, "max_depth": d}
            for n in [200, 400]
            for d in [6, 12, None]
        ],
    },
    "xgboost": {
        "factory": lambda params: XGBoostWrapper(params),
        "param_grid": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [200, 400]
            for d in [3, 6]
            for lr in [0.05, 0.1]
        ],
    },
}


def impute_missing_features(features: pd.DataFrame) -> pd.DataFrame:
    """Impute season-opener table position and never-met h2h with neutral defaults.

    Shared between training (prepare_dataset) and live serving (src/serving/api.py),
    so the same rule is applied consistently and served predictions never silently
    diverge from what the model was trained on (train/serve skew).
    """
    features = features.copy()
    features["home_table_position"] = features["home_table_position"].fillna(10.5)
    features["away_table_position"] = features["away_table_position"].fillna(10.5)
    features["h2h_home_pts"] = features["h2h_home_pts"].fillna(0)
    features["h2h_away_pts"] = features["h2h_away_pts"].fillna(0)
    return features


def prepare_dataset() -> pd.DataFrame:
    """Load validated matches, build features, and prepare the training/evaluation set.

    Rows without enough team history (has_min_history == False) are excluded entirely -
    see vault/Architecture-Decisions.md ADR-004/007 for the cold-start rationale.
    Season-opener fixtures (no as-of table position yet) and never-met team pairs
    (no h2h history) are imputed with neutral values rather than dropped, since dropping
    them would discard ~160 otherwise valid matches per season-opener alone.
    """
    matches = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
    features = build_features(matches)
    features = features[features["has_min_history"]].copy()
    features = impute_missing_features(features)
    features = features.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return features


def train_and_log(
    model_name: str, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = MODEL_SPECS[model_name]
    fold_results, oof_predictions = walk_forward_validate(
        data,
        feature_cols=FEATURE_COLUMNS,
        target_col=TARGET_COLUMN,
        model_factory=spec["factory"],
        param_grid=spec["param_grid"],
        min_train_seasons=MIN_TRAIN_SEASONS,
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("min_train_seasons", MIN_TRAIN_SEASONS)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("nested_tuning", spec["param_grid"] is not None)

        for _, row in fold_results.iterrows():
            mlflow.log_metric("log_loss", row["log_loss"], step=int(row["test_season"]))

        mlflow.log_metric("mean_log_loss", fold_results["log_loss"].mean())
        mlflow.log_metric("std_log_loss", fold_results["log_loss"].std())

        FOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fold_path = FOLD_RESULTS_DIR / f"{model_name}_folds.csv"
        fold_results.to_csv(fold_path, index=False)
        mlflow.log_artifact(str(fold_path))

        oof_path = FOLD_RESULTS_DIR / f"{model_name}_oof_predictions.csv"
        oof_predictions.to_csv(oof_path)
        mlflow.log_artifact(str(oof_path))

    logger.info(
        "%s: mean log-loss %.4f (+/- %.4f)",
        model_name,
        fold_results["log_loss"].mean(),
        fold_results["log_loss"].std(),
    )
    return fold_results, oof_predictions


def train_all() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    mlflow.set_experiment("football-outcome-mlops")
    data = prepare_dataset()
    return {name: train_and_log(name, data) for name in MODEL_SPECS}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    train_all()
