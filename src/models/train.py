"""Train baseline and candidate models with walk-forward validation + MLflow logging."""

import logging
from pathlib import Path
from typing import ClassVar

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
    "home_shots_for_5",
    "home_shots_against_5",
    "home_shots_on_target_for_5",
    "home_shots_on_target_against_5",
    "home_corners_for_5",
    "home_corners_against_5",
    "home_cards_for_5",
    "home_streak_before",
    "home_rest_days",
    "home_table_position",
    "away_form_pts_5",
    "away_form_pts_10",
    "away_goals_scored_5",
    "away_goals_conceded_5",
    "away_shots_for_5",
    "away_shots_against_5",
    "away_shots_on_target_for_5",
    "away_shots_on_target_against_5",
    "away_corners_for_5",
    "away_corners_against_5",
    "away_cards_for_5",
    "away_streak_before",
    "away_rest_days",
    "away_table_position",
    "h2h_home_pts",
    "h2h_away_pts",
    "h2h_matches_count",
]
BOOKMAKER_FEATURE_COLUMNS = [
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
    _label_to_idx: ClassVar[dict[str, int]] = {"A": 0, "D": 1, "H": 2}

    def __init__(self, params: dict):
        self.model = XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_lambda=params.get("reg_lambda", 1.0),
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


# Fixed hyperparameters for the ensemble's three sub-models - each one's own
# already-established best setting from walk-forward tuning (see
# vault/Evaluation-Log.md), not re-tuned per fold. Nested tuning the
# ensemble itself (tuning 3 models' hyperparameters jointly, per fold) would
# multiply the grid size of all three together for a benefit that's usually
# marginal compared to just averaging already-good models - not worth the
# extra training time.
ENSEMBLE_MEMBER_PARAMS = {
    "logistic_regression": {"C": 1.0},
    "random_forest": {"n_estimators": 300, "max_depth": 6},
    "xgboost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1},
}


class EnsembleAverage:
    """Simple average of logistic regression, random forest, and XGBoost's
    own predicted probabilities - each refit from scratch on whatever
    training window walk_forward_validate passes in, so its OOF predictions
    are exactly as leak-free as any individual model's.
    """

    classes_ = np.array(["A", "D", "H"])

    def __init__(self, params: dict | None = None):
        self.models = {
            "logistic_regression": make_logreg(
                ENSEMBLE_MEMBER_PARAMS["logistic_regression"]
            ),
            "random_forest": make_random_forest(
                ENSEMBLE_MEMBER_PARAMS["random_forest"]
            ),
            "xgboost": XGBoostWrapper(ENSEMBLE_MEMBER_PARAMS["xgboost"]),
        }

    def fit(self, X, y):
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for model in self.models.values()]
        return np.mean(probs, axis=0)


MODEL_SPECS = {
    "bookmaker_baseline": {
        "factory": lambda params: BookmakerBaseline(),
        "param_grid": None,
        "feature_cols": BOOKMAKER_FEATURE_COLUMNS,
    },
    "logistic_regression": {
        "factory": make_logreg,
        "param_grid": [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]],
    },
    "random_forest": {
        "factory": make_random_forest,
        "param_grid": [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
            for n in [200, 300, 400]
            for d in [6, 10, None]
            for leaf in [1, 3]
        ],
    },
    "xgboost": {
        "factory": lambda params: XGBoostWrapper(params),
        "param_grid": [
            {
                "n_estimators": n,
                "max_depth": d,
                "learning_rate": lr,
                "subsample": sub,
            }
            for n in [200, 400]
            for d in [3, 5, 7]
            for lr in [0.03, 0.1]
            for sub in [0.8, 1.0]
        ],
    },
    "ensemble": {
        "factory": lambda params: EnsembleAverage(params),
        "param_grid": None,
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
    features = features.dropna(
        subset=FEATURE_COLUMNS + BOOKMAKER_FEATURE_COLUMNS + [TARGET_COLUMN]
    )
    return features


def train_and_log(
    model_name: str, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = MODEL_SPECS[model_name]
    feature_cols = spec.get("feature_cols", FEATURE_COLUMNS)
    fold_results, oof_predictions = walk_forward_validate(
        data,
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        model_factory=spec["factory"],
        param_grid=spec["param_grid"],
        min_train_seasons=MIN_TRAIN_SEASONS,
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("min_train_seasons", MIN_TRAIN_SEASONS)
        mlflow.log_param("n_features", len(feature_cols))
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
