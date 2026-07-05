"""Walk-forward validation with nested hyperparameter tuning.

Model-agnostic: `model_factory(params)` must return an object exposing
`.fit(X, y)` and `.predict_proba(X)` (the scikit-learn estimator interface).

Outer loop: expanding walk-forward by season (train on seasons 1..N, test on
season N+1). Inner loop: hyperparameters are chosen using only the training
seasons, holding out the single most recent training season as an inner
validation fold - the outer test season is never touched until the final,
already-tuned model is evaluated on it. See tests/test_validation.py.
"""

from typing import Any, Callable

import pandas as pd
from sklearn.metrics import log_loss

ModelFactory = Callable[[dict], Any]
CLASSES = ["A", "D", "H"]  # alphabetical, matching sklearn's default label ordering


def season_folds(
    seasons: list[int], min_train_seasons: int
) -> list[tuple[list[int], int]]:
    """Expanding walk-forward folds: (train_seasons, test_season) for each step."""
    seasons = sorted(seasons)
    return [(seasons[:i], seasons[i]) for i in range(min_train_seasons, len(seasons))]


def inner_tune(
    data: pd.DataFrame,
    train_seasons: list[int],
    feature_cols: list[str],
    target_col: str,
    model_factory: ModelFactory,
    param_grid: list[dict],
) -> dict:
    """Pick the params with lowest log-loss on an inner holdout (the most recent training season).

    Never receives the outer test season - only `train_seasons` is available here.
    """
    if len(train_seasons) < 2:
        return param_grid[0]

    inner_train_seasons, inner_val_season = train_seasons[:-1], train_seasons[-1]
    train_mask = data["season_start_year"].isin(inner_train_seasons)
    val_mask = data["season_start_year"] == inner_val_season

    best_params, best_loss = param_grid[0], float("inf")
    for params in param_grid:
        model = model_factory(params)
        model.fit(data.loc[train_mask, feature_cols], data.loc[train_mask, target_col])
        proba = model.predict_proba(data.loc[val_mask, feature_cols])
        loss = log_loss(data.loc[val_mask, target_col], proba, labels=model.classes_)
        if loss < best_loss:
            best_params, best_loss = params, loss
    return best_params


def walk_forward_validate(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_factory: ModelFactory,
    param_grid: list[dict] | None = None,
    min_train_seasons: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding walk-forward validation with nested hyperparameter tuning.

    Returns (fold_results, oof_predictions):
    - fold_results: one row per outer fold (test season, train seasons, chosen
      params, fold log-loss, number of test matches).
    - oof_predictions: one row per test match across all folds, with the
      original row index, true label, and predicted probability per class
      (columns proba_A/proba_D/proba_H) - out-of-fold, so safe to use for
      Brier score, calibration, and bootstrap CI without leakage.
    """
    seasons = sorted(data["season_start_year"].unique())
    folds = season_folds(seasons, min_train_seasons)

    results = []
    oof_frames = []
    for train_seasons, test_season in folds:
        best_params = (
            inner_tune(
                data, train_seasons, feature_cols, target_col, model_factory, param_grid
            )
            if param_grid
            else {}
        )

        train_mask = data["season_start_year"].isin(train_seasons)
        test_mask = data["season_start_year"] == test_season
        model = model_factory(best_params)
        model.fit(data.loc[train_mask, feature_cols], data.loc[train_mask, target_col])
        test_data = data.loc[test_mask]
        proba = model.predict_proba(test_data[feature_cols])
        fold_loss = log_loss(test_data[target_col], proba, labels=model.classes_)

        results.append(
            {
                "test_season": test_season,
                "train_seasons": train_seasons,
                "params": best_params,
                "log_loss": fold_loss,
                "n_test": int(test_mask.sum()),
            }
        )

        class_order = list(model.classes_)
        oof = pd.DataFrame(
            {f"proba_{cls}": proba[:, class_order.index(cls)] for cls in CLASSES},
            index=test_data.index,
        )
        oof["test_season"] = test_season
        oof["true_label"] = test_data[target_col].values
        oof_frames.append(oof)

    return pd.DataFrame(results), pd.concat(oof_frames)
