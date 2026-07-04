"""Evaluate trained models: Brier score, calibration, bootstrap CI vs bookmaker, feature importance/ablation.

Reuses the out-of-fold predictions saved by src/models/train.py, so every
number here is computed on genuinely held-out walk-forward test data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.models.train import (
    FEATURE_COLUMNS,
    FOLD_RESULTS_DIR,
    MIN_TRAIN_SEASONS,
    MODEL_SPECS,
    TARGET_COLUMN,
    make_random_forest,
    prepare_dataset,
)
from src.models.validation import CLASSES, walk_forward_validate

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "reports"

FEATURE_GROUPS = {
    "form": [c for c in FEATURE_COLUMNS if "form_pts" in c or "goals_" in c or "streak" in c],
    "rest_days": [c for c in FEATURE_COLUMNS if "rest_days" in c],
    "table_position": [c for c in FEATURE_COLUMNS if "table_position" in c],
    "h2h": [c for c in FEATURE_COLUMNS if c.startswith("h2h")],
    "odds": [c for c in FEATURE_COLUMNS if c.startswith("odds_")],
}


def load_oof(model_name: str) -> pd.DataFrame:
    """Load out-of-fold predictions, renormalizing probabilities to sum exactly to 1.

    XGBoost's predict_proba has float32-level precision (~1e-7 rounding error). After
    a CSV round-trip pandas reads columns back as float64, and sklearn's log_loss
    applies a much tighter float64 tolerance - triggering a spurious "does not sum to
    one" warning for an already-negligible rounding difference. Renormalizing fixes
    the root cause instead of silencing the warning.
    """
    path = FOLD_RESULTS_DIR / f"{model_name}_oof_predictions.csv"
    df = pd.read_csv(path, index_col=0)
    proba_cols = [f"proba_{c}" for c in CLASSES]
    df[proba_cols] = df[proba_cols].div(df[proba_cols].sum(axis=1), axis=0)
    return df


def log_loss_of(oof: pd.DataFrame) -> float:
    proba = oof[[f"proba_{c}" for c in CLASSES]].to_numpy()
    return log_loss(oof["true_label"], proba, labels=CLASSES)


def brier_score(oof: pd.DataFrame) -> float:
    """Multiclass Brier score: mean squared error between one-hot true label and predicted probs."""
    y_true_onehot = np.column_stack([(oof["true_label"] == cls).astype(float) for cls in CLASSES])
    y_pred = oof[[f"proba_{cls}" for cls in CLASSES]].to_numpy()
    return float(np.mean(np.sum((y_true_onehot - y_pred) ** 2, axis=1)))


def calibration_table(oof: pd.DataFrame, cls: str, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data for one class: mean predicted prob vs actual frequency, per bin."""
    probs = oof[f"proba_{cls}"]
    bins = pd.cut(probs, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    actual = (oof["true_label"] == cls).astype(float)
    table = (
        pd.DataFrame({"bin": bins, "predicted": probs, "actual": actual})
        .groupby("bin", observed=True)
        .agg(mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean"), n=("actual", "size"))
        .reset_index()
    )
    return table


def bootstrap_log_loss_diff(
    oof_model: pd.DataFrame, oof_baseline: pd.DataFrame, n_boot: int = 2000, seed: int = 42
) -> dict:
    """95% bootstrap CI for (model log-loss - baseline log-loss) over the shared test matches.

    Resamples matches (not folds) with replacement, so the CI reflects match-level
    variance in outcomes - the same logic as bootstrapping any paired comparison.
    """
    common_idx = oof_model.index.intersection(oof_baseline.index)
    model = oof_model.loc[common_idx]
    baseline = oof_baseline.loc[common_idx]
    assert (model["true_label"].to_numpy() == baseline["true_label"].to_numpy()).all(), (
        "model and baseline out-of-fold predictions must cover the same matches"
    )

    true_labels = model["true_label"].to_numpy()
    model_proba = model[[f"proba_{c}" for c in CLASSES]].to_numpy()
    baseline_proba = baseline[[f"proba_{c}" for c in CLASSES]].to_numpy()

    point_diff = log_loss(true_labels, model_proba, labels=CLASSES) - log_loss(
        true_labels, baseline_proba, labels=CLASSES
    )

    rng = np.random.default_rng(seed)
    n = len(true_labels)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = log_loss(true_labels[idx], model_proba[idx], labels=CLASSES) - log_loss(
            true_labels[idx], baseline_proba[idx], labels=CLASSES
        )

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {"point_diff": point_diff, "ci_low": ci_low, "ci_high": ci_high, "n_matches": n}


def _rf_walk_forward_log_loss(data: pd.DataFrame, feature_cols: list[str]) -> float:
    fold_results, _ = walk_forward_validate(
        data,
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        model_factory=make_random_forest,
        param_grid=None,
        min_train_seasons=MIN_TRAIN_SEASONS,
    )
    return fold_results["log_loss"].mean()


def ablation_study(data: pd.DataFrame) -> pd.DataFrame:
    """Log-loss impact of removing each feature group, using a fixed (non-tuned) Random Forest."""
    rows = [{"removed_group": "none", "n_features": len(FEATURE_COLUMNS), "mean_log_loss": _rf_walk_forward_log_loss(data, FEATURE_COLUMNS)}]
    for group, cols in FEATURE_GROUPS.items():
        remaining = [c for c in FEATURE_COLUMNS if c not in cols]
        rows.append(
            {
                "removed_group": group,
                "n_features": len(remaining),
                "mean_log_loss": _rf_walk_forward_log_loss(data, remaining),
            }
        )
    result = pd.DataFrame(rows)
    result["log_loss_increase_vs_full"] = result["mean_log_loss"] - result.loc[result["removed_group"] == "none", "mean_log_loss"].iloc[0]
    return result


def feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    """Feature importance from a Random Forest fit on the full dataset (for interpretability only)."""
    model = make_random_forest({})
    model.fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_evaluation() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_dataset()

    oof = {name: load_oof(name) for name in MODEL_SPECS}
    summary_rows = []
    for name, o in oof.items():
        summary_rows.append({"model": name, "log_loss": log_loss_of(o), "brier_score": brier_score(o), "n_matches": len(o)})
    summary = pd.DataFrame(summary_rows).sort_values("log_loss")
    summary.to_csv(REPORTS_DIR / "model_summary.csv", index=False)
    logger.info("Model summary:\n%s", summary.to_string(index=False))

    ci_rows = []
    for name in MODEL_SPECS:
        if name == "bookmaker_baseline":
            continue
        ci = bootstrap_log_loss_diff(oof[name], oof["bookmaker_baseline"])
        ci_rows.append({"model": name, **ci})
        logger.info(
            "%s vs bookmaker: log-loss diff %.4f, 95%% CI [%.4f, %.4f] (negative = model beats bookmaker)",
            name, ci["point_diff"], ci["ci_low"], ci["ci_high"],
        )
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(REPORTS_DIR / "bootstrap_ci_vs_bookmaker.csv", index=False)

    home_calibration = calibration_table(oof["random_forest"], "H")
    home_calibration.to_csv(REPORTS_DIR / "random_forest_calibration_home.csv", index=False)

    ablation = ablation_study(data)
    ablation.to_csv(REPORTS_DIR / "random_forest_ablation.csv", index=False)
    logger.info("Ablation study (Random Forest):\n%s", ablation.to_string(index=False))

    importance = feature_importance(data)
    importance.to_csv(REPORTS_DIR / "random_forest_feature_importance.csv", index=False)
    logger.info("Top 5 features:\n%s", importance.head(5).to_string(index=False))

    return {
        "summary": summary,
        "bootstrap_ci": ci_df,
        "calibration": home_calibration,
        "ablation": ablation,
        "feature_importance": importance,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_evaluation()
