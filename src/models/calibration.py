"""Check whether post-hoc probability recalibration improves walk-forward OOF predictions.

Two standard post-hoc methods, applied per-class (one-vs-rest) directly to
a model's saved out-of-fold probabilities, then renormalized to sum to 1:
isotonic regression (flexible, monotonic) and Platt scaling (a logistic fit
on the raw probability, smoother/less prone to overfitting on small data).

Re-checked after ADR-026 added the shots/corners/cards features (the
original ADR-009 check predates them and found no benefit) - the script
itself didn't survive in the repo, rebuilt here from the same method.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.models.evaluate import load_oof, log_loss_of
from src.models.validation import CLASSES

logger = logging.getLogger(__name__)


def _recalibrate_one_class(probs: np.ndarray, actual: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        return model.fit_transform(probs, actual)
    if method == "platt":
        model = LogisticRegression()
        model.fit(probs.reshape(-1, 1), actual)
        return model.predict_proba(probs.reshape(-1, 1))[:, 1]
    raise ValueError(f"unknown recalibration method: {method}")


def recalibrated_log_loss(oof: pd.DataFrame, method: str) -> float:
    """Fit a per-class recalibration on the OOF probabilities themselves and
    report the resulting log-loss.

    This refits and evaluates on the same data - optimistic versus a real
    deployment, which would recalibrate on a held-out slice - but that's the
    same simplification the original ADR-009 check made. The question here
    is "is there recalibration headroom at all": if it doesn't help even
    under this generous setup, it certainly won't help for real.
    """
    recalibrated = {
        cls: _recalibrate_one_class(
            oof[f"proba_{cls}"].to_numpy(),
            (oof["true_label"] == cls).astype(float).to_numpy(),
            method,
        )
        for cls in CLASSES
    }
    matrix = np.column_stack([recalibrated[c] for c in CLASSES])
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return log_loss(oof["true_label"], matrix, labels=CLASSES)


def _fit_and_apply(train_probs: np.ndarray, train_actual: np.ndarray, test_probs: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(train_probs, train_actual).predict(test_probs)
    if method == "platt":
        model = LogisticRegression().fit(train_probs.reshape(-1, 1), train_actual)
        return model.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    raise ValueError(f"unknown recalibration method: {method}")


def leakfree_recalibrated_log_loss(oof: pd.DataFrame, method: str) -> dict:
    """Leak-free version of recalibrated_log_loss.

    For each outer walk-forward test season (in order), fits the
    recalibrator only on OOF rows from strictly *earlier* seasons, then
    scores that season's rows with it - the same "never look at the future"
    rule this project applies everywhere else (build_features.py's rolling
    stats, the walk-forward split itself). The earliest test season has no
    earlier OOF data to calibrate from and is skipped entirely, the same
    cold-start trade-off MIN_HISTORY already makes elsewhere: losing one
    season's matches from this check rather than faking a calibration for
    them.
    """
    seasons = sorted(oof["test_season"].unique())
    skipped_season = seasons[0]

    all_true = []
    all_proba_rows = []
    for season in seasons[1:]:
        train_rows = oof[oof["test_season"] < season]
        test_rows = oof[oof["test_season"] == season]

        recalibrated = {
            cls: _fit_and_apply(
                train_rows[f"proba_{cls}"].to_numpy(),
                (train_rows["true_label"] == cls).astype(float).to_numpy(),
                test_rows[f"proba_{cls}"].to_numpy(),
                method,
            )
            for cls in CLASSES
        }

        matrix = np.column_stack([recalibrated[c] for c in CLASSES])
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        all_true.append(test_rows["true_label"].to_numpy())
        all_proba_rows.append(matrix)

    true_labels = np.concatenate(all_true)
    proba_matrix = np.vstack(all_proba_rows)
    return {
        "log_loss": log_loss(true_labels, proba_matrix, labels=CLASSES),
        "n_matches": len(true_labels),
        "skipped_season": int(skipped_season),
    }


def check_recalibration(model_name: str) -> dict:
    oof = load_oof(model_name)
    baseline = log_loss_of(oof)
    isotonic = recalibrated_log_loss(oof, "isotonic")
    platt = recalibrated_log_loss(oof, "platt")
    isotonic_lf = leakfree_recalibrated_log_loss(oof, "isotonic")
    platt_lf = leakfree_recalibrated_log_loss(oof, "platt")
    result = {
        "model": model_name,
        "baseline_log_loss": baseline,
        "isotonic_log_loss": isotonic,
        "platt_log_loss": platt,
        "isotonic_helps": isotonic < baseline,
        "platt_helps": platt < baseline,
        "isotonic_leakfree_log_loss": isotonic_lf["log_loss"],
        "platt_leakfree_log_loss": platt_lf["log_loss"],
        "isotonic_leakfree_helps": isotonic_lf["log_loss"] < baseline,
        "platt_leakfree_helps": platt_lf["log_loss"] < baseline,
        "leakfree_n_matches": isotonic_lf["n_matches"],
    }
    logger.info(
        "%s: baseline=%.4f | same-data isotonic=%.4f (%s) platt=%.4f (%s) | "
        "LEAK-FREE isotonic=%.4f (%s) platt=%.4f (%s) [n=%d, season %d dropped]",
        model_name,
        baseline,
        isotonic,
        "better" if result["isotonic_helps"] else "worse",
        platt,
        "better" if result["platt_helps"] else "worse",
        isotonic_lf["log_loss"],
        "better" if result["isotonic_leakfree_helps"] else "worse",
        platt_lf["log_loss"],
        "better" if result["platt_leakfree_helps"] else "worse",
        result["leakfree_n_matches"],
        isotonic_lf["skipped_season"],
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for name in ["logistic_regression", "random_forest", "xgboost", "ensemble"]:
        check_recalibration(name)
