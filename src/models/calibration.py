"""Post-hoc probability recalibration (Platt scaling / isotonic regression).

Only needed if a model is trained with class reweighting (see
vault/Architecture-Decisions.md ADR-004) - reweighting distorts predicted
probabilities away from real-world frequencies. This module lets that
recalibration be applied and, importantly, its benefit measured: fit
calibrators on an early slice of out-of-fold predictions, then check whether
they actually improve log-loss on a later, untouched slice - so "we didn't
need recalibration" is a verified finding, not an assumption.
"""

import logging

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.models.validation import CLASSES

logger = logging.getLogger(__name__)


def fit_calibrators(oof_calib: pd.DataFrame, method: str = "isotonic") -> dict:
    """Fit one binary calibrator per class (one-vs-rest) on held-out out-of-fold predictions."""
    calibrators = {}
    for cls in CLASSES:
        y_binary = (oof_calib["true_label"] == cls).astype(int)
        x = oof_calib[f"proba_{cls}"].to_numpy()
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(x, y_binary)
        elif method == "sigmoid":
            cal = LogisticRegression()
            cal.fit(x.reshape(-1, 1), y_binary)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        calibrators[cls] = cal
    return calibrators


def apply_calibrators(oof: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    """Apply fitted calibrators and renormalize so probabilities sum to 1 across classes."""
    calibrated = oof.copy()
    for cls in CLASSES:
        x = oof[f"proba_{cls}"].to_numpy()
        cal = calibrators[cls]
        if isinstance(cal, IsotonicRegression):
            calibrated[f"proba_{cls}"] = cal.predict(x)
        else:
            calibrated[f"proba_{cls}"] = cal.predict_proba(x.reshape(-1, 1))[:, 1]

    total = calibrated[[f"proba_{c}" for c in CLASSES]].sum(axis=1)
    for cls in CLASSES:
        calibrated[f"proba_{cls}"] = calibrated[f"proba_{cls}"] / total
    return calibrated


def evaluate_recalibration_benefit(oof: pd.DataFrame, split_season: int, method: str = "isotonic") -> dict:
    """Fit calibrators on seasons < split_season, measure log-loss change on seasons >= split_season.

    A negative `log_loss_change` means recalibration helped; positive means it hurt.
    """
    calib_fit_set = oof[oof["test_season"] < split_season]
    calib_eval_set = oof[oof["test_season"] >= split_season]

    proba_cols = [f"proba_{c}" for c in CLASSES]
    loss_before = log_loss(calib_eval_set["true_label"], calib_eval_set[proba_cols].to_numpy(), labels=CLASSES)

    calibrators = fit_calibrators(calib_fit_set, method=method)
    recalibrated = apply_calibrators(calib_eval_set, calibrators)
    loss_after = log_loss(recalibrated["true_label"], recalibrated[proba_cols].to_numpy(), labels=CLASSES)

    return {
        "method": method,
        "split_season": split_season,
        "n_calib_fit": len(calib_fit_set),
        "n_calib_eval": len(calib_eval_set),
        "log_loss_before": loss_before,
        "log_loss_after": loss_after,
        "log_loss_change": loss_after - loss_before,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.models.evaluate import load_oof

    oof = load_oof("random_forest")
    seasons = sorted(oof["test_season"].unique())
    split_season = seasons[len(seasons) // 2]
    for method in ["isotonic", "sigmoid"]:
        result = evaluate_recalibration_benefit(oof, split_season, method=method)
        logger.info("%s recalibration: %s", method, result)
