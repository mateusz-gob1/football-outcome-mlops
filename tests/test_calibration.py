import numpy as np
import pandas as pd

from src.models.calibration import leakfree_recalibrated_log_loss, recalibrated_log_loss
from src.models.evaluate import log_loss_of


def _miscalibrated_oof(n=400, seed=0) -> pd.DataFrame:
    """Synthetic OOF where the stated home-win probability is a compressed
    (underconfident) version of the true one - a textbook case recalibration
    should be able to fix, used to prove the mechanism actually works.
    """
    rng = np.random.default_rng(seed)
    true_home_prob = rng.uniform(0, 1, size=n)
    actual = rng.binomial(1, true_home_prob)

    # Compressed toward 0.5 - systematically underconfident.
    stated_home = 0.5 + (true_home_prob - 0.5) * 0.3
    stated_away = 1 - stated_home

    return pd.DataFrame(
        {
            "true_label": np.where(actual == 1, "H", "A"),
            "proba_H": stated_home,
            "proba_D": 0.0,
            "proba_A": stated_away,
        }
    )


def test_isotonic_recalibration_improves_a_systematically_underconfident_model():
    oof = _miscalibrated_oof()
    baseline = log_loss_of(oof)
    isotonic = recalibrated_log_loss(oof, "isotonic")
    assert isotonic < baseline


def test_recalibrated_probabilities_still_sum_to_one():
    from src.models.calibration import _recalibrate_one_class
    from src.models.validation import CLASSES

    # Platt scaling needs both classes present for whichever outcome it's
    # fitting - a real three-way OOF set (thousands of matches) always has
    # some draws; this hand-built fixture must too, or LogisticRegression
    # can't fit on the "is this a draw" indicator at all.
    rng = np.random.default_rng(1)
    n = 60
    true_label = rng.choice(["H", "D", "A"], size=n, p=[0.45, 0.25, 0.3])
    oof = pd.DataFrame(
        {
            "true_label": true_label,
            "proba_H": rng.uniform(0.2, 0.6, size=n),
            "proba_D": rng.uniform(0.1, 0.4, size=n),
            "proba_A": rng.uniform(0.1, 0.4, size=n),
        }
    )
    oof[["proba_H", "proba_D", "proba_A"]] = oof[
        ["proba_H", "proba_D", "proba_A"]
    ].div(oof[["proba_H", "proba_D", "proba_A"]].sum(axis=1), axis=0)

    recalibrated = {
        cls: _recalibrate_one_class(
            oof[f"proba_{cls}"].to_numpy(),
            (oof["true_label"] == cls).astype(float).to_numpy(),
            "platt",
        )
        for cls in CLASSES
    }
    matrix = np.column_stack([recalibrated[c] for c in CLASSES])
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def _multi_season_oof(season_sizes: dict, seed=2) -> pd.DataFrame:
    """Synthetic multi-season OOF (needs a `test_season` column, unlike the
    single-batch fixtures above) so leak-free per-season recalibration has
    something to walk through season by season.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for season, n in season_sizes.items():
        true_label = rng.choice(["H", "D", "A"], size=n, p=[0.45, 0.25, 0.3])
        probs = rng.uniform(0.1, 0.5, size=(n, 3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        frames.append(
            pd.DataFrame(
                {
                    "test_season": season,
                    "true_label": true_label,
                    "proba_H": probs[:, 0],
                    "proba_D": probs[:, 1],
                    "proba_A": probs[:, 2],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_leakfree_recalibration_drops_the_earliest_season():
    oof = _multi_season_oof({2015: 10, 2016: 20, 2017: 30})
    result = leakfree_recalibrated_log_loss(oof, "isotonic")

    assert result["skipped_season"] == 2015
    # 2015 has nothing earlier to calibrate from and is dropped entirely;
    # 2016 is calibrated from 2015, 2017 from 2015+2016.
    assert result["n_matches"] == 20 + 30
    assert np.isfinite(result["log_loss"])


def test_leakfree_recalibration_never_uses_the_test_season_itself_to_fit(monkeypatch):
    """Regression test for the actual leak this function exists to avoid:
    patches the fit-and-apply step to record exactly which rows it was
    asked to fit on for each call, and asserts none of them come from the
    season being scored.
    """
    import src.models.calibration as calibration_module

    oof = _multi_season_oof({2015: 10, 2016: 20, 2017: 30})
    seen_train_sizes = []

    real_fit_and_apply = calibration_module._fit_and_apply

    def spy(train_probs, train_actual, test_probs, method):
        seen_train_sizes.append(len(train_probs))
        return real_fit_and_apply(train_probs, train_actual, test_probs, method)

    monkeypatch.setattr(calibration_module, "_fit_and_apply", spy)
    calibration_module.leakfree_recalibrated_log_loss(oof, "isotonic")

    # 3 classes x (2016's fit-on-2015 + 2017's fit-on-2015+2016) = 6 calls.
    assert seen_train_sizes == [10, 10, 10, 30, 30, 30]
