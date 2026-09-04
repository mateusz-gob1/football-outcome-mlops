"""Tests guarding the odds/no-odds feature split in src/models/train.py.

Regression test for a real bug caught during design: FEATURE_COLUMNS and
BOOKMAKER_FEATURE_COLUMNS must stay disjoint, and BookmakerBaseline must keep
working when it's only ever sliced with its own dedicated feature list (not
the shared FEATURE_COLUMNS the other three models use).
"""

import numpy as np
import pandas as pd

from src.models.train import (
    BOOKMAKER_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_SPECS,
    BookmakerBaseline,
)


def test_feature_columns_exclude_odds():
    assert not any(c.startswith("odds_") for c in FEATURE_COLUMNS)
    assert all(c.startswith("odds_") for c in BOOKMAKER_FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS).isdisjoint(BOOKMAKER_FEATURE_COLUMNS)


def test_bookmaker_baseline_spec_uses_its_own_feature_columns():
    assert (
        MODEL_SPECS["bookmaker_baseline"]["feature_cols"] == BOOKMAKER_FEATURE_COLUMNS
    )
    for name in ("logistic_regression", "random_forest", "xgboost"):
        assert "feature_cols" not in MODEL_SPECS[name]


def test_bookmaker_baseline_predicts_from_odds_columns_only():
    X = pd.DataFrame(
        {
            "odds_implied_home_prob": [0.5, 0.2],
            "odds_implied_draw_prob": [0.3, 0.3],
            "odds_implied_away_prob": [0.2, 0.5],
        }
    )
    model = BookmakerBaseline().fit(X[BOOKMAKER_FEATURE_COLUMNS], y=None)
    proba = model.predict_proba(X[BOOKMAKER_FEATURE_COLUMNS])

    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # column order must match model.classes_ == ["A", "D", "H"]
    assert np.allclose(proba[0], [0.2, 0.3, 0.5])
