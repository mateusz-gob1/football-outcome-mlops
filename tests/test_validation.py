import numpy as np
import pandas as pd

from src.models.validation import season_folds, walk_forward_validate


def test_season_folds_expand_training_window():
    folds = season_folds([2010, 2011, 2012, 2013, 2014], min_train_seasons=3)
    assert folds == [
        ([2010, 2011, 2012], 2013),
        ([2010, 2011, 2012, 2013], 2014),
    ]


class _RecordingModel:
    """Fake estimator that logs which seasons it was fit/evaluated on."""

    def __init__(self, params: dict, log: list):
        self.params = params
        self.log = log
        self.classes_ = np.array(["H", "D", "A"])

    def fit(self, X, y):
        self.log.append(("fit", set(X["season_start_year"])))
        return self

    def predict_proba(self, X):
        self.log.append(("predict", set(X["season_start_year"])))
        return np.tile([1 / 3, 1 / 3, 1 / 3], (len(X), 1))


def _synthetic_data(seasons: list[int], matches_per_season: int = 4) -> pd.DataFrame:
    rows = []
    for season in seasons:
        for _ in range(matches_per_season):
            rows.append({"season_start_year": season, "x1": 0.0, "target": "H"})
    return pd.DataFrame(rows)


def test_inner_tuning_never_fits_on_the_outer_test_season():
    # A single fold: train on 2010-2012, test on 2013.
    data = _synthetic_data([2010, 2011, 2012, 2013])
    log: list = []

    result, oof = walk_forward_validate(
        data,
        feature_cols=["season_start_year", "x1"],
        target_col="target",
        model_factory=lambda params: _RecordingModel(params, log),
        param_grid=[{"a": 1}, {"a": 2}],
        min_train_seasons=3,
    )

    assert len(result) == 1
    assert result.iloc[0]["test_season"] == 2013
    assert (oof["test_season"] == 2013).all()

    fit_seasons_seen = set().union(*(seasons for kind, seasons in log if kind == "fit"))
    assert 2013 not in fit_seasons_seen, "the outer test season leaked into a .fit() call"

    # The test season must appear in exactly one predict_proba call: the final evaluation.
    predict_calls_with_test_season = [
        seasons for kind, seasons in log if kind == "predict" and 2013 in seasons
    ]
    assert len(predict_calls_with_test_season) == 1
    assert predict_calls_with_test_season[0] == {2013}

    # The final fit (after tuning) uses the full outer training window, not just the inner split.
    fit_calls = [seasons for kind, seasons in log if kind == "fit"]
    assert {2010, 2011, 2012} in fit_calls


def test_walk_forward_validate_without_param_grid_skips_tuning():
    data = _synthetic_data([2010, 2011, 2012, 2013])
    log: list = []

    result, _ = walk_forward_validate(
        data,
        feature_cols=["season_start_year", "x1"],
        target_col="target",
        model_factory=lambda params: _RecordingModel(params, log),
        param_grid=None,
        min_train_seasons=3,
    )

    assert result.iloc[0]["params"] == {}
    fit_seasons_seen = set().union(*(seasons for kind, seasons in log if kind == "fit"))
    assert 2013 not in fit_seasons_seen
