# Evaluation Log

Filled in after each training/evaluation run. Tracks metrics over iterations
so changes to features, models, or validation logic can be judged by their
actual effect on numbers, not assumed.

Template per run:

## Run 00X — YYYY-MM-DD

**Data:** seasons covered, number of matches
**Models compared:** baseline (bookmaker), logistic regression, random forest, XGBoost/LightGBM
**Walk-forward folds:** N

| Model | Log-loss | Brier | Notes |
|---|---|---|---|
| Bookmaker baseline | | | |
| Logistic Regression | | | |
| Random Forest | | | |
| XGBoost | | | |

**Model vs bookmaker log-loss diff:** X, 95% CI [a, b]
**Calibration:** reliability diagram summary
**Changes since last run:**
