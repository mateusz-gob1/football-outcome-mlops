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

---

## Run 001 — 2026-07-04

**Data:** Premier League, seasons 2010/11-2025/26 (16 seasons, 6080 matches ingested; 4954 used for
training/evaluation after excluding cold-start rows per ADR-004/007)
**Models compared:** bookmaker baseline (Bet365 implied probability), logistic regression, random
forest, XGBoost
**Walk-forward folds:** 11 (expanding window, min. 5 training seasons, nested hyperparameter tuning
on the innermost held-out season) — 4125 out-of-fold test matches total

| Model | Log-loss | Brier |
|---|---|---|
| Bookmaker baseline | 0.9598 | 0.5686 |
| Random Forest | 0.9683 | 0.5740 |
| Logistic Regression | 0.9701 | 0.5730 |
| XGBoost | 0.9807 | 0.5807 |

**Model vs bookmaker log-loss diff (bootstrap, 2000 resamples, 95% CI):**
- Random Forest: +0.0085, CI [0.0051, 0.0120]
- Logistic Regression: +0.0103, CI [0.0052, 0.0167]
- XGBoost: +0.0208, CI [0.0145, 0.0275]

All three intervals are entirely above zero: none of our models beat the bookmaker, and the gap is
statistically significant, not noise. This matches the plan's expectation that the betting market is
a very hard baseline to beat - an honest result, not a failure.

**Calibration (Random Forest, home-win predictions):** well calibrated across the full probability
range without any class reweighting - e.g. predicted ~0.75 -> actual ~0.74, predicted ~0.82 ->
actual ~0.85. Confirms ADR-004 (leave class distribution natural, no explicit recalibration needed
for this run - see vault/Architecture-Decisions.md ADR-009).

**Feature importance / ablation (Random Forest):** removing the bookmaker-odds feature group hurts
log-loss the most (+0.0368), followed by team-form features (+0.0298). Rest days, table position,
and head-to-head history matter much less (+0.004-0.006 each). Top individual features are the three
odds-implied probabilities themselves, then table position and 10-match form.

**Changes since last run:** initial Phase 1 training run.

---

## Run 002 — 2026-07-05

**Data:** same 16 seasons, 6080 matches ingested; 4954 used for training/evaluation (unchanged
row counts - the multi-bookmaker change affects the odds feature, not which rows are usable).
**Change:** `odds_implied_*` now averages Bet365 + Bet&Win's own overround-normalized
probabilities (linear pool), falling back to Bet365 alone where Bet&Win is missing (144 matches,
mostly the 2024/25 gap) - see ADR-012. Raised directly by Mateusz: comparing against a single
bookmaker understates how strong the baseline really is, since bookmaker consensus is normally a
*more* efficient estimate than any one bookmaker alone.
**Walk-forward folds:** 11 (unchanged)

| Model | Log-loss | Brier |
|---|---|---|
| Bookmaker baseline (B365+BW avg) | 0.9595 | 0.5685 |
| Random Forest | 0.9685 | 0.5741 |
| Logistic Regression | 0.9700 | 0.5729 |
| XGBoost | 0.9836 | 0.5826 |

**Model vs bookmaker log-loss diff (bootstrap, 2000 resamples, 95% CI):**
- Random Forest: +0.0090, CI [0.0057, 0.0124]
- Logistic Regression: +0.0105, CI [0.0054, 0.0169]
- XGBoost: +0.0241, CI [0.0177, 0.0309]

Same conclusion as Run 001, on a very slightly tougher baseline: the two-bookmaker consensus
(0.9595) is marginally stronger than Bet365 alone (0.9598 in Run 001), and none of the three
models close that gap - the CIs are still entirely above zero.

**Calibration:** unchanged in shape (still well calibrated across H/D/A without reweighting).

**Feature importance / ablation (Random Forest):** unchanged story - removing odds still hurts
most (+0.0354), form second (+0.0259). The (now dual-bookmaker) odds features remain the top 3
individually most important features.

**Recalibration re-check:** isotonic now slightly *hurts* (+0.0340, vs. -0.0002/noise in Run 001)
on this data; Platt/sigmoid still hurts (+0.0041). ADR-009's conclusion (don't recalibrate) still
holds - if anything more clearly than before.

**Model registry note (ADR-013):** the new Random Forest (v6) was marginally *worse* than the
previously-promoted version (0.9685 vs 0.9683) and the automatic promote-if-better gate correctly
refused to promote it. It was promoted anyway, manually, because the *feature definition* changed
underneath it - keeping the old model in production would have created train/serve skew against
the now-current `build_features.py`. A model-quality gate isn't the right check for a feature-schema
change.
