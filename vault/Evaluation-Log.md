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

---

## Run 003 — 2026-09-03

**Data:** same processed dataset as Run 002 (no re-ingestion this run - odds/feature-schema
change only, not a new data pull).
**Change:** removed `odds_implied_home/draw/away_prob` from `FEATURE_COLUMNS` entirely - Logistic
Regression, Random Forest, and XGBoost now train and predict with zero visibility into bookmaker
odds. `BookmakerBaseline` keeps using them via its own dedicated `BOOKMAKER_FEATURE_COLUMNS` list.
See ADR-022. Decided by Mateusz: the project's framing was moving away from "beat the bookmaker,"
and using the bookmaker's own odds as a model input undermined any independent comparison; also
needed for Phase 2 (predicting genuinely future fixtures without a live odds feed).
**Walk-forward folds:** 11 (unchanged)

| Model | Log-loss | Brier |
|---|---|---|
| Bookmaker baseline (B365+BW avg, unchanged) | 0.9596 | 0.5685 |
| Logistic Regression | 1.0025 | 0.5956 |
| Random Forest | 1.0047 | 0.5996 |
| XGBoost | 1.0171 | 0.6065 |

**Model vs bookmaker log-loss diff (bootstrap, 2000 resamples, 95% CI):**
- Logistic Regression: +0.0430, CI [0.0346, 0.0521]
- Random Forest: +0.0452, CI [0.0372, 0.0527]
- XGBoost: +0.0576, CI [0.0484, 0.0669]

As expected from the Run 001/002 ablation (odds were the single most important feature group by a
wide margin), all three models get measurably worse without them - roughly 4-5x the gap to the
bookmaker seen in Run 002. This is an accepted trade-off, not a bug: the project no longer treats
"closing the gap to the bookmaker" as the goal, and none of the remaining features can substitute
for what pre-match odds encode (injury news, team-news, market sentiment - information no
historical box-score feature captures). This number is expected to stay roughly here going
forward; it is not a target to optimize back down.

**Feature importance / ablation:** ablation's "odds" group was removed from
`src/models/evaluate.py::FEATURE_GROUPS` (it would otherwise silently no-op, since there are no
odds columns left in `FEATURE_COLUMNS` to remove). Remaining groups (form, rest_days,
table_position, h2h) and their relative importance are unchanged from Run 002 in shape, just
computed over 17 features instead of 20 - form still dominates, table position now the single most
important individual feature (previously 2nd/3rd behind the two odds columns).

**Model registry note (ADR-022, same pattern as ADR-013):** the new Random Forest (v10) scored
worse than the already-promoted odds-based version (1.0047 vs 0.9685) and the automatic
promote-if-better gate correctly refused to promote it. Promoted anyway, manually, for the same
reason as ADR-013: this is a feature-*schema* change, so a same-schema quality gate isn't the
right check, and leaving the old model promoted would have broken live serving (confirmed: the
full test suite genuinely failed on `tests/test_api.py` with a real sklearn feature-mismatch error
until the alias was moved).

---

## Run 004 — 2026-09-03

**Data:** re-ingested since Run 003 - now 6100 matches (17 seasons, 2010/11-2026/27; the 2026/27
season file has 20 matches played so far). 4143 out-of-fold test matches (up from 4125 - the
partial 2026/27 season now contributes its own walk-forward test fold).
**Change:** added 14 new features - shots for/against, shots on target for/against, corners
for/against, and cards for (yellow+red combined), each a rolling-5-match average, for both home
and away teams (31 features total, up from 17). See ADR-026. Raised by Mateusz wanting to use
"all the stats that make sense," since football-data.co.uk already carries them; xG was
considered and explicitly deferred (only available from 2026/27 onward - not enough seasons yet
for walk-forward validation).
**Walk-forward folds:** 12 (one more than Run 003 - the new partial 2026/27 season becomes an
additional outer test fold)

| Model | Log-loss | Brier |
|---|---|---|
| Bookmaker baseline (unchanged) | 0.9597 | 0.5686 |
| Random Forest | 0.9935 | 0.5921 |
| Logistic Regression | 0.9951 | 0.5904 |
| XGBoost | 1.0096 | 0.6013 |

All three models improved meaningfully versus Run 003 (Random Forest: 1.0047 -> 0.9935; Logistic
Regression: 1.0025 -> 0.9951; XGBoost: 1.0171 -> 1.0096) - shots/corners/cards carry real signal
beyond what form/table-position/h2h already captured.

**Model vs bookmaker log-loss diff (bootstrap, 2000 resamples, 95% CI):**
- Random Forest: +0.0338, CI [0.0265, 0.0410]
- Logistic Regression: +0.0355, CI [0.0272, 0.0436]
- XGBoost: +0.0499, CI [0.0408, 0.0582]

Gap to the bookmaker narrowed for all three (Run 003 was +0.045 to +0.058) but is still positive
and significant - expected, per ADR-022/024, this was never the goal to close.

**Feature importance (Random Forest):** shots-based features now dominate the top of the list -
`away_shots_against_5`, `home_shots_for_5`, `home_shots_against_5`, `away_shots_for_5` are all in
the top 5, ahead of `home_table_position` (previously the single most important feature in Run 003).

**Ablation (Random Forest, `form` group now includes the new shots/corners/cards stats alongside
goals/streak):** removing `form` costs +0.0050 log-loss (down from +0.0840 in Run 003, since the
group is now much larger and "form" alone isn't singularly dominant the way odds used to be);
`h2h` is now the next most costly group to remove (+0.0084), ahead of `table_position` (+0.0064)
and `rest_days` (+0.0073).

**Model registry note:** Random Forest v18 scored better than the previous production version
(0.9935 vs 1.0047) - the automatic promote-if-better gate promoted it **on its own**, no manual
override needed this time (contrast with Run 002/003, both of which needed a manual promotion for
a feature-schema change). This is the normal case the gate is designed for: a same-schema quality
improvement.

---

## Run 005 — 2026-09-03

**Data:** unchanged from Run 004 (6100 matches, 4143 OOF test matches, 12 walk-forward folds).
**Change:** three things checked in response to "are we getting everything out of these models" -
see ADR-027 for the full reasoning behind each.
1. Added a 5th "model": `EnsembleAverage`, a plain average of logistic regression/random
   forest/XGBoost's probabilities (fixed hyperparameters per member, not nested-tuned).
2. Widened the hyperparameter grids for all three ML models (Random Forest: `min_samples_leaf`;
   XGBoost: `subsample`; Logistic Regression: wider `C` range).
3. Re-ran the isotonic/Platt recalibration check (ADR-009) now that ADR-026's features exist.

| Model | Log-loss | Brier | vs Run 004 |
|---|---|---|---|
| Bookmaker baseline (unchanged) | 0.9597 | 0.5686 | - |
| Random Forest | 0.9936 | 0.5920 | +0.0001 (noise) |
| Logistic Regression | 0.9936 | 0.5899 | +0.0016 (worse; nested-CV variance, see ADR-027) |
| Ensemble | 0.9979 | 0.5943 | new; worse than either RF or LogReg alone |
| XGBoost | 0.9999 | 0.5956 | -0.0097 (real improvement) |

Random Forest and Logistic Regression are now statistically indistinguishable at 4 decimal
places. XGBoost improved meaningfully with the wider grid but still trails the other two. The
ensemble, dragged down by XGBoost, scored worse than either of its two better members - equal-
weight averaging isn't automatically a win.

**Model registry note:** new Random Forest candidate (v22) scored 0.99358 vs the current
production version's 0.99348 - worse by 0.0001, well within noise. The automatic promote-if-better
gate correctly declined to promote it. No manual override - this is exactly the case the gate is
designed for (same feature schema, marginal/no real improvement).

**Recalibration (isotonic, evaluated optimistically on the same OOF data used to fit it - see
ADR-027's caveat):**

| Model | Baseline | Isotonic | Platt |
|---|---|---|---|
| Logistic Regression | 0.9936 | 0.9807 | 0.9902 |
| Random Forest | 0.9936 | 0.9839 | 0.9927 |
| XGBoost | 0.9999 | 0.9872 | 0.9966 |
| Ensemble | 0.9979 | 0.9858 | 0.9950 |

Every model improves under isotonic recalibration by 0.01-0.013 log-loss - the opposite of
ADR-009's original (pre-ADR-026) finding. Not yet built into the actual served models (needs a
leak-free, per-fold version first, not the same-data check done here) - flagged as the most
promising concrete next step, ahead of further model/hyperparameter search.

**Update same day - fair re-check (ADR-028):** the same-data isotonic result above doesn't
survive an honest, leak-free re-measurement (`leakfree_recalibrated_log_loss()` - recalibrator
fit only on strictly earlier walk-forward seasons, never the season being scored).

| Model | Baseline | Isotonic (leak-free) | Platt (leak-free) |
|---|---|---|---|
| Logistic Regression | 0.9936 | 1.1211 (much worse) | 0.9906 (better) |
| Random Forest | 0.9936 | 1.0472 (much worse) | 0.9948 (worse) |
| XGBoost | 0.9999 | 1.0556 (worse) | 0.9975 (better) |
| Ensemble | 0.9979 | 1.0476 (worse) | 0.9959 (better) |

n=3773 (2015, the earliest test season, dropped - nothing earlier to calibrate from).

Isotonic's earlier "improvement" was overfitting: it's flexible enough to fit noise in the small
per-season calibration samples (as few as ~340 matches), and that noise doesn't generalize to the
next season. Platt scaling (simpler, 2 parameters) mostly survives the fair check - real,
modest gains for 3 of 4 models - but makes Random Forest (the current production model) very
slightly *worse*. **No recalibration change deployed to production as a result of this check.**
See ADR-028 for the full writeup.
