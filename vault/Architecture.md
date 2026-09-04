# Architecture

## Goal

Predict football match outcomes (Home Win / Draw / Away Win) with rigorous
temporal validation and a complete MLOps stack, benchmarked against bookmaker
odds as the quality baseline.

## Scope (Phase 1 — MVP, complete)

- Data: football-data.co.uk, Premier League, 2010/11–2025/26, direct HTTP
  download (see ADR-008 for why `soccerdata` was dropped)
- Feature engineering: rolling form, head-to-head, rest days, home advantage,
  as-of table position, bookmaker implied probability
- Validation: walk-forward (season-by-season), nested hyperparameter tuning
- Models: bookmaker baseline, logistic regression, random forest, XGBoost
- Evaluation: log-loss (primary), Brier score, calibration, bootstrap CI vs baseline
- Experiment tracking: MLflow (tracking + model registry, via aliases — ADR-010)
- Serving: FastAPI (`/predict`, `/health`), Docker (verified end-to-end by
  GitHub Actions CI on every push — ADR-011/014)

## Scope (Phase 2 — full MLOps stack, complete)

Resequenced (see `Architecture-Decisions.md`) to do what's verifiable in this
dev environment first: Evidently drift monitoring, the negative
promote-if-better test, GitHub Actions CI, multi-bookmaker baseline
(ADR-012), Kubernetes manifests (ADR-016/017 — written and reasoned through,
not applied to a live cluster, matching the plan's own scope), the Airflow
DAG (ADR-018), and MinIO artifact storage (ADR-019). All four CI jobs
(`lint`, `test`, `docker`, `airflow`) verify this end-to-end on every push -
see the README's Continuous Integration section.

## Scope (Phase 3 — portfolio polish, in progress)

Done: static demo dashboard (`frontend/`, ADR-020). Remaining: architecture
diagram. Not pursued: Medium article (scope trimmed by explicit decision).

## Pivot (2026-09): away from "beat the bookmaker", toward a live matchday site

Decided by Mateusz: the original framing (benchmark against a bookmaker
baseline, headline result "none of our models beat the market") is being
de-emphasized in favor of a public-facing site showing **predictions for the
next Premier League gameweek** from all three ML models side by side, with
the bookmaker's own pick shown only as a reference, updated automatically
after every round. See `vault/Architecture-Decisions.md` ADR-022/023 and
`vault/Evaluation-Log.md` Run 003 for the concrete changes so far:

- **Phase 1 (done):** bookmaker odds removed as a model *input* feature
  (ADR-022) - the three ML models now predict independent of the betting
  market; the bookmaker baseline keeps using odds via its own dedicated
  feature list.
- **Phase 2 (done):** `src/pipeline/predict_upcoming.py` predicts the next
  *unplayed* gameweek from all three ML models at once, sourcing fixtures
  from openfootball/england (ADR-023) since football-data.co.uk has no
  future-fixture list, with pre-match odds attached opportunistically for
  display only.
- **Phase 3 (done):** frontend redesigned around a "Next Matchday" home view
  (all three ML models per fixture, club badges via TheSportsDB with a
  colored-initials fallback, a per-gameweek model-agreement chart),
  multi-model historical browsing (model-select dropdown), the
  bookmaker-comparison content demoted to a secondary "Methodology" section,
  and a README rewrite dropping the "beat the bookmaker" framing (ADR-024).
- **Phase 4 (not started):** weekly GitHub Actions automation so the whole
  cycle (retrain -> predict next gameweek -> refresh the demo) runs without
  a manual step.

## Data flow

```
football-data.co.uk (season CSVs)                openfootball/england (future fixtures)
  -> src/data/ingest.py (DVC-tracked raw data)      -> src/data/fixtures_openfootball.py
  -> src/data/validate.py (schema/range checks)     -> src/data/team_names.py (name normalization)
  -> src/features/build_features.py (leak-free feature engineering, historical matches)
                                                     -> src/data/fixtures_odds.py (pre-match odds, display-only)
  -> src/models/train.py (walk-forward + nested tuning, MLflow logging;      |
       bookmaker baseline uses its own odds-only feature list - ADR-022)     |
  -> src/models/evaluate.py (log-loss, Brier, calibration, bootstrap CI)     |
  -> src/models/registry.py (MLflow Model Registry, "production" alias;     |
       fit_full_model() generalizes "fit on all data" beyond one candidate) |
  -> src/serving/api.py (FastAPI, single-fixture live predict)   <----------+
  -> src/pipeline/predict_upcoming.py (whole next gameweek, all 3 ML models,
       via src/features/query_features.py - the same leak-free
       synthetic-fixture trick api.py uses, shared instead of duplicated)
  -> src/monitoring/drift.py (Evidently: feature drift, recent seasons vs history)
  -> dags/retrain_dag.py (Airflow: the retrain cycle, scheduled weekly)
  -> frontend/prepare_data.py (exports out-of-fold + upcoming-gameweek results to static JSON for the demo)
```

This document is updated as each component is built.
