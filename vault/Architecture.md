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
- Serving: FastAPI (`/predict`, `/health`), Docker (build unverified — ADR-011)

## Scope (Phase 2 — full MLOps stack, in progress)

Resequenced (see `Architecture-Decisions.md`) to do what's verifiable in this
dev environment first: Evidently drift monitoring, the negative
promote-if-better test, GitHub Actions CI, and Kubernetes manifests. Airflow
and MinIO both require Docker, which isn't available here (same constraint as
ADR-011) — written and reasoned through, verification deferred.

## Out of scope for Phase 2 (deferred to Phase 3)

Streamlit demo, HF Spaces deployment, architecture diagram, Medium article.

## Data flow

```
football-data.co.uk (CSV)
  -> src/data/ingest.py (direct requests/pandas, DVC-tracked raw data)
  -> src/data/validate.py (schema/range checks)
  -> src/features/build_features.py (leak-free feature engineering)
  -> src/models/train.py (walk-forward + nested tuning, MLflow logging)
  -> src/models/evaluate.py (log-loss, Brier, calibration, bootstrap CI)
  -> src/models/calibration.py (recalibration, if class reweighting is used)
  -> src/models/registry.py (MLflow Model Registry, "production" alias)
  -> src/serving/api.py (FastAPI, loads the aliased model)
  -> src/monitoring/drift.py (Evidently: feature drift, recent seasons vs history)
```

This document is updated as each component is built.
