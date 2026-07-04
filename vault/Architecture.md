# Architecture

## Goal

Predict football match outcomes (Home Win / Draw / Away Win) with rigorous
temporal validation and a complete MLOps stack, benchmarked against bookmaker
odds as the quality baseline.

## Scope (Phase 1 — MVP)

- Data: football-data.co.uk, Premier League, 2010/11–present (via `soccerdata`)
- Feature engineering: rolling form, head-to-head, rest days, home advantage,
  as-of table position, bookmaker implied probability
- Validation: walk-forward (season-by-season), nested hyperparameter tuning
- Models: bookmaker baseline, logistic regression, random forest, XGBoost/LightGBM
- Evaluation: log-loss (primary), Brier score, calibration, bootstrap CI vs baseline
- Experiment tracking: MLflow (tracking + model registry)
- Serving: FastAPI (`/predict`, `/health`), Docker

## Out of scope for Phase 1 (deferred)

Airflow orchestration, Kubernetes manifests, MinIO artifact storage, Evidently
drift monitoring, CI/CD, Streamlit demo — see `Architecture-Decisions.md` for
why these are sequenced later rather than built in parallel.

## Data flow

```
football-data.co.uk (CSV)
  -> src/data/ingest.py (soccerdata, DVC-tracked raw data)
  -> src/data/validate.py (schema/range checks)
  -> src/features/build_features.py (leak-free feature engineering)
  -> src/models/train.py (walk-forward + nested tuning, MLflow logging)
  -> src/models/evaluate.py (log-loss, Brier, calibration, bootstrap CI)
  -> src/models/calibration.py (recalibration, if class reweighting is used)
  -> MLflow Model Registry (Production stage)
  -> src/serving/api.py (FastAPI, loads Production model)
```

This document is updated as each component is built.
