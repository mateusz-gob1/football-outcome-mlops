# Football Match Outcome Prediction — MLOps Pipeline

[![CI](https://github.com/mateusz-gob1/football-outcome-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/mateusz-gob1/football-outcome-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicts football match outcomes (Home Win / Draw / Away Win) with a focus on
methodological rigor (leak-free temporal validation, calibrated probabilities,
statistically grounded evaluation) and a complete MLOps stack (experiment
tracking, model registry, serving, monitoring, CI/CD).

Model quality is benchmarked against a bookmaker consensus baseline, not just
against random guessing — the goal is to know honestly how close the model
gets to a market that is very hard to beat.

**Status: Phase 1 (MVP) complete. Phase 2 complete. Phase 3 complete.**
Evidently drift monitoring, retraining pipeline with a promote-if-better
guard, Kubernetes manifests, the Airflow DAG, and MinIO artifact storage are
all verified end-to-end on every push (four green CI jobs: `lint`, `test`,
`docker`, `airflow`). The static demo dashboard is live (see below).

## Architecture

![Architecture diagram](docs/architecture.svg)

Left to right: raw data → validation → leak-free feature engineering →
walk-forward training → MLflow Registry → FastAPI serving. The dashed box is
the Airflow DAG that wraps the same pipeline on a weekly schedule and loops
promote-if-better back into next week's ingest. The demo dashboard sits
outside this loop deliberately — it ships pre-computed exports, not a live
model (see ADR-020).

## Result

Premier League, 2010/11–2025/26 (16 seasons, 4125 out-of-fold walk-forward
test matches). None of the three trained models beat the bookmaker baseline,
and the gap is statistically significant (95% bootstrap CI entirely above
zero for all three) — see `vault/Evaluation-Log.md` Run 002 for the full
breakdown.

| Model | Log-loss | Brier | vs bookmaker (95% CI) |
|---|---|---|---|
| Bookmaker baseline (Bet365 + Bet&Win avg) | **0.9595** | 0.5685 | — |
| Random Forest | 0.9685 | 0.5741 | +0.0090 [0.0057, 0.0124] |
| Logistic Regression | 0.9700 | 0.5729 | +0.0105 [0.0054, 0.0169] |
| XGBoost | 0.9836 | 0.5826 | +0.0241 [0.0177, 0.0309] |

The bookmaker-implied probability is also, unsurprisingly, the single most
important feature group for every model we trained (ablation study, same
log). The baseline averages two bookmakers' overround-normalized
probabilities rather than using one alone — see ADR-012 for why that's a
fairer (harder) comparison, not a softer one.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pre-commit install  # optional: black + ruff on every commit
```

## Running the pipeline

```bash
python -m src.data.ingest        # download 16 seasons of Premier League CSVs
python -m src.data.validate      # schema/anomaly checks -> data/processed/matches_validated.csv
python -m src.models.train       # walk-forward CV + nested tuning, 4 models, MLflow logging
python -m src.models.evaluate    # Brier, calibration, bootstrap CI, feature importance/ablation
python -m src.models.calibration # verifies whether recalibration would help (it doesn't - see ADR-009)
python -m src.models.registry    # registers the best model, promotes to the "production" alias
python -m src.monitoring.drift   # Evidently data drift report (recent seasons vs history)
```

Or run the whole cycle at once (this is what a scheduled retraining job
would call - see `src/pipeline/retrain_pipeline.py`):

```bash
python -m src.pipeline.retrain_pipeline
```

## Running the API

```bash
uvicorn src.serving.api:app --reload
```

- `GET /health` — service status and loaded model version
- `POST /predict` — requires `home_team`, `away_team`, `match_date`, and
  Bet365 odds (`b365h`, `b365d`, `b365a`); Bet&Win odds (`bwh`, `bwd`, `bwa`)
  are optional and averaged in when provided. Odds are otherwise a required
  input, not optional — see ADR in `vault/Architecture-Decisions.md` on why
  (the API is designed for near-term fixtures, where odds already exist, not
  for predicting far-future/pre-season matches).

`docker-compose.yml` runs the API alongside MLflow (backed by MinIO for
artifact storage - `s3://mlflow-artifacts`, see ADR-019) for local
development, and is exercised end-to-end (build, register a real model
against the containerized MLflow, start, `/health`, `/predict`) by the
`docker` job in `.github/workflows/ci.yml` on every push - see ADR-011/014.
MinIO's web console is at `http://localhost:9001`
(`minioadmin` / `minioadmin123` - dev-only credentials).

## Orchestration (Airflow)

`dags/retrain_dag.py` wraps the same five steps as
`src/pipeline/retrain_pipeline.py` (ingest, validate, train, evaluate,
promote-if-better) as an Airflow DAG on a weekly schedule. Run locally via:

```bash
docker compose -f docker-compose.airflow.yml up
# UI at http://localhost:8080
```

Verified end-to-end by the `airflow` job in CI - not just that the DAG
parses, but that a real trigger runs all five tasks to completion. See
ADR-018 for the `airflow standalone` (single-container) choice and why.

## Continuous Integration

Every push to `main` runs four jobs on GitHub's own runners, not just locally
claimed to work: **[see live runs →](https://github.com/mateusz-gob1/football-outcome-mlops/actions)**

| Job | What it actually does |
|---|---|
| `lint` | `black --check` + `ruff check` across the whole codebase |
| `test` | Fresh `ingest → validate → train → registry` run, then all 18 `pytest` tests |
| `docker` | Builds both images, brings up MinIO + MLflow, registers a real model against them, starts the API, and curls `/health` + `/predict` |
| `airflow` | Builds the Airflow image, starts it, and **triggers the actual DAG** - all 5 tasks run to completion, not just "the file parses" |

Each of these is documented in `vault/Architecture-Decisions.md` with the
real bugs it caught on the way to green (ADR-014, ADR-018, ADR-019) - the
point isn't that it worked on the first try, it's that every claim here is
checked by a machine on every push, not asserted in prose.

## Demo

**Live: [matigob-football-outcome-predictor.static.hf.space](https://matigob-football-outcome-predictor.static.hf.space)**

`frontend/` is a standalone static dashboard (browse historical predictions
vs bookmaker odds vs actual results, plus the full model-vs-bookmaker
comparison from the Result table above) - plain HTML/CSS/JS, no server, no
live model. It ships with pre-computed out-of-fold predictions rather than a
live model - see ADR-020 for why. Deployed to Hugging Face Spaces (Static
SDK) via `git subtree push --prefix=frontend`. Run locally:

```bash
cd frontend
python -m http.server 8000
# open http://localhost:8000
```

Regenerate the underlying JSON (after retraining, or if the data changes):

```bash
python frontend/prepare_data.py
```

## Tests

```bash
pytest
```

18 tests: leak-free feature engineering (rolling stats, as-of table position,
cold-start, H2H window), walk-forward validation (nested tuning never touches
the outer test season), the API (valid predictions, unknown team /
zero-history / bad-input error handling), and the model registry's
promote-if-better guard (a deliberately worse candidate must not move the
`production` alias).

## Project structure

See `vault/Architecture.md` for the full design and `vault/Architecture-Decisions.md`
for the reasoning behind key methodological choices (temporal validation, nested
tuning, class imbalance handling, cold-start, multi-bookmaker baseline, MLflow
registry, Docker, CI/CD).
