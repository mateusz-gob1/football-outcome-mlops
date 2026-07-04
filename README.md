# Football Match Outcome Prediction — MLOps Pipeline

Predicts football match outcomes (Home Win / Draw / Away Win) with a focus on
methodological rigor (leak-free temporal validation, calibrated probabilities,
statistically grounded evaluation) and a complete MLOps stack (experiment
tracking, model registry, serving, automated retraining).

Model quality is benchmarked against bookmaker odds, not just against random
guessing — the goal is to know honestly how close the model gets to a market
that is very hard to beat.

Status: work in progress (Phase 1 — MVP).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure

See `vault/Architecture.md` for the full design and `vault/Architecture-Decisions.md`
for the reasoning behind key methodological choices (temporal validation, nested
tuning, class imbalance handling, cold-start).
