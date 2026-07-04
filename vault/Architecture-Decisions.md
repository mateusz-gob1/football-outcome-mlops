# Architecture Decisions

## ADR-001: Classical ML + MLOps project, separate from the agentic portfolio

**Decision:** This project has no LLM component. It complements `football-agent`
and `financial-doc-agent` (both LangGraph/agentic), which cover agentic AI but
not classical ML fundamentals or MLOps tooling (MLflow, Airflow, K8s).

**Why:** An "AI Engineer" profile needs both agentic system design and solid
ML/MLOps fundamentals. Without this project, the portfolio only demonstrates
the former.

---

## ADR-002: Bookmaker odds as the evaluation baseline, not just random guessing

**Decision:** Every model is compared against the implied probability from
bookmaker odds using log-loss, with a bootstrap confidence interval on the
difference.

**Why:** Beating random guessing is a low bar and not informative. The betting
market is a strong, well-calibrated baseline — comparing against it is honest
and, whether the model wins or loses, produces a real result worth reporting.

---

## ADR-003: Walk-forward validation with nested hyperparameter tuning

**Decision:** No random train/test split. Walk-forward by season (train on
seasons 1..N, test on N+1, slide forward). Hyperparameter tuning happens on an
inner time split within the training portion of each fold, never on the outer
test fold.

**Why:** Match data is sequential; random splits leak future information into
training. Tuning on the same window used for final reporting would leak test
information indirectly into model selection, inflating reported metrics.
This is one of the most commonly skipped rigor points in similar projects.

---

## ADR-004: Class imbalance — leave natural distribution by default

**Decision:** Do not use `class_weight='balanced'` or oversampling by default.
Log-loss as a proper scoring rule already forces the model to output sensible
probabilities for the rarer draw class. If reweighting is used later to
improve draw recall, output probabilities are recalibrated afterward
(Platt scaling / isotonic regression, `src/models/calibration.py`) on a
separate calibration fold.

**Why:** Reweighting without recalibration distorts predicted probabilities
away from real-world frequencies, which directly breaks calibration — one of
the primary evaluation metrics for this project.

---

## ADR-005: Environment — venv + pip

**Decision:** Standard `venv` and `requirements.txt`, no Poetry or conda.

**Why:** Matches the plan's original structure, simplest to explain in an
interview, most portable for reviewers cloning the repo.

---

## ADR-006: Airflow environment — deferred to Phase 2

**Decision:** Not yet decided whether Airflow runs via Docker Compose or WSL2.

**Why:** Airflow has no native Windows support. Decision is deferred until
Phase 2 (retraining orchestration) is actually reached, to avoid designing
around a constraint before it matters. Leaning toward Docker Compose for
consistency with the rest of the stack (all other services are containerized).

---

## ADR-007: Phase 1 scope only, for now

**Decision:** Build Phase 1 (MVP: data through FastAPI serving) completely and
tested before deciding whether/when to build Phase 2 (Airflow, K8s, MinIO,
Evidently) and Phase 3 (Streamlit demo, HF Spaces, Medium article).

**Why:** A fully working, tested Phase 1 is worth more in an interview than
three phases each half-finished. Kubernetes manifests nobody ran, or an
Airflow DAG never actually triggered, undermine credibility more than not
having them at all.

---

## ADR-008: Direct HTTP download instead of the `soccerdata` library

**Decision:** `src/data/ingest.py` fetches CSVs directly via `requests`
(`https://www.football-data.co.uk/mmz4281/{season}/{league}.csv`) instead of
using the `soccerdata` library originally proposed in the plan.

**Why:** `soccerdata`'s `MatchHistory` reader uses a custom TLS client
(`tls_requests`) to mimic browser fingerprints. That client received HTTP 503
from football-data.co.uk in testing, while a plain `curl`/`requests` GET on
the same URL returned 200 with valid data. Beyond fixing the immediate
failure, dropping `soccerdata` removes a large, unnecessary dependency chain
(`selenium`, `seleniumbase`, `tls_requests`) for what is just a public,
unauthenticated CSV download — simpler and more reproducible.

**Note:** football-data.co.uk does not return HTTP errors for malformed
season/league codes in all cases (observed: an invalid season code returned
200 with unrelated historical data instead of 404). `ingest.py` therefore only
fetches and saves raw bytes; schema/date-range sanity checking happens in
`src/data/validate.py`, not during ingestion.

**Note:** Column schema differs across seasons (71 columns in 2010/11 vs. 132
in 2025/26 — more betting markets were added over time). Feature engineering
must only rely on columns present across all seasons in scope.

---

## ADR-009: No recalibration applied in this iteration (verified, not assumed)

**Decision:** None of the four Phase 1 models (bookmaker baseline, logistic
regression, random forest, XGBoost) use class reweighting, and none have
post-hoc recalibration (Platt scaling / isotonic regression) applied to their
output probabilities.

**Why:** Per ADR-004, recalibration is only needed when a model is trained
with reweighted classes. `src/models/calibration.py` was still built and
exercised, not left as an unused stub, so this is a measured finding rather
than an assumption: fitting isotonic/Platt calibrators on the first half of
Random Forest's out-of-fold seasons (2015–2019) and evaluating on the second
half (2020–2025) shows isotonic recalibration changes log-loss by only
-0.0002 (noise-level) and Platt scaling makes it *worse* by +0.0044. The
reliability diagrams for all three outcome classes (H/D/A) already show
predicted probabilities tracking actual frequencies closely without any
reweighting — see `vault/Evaluation-Log.md` Run 001.

**How to apply:** If a future iteration reweights classes (e.g. to chase
draw recall), rerun `src/models/calibration.py`'s benefit check before
deciding whether to ship the recalibrated probabilities — don't assume it
helps just because reweighting was used.
