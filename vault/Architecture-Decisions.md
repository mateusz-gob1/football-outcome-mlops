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

---

## ADR-010: MLflow "alias" instead of "stage" for the production model

**Decision:** `src/models/registry.py` promotes model versions using MLflow's
alias mechanism (`client.set_registered_model_alias(name, "production", version)`)
rather than the older stage transition API (`transition_model_version_stage`).
Serving code (Step 10) loads the model via `models:/football_outcome_predictor@production`.

**Why:** MLflow deprecated model stages in favor of aliases/tags starting
around 2.9; the installed version here (3.14) still exposes the old stage API
for backward compatibility, but aliases are the current recommended approach.
Functionally this achieves exactly what the plan calls "stage: Production" -
same promote/rollback semantics, current API.

**Note:** the registered model is loaded via `mlflow.sklearn.load_model(...)`,
not `mlflow.pyfunc.load_model(...)` - the pyfunc flavor's default `.predict()`
only returns the winning class label, not the full H/D/A probability
distribution the API needs. The sklearn flavor returns the real
`RandomForestClassifier`, exposing `.predict_proba()` directly.

---

## ADR-011: Docker setup written but not verified with an actual build

**Decision:** `Dockerfile`, `docker-compose.yml`, and `.dockerignore` exist and
are reasoned through carefully, but as of this iteration no `docker build` /
`docker compose up` has actually been run - neither this machine nor the
remote sandbox used to attempt verification had Docker installed.

**What was verified instead:** the full pipeline (ingest -> validate -> train
-> registry) was run from a fresh clone of the repo on a separate machine,
confirming the code works outside this one dev environment. The FastAPI app
was also run directly (uvicorn, no container) against that freshly-registered
model: `/health`, a valid `/predict` call, and the unknown-team 422 case all
passed.

**A real bug this process still caught without a build:** static review of
`docker-compose.yml` found that `api`'s `depends_on: [mlflow]` only waits for
the `mlflow` container to *start*, not for the `mlflow server` process inside
it to actually bind port 5000. Since the FastAPI `lifespan` in
`src/serving/api.py` connects to MLflow synchronously at startup with no
retry, `api` would very likely crash on boot in a real `docker compose up`.
Fixed by adding a `healthcheck` to the `mlflow` service and switching `api`'s
dependency to `condition: service_healthy`.

**Update (superseded by ADR-014):** GitHub Actions CI now runs a real
`docker compose build` + `up` + curl smoke test on every push (see ADR-014) -
this is no longer unverified, and it caught a real bug on the first run.

---

## ADR-014: Real Docker verification via CI caught a genuine startup bug

**Decision:** The `docker` job in `.github/workflows/ci.yml` is the actual
verification ADR-011 deferred to "someday" - it builds both images, runs
`docker compose up -d`, polls `/health`, and curls `/predict`, all on
GitHub's own runners (which have Docker; this dev machine doesn't).

**What it found on the very first real run:** the `api` container crashed on
startup with `mlflow.exceptions.MlflowException: ... 403 ... 'Invalid Host
header - possible DNS rebinding attack detected'`. MLflow's server validates
the incoming `Host` header against an allowlist (default: `localhost` and
private IP ranges) to block DNS-rebinding attacks. The `api` container
reaches MLflow via the Docker Compose service name (`http://mlflow:5000`),
so the `Host` header is literally the string `mlflow` - not `localhost`, not
an IP address - and got rejected.

**Fix (round 1):** added `--allowed-hosts mlflow,localhost` to the `mlflow`
service's startup command in `docker-compose.yml`.

**Fix (round 2 - the first fix wasn't enough):** the next CI run still
failed, earlier this time - `docker compose up -d` itself errored with
"container football-outcome-mlops-mlflow-1 is unhealthy". The mlflow
container's own logs showed why: `Rejected request with invalid Host header:
localhost:5000`. The allowlist match is against the Host header *verbatim,
including the port* - `mlflow`/`localhost` (no port) didn't cover the
container's own healthcheck, which calls `http://localhost:5000` and sends
`Host: localhost:5000`. Updated to
`--allowed-hosts mlflow,mlflow:5000,localhost,localhost:5000` to cover both
the bare and port-qualified forms used by the healthcheck and the `api`
container's real traffic.

**Why this matters beyond the one bug:** this is exactly the kind of failure
static review (ADR-011's earlier pass) could not have caught - it only shows
up when the two containers actually try to talk to each other over the
Compose network. It's the concrete argument for why "I wrote a Dockerfile"
and "CI actually builds and runs it on every push" are different claims, and
why the second one is worth having.

**Fix (round 3 - the real bug):** with both host-header issues fixed, `mlflow`
started healthy and `api` got further, but crashed with
`mlflow.exceptions.MlflowException: No such artifact: ''` while loading the
model. Root cause, confirmed by inspecting `mlflow.db` directly
(`sqlite3 mlflow.db "SELECT artifact_location FROM experiments"`): every run's
`artifact_uri` was an **absolute host filesystem path** -
`file:///D:/AI-Career/football-outcome-mlops/mlruns/...` locally, or
`file:///home/runner/work/.../mlruns/...` in CI - because `train.py` /
`registry.py` call `mlflow.set_experiment(...)` directly against
`sqlite:///mlflow.db`, with no `mlflow server` process involved at all. That
absolute path means nothing inside a container's isolated filesystem, so any
client without the *exact same* absolute path mounted at the *exact same*
location cannot resolve it - "no such artifact" is that failure.

Verified locally before touching CI (`mlflow server --backend-store-uri
sqlite:///... --artifacts-destination ./test_artifacts`, then logged/loaded a
throwaway sklearn model against `http://127.0.0.1:5001`): when a *real*
`mlflow server` process is involved, `--serve-artifacts` defaults to `True`,
and new experiments get an `artifact_uri` of `mlflow-artifacts:/...` instead
of `file:///...` - clients then fetch artifact bytes over HTTP through the
tracking server (proxied), never touching the filesystem directly. Confirmed
end-to-end: logged a model against the test server, loaded it back via
`models:/proxy_test_model/1`, worked.

**Real fix:** `.github/workflows/ci.yml`'s `test` job now starts an actual
`mlflow server --artifacts-destination ./mlruns` in the background (not a
bare `sqlite:///` connection) before running the pipeline, with
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000` exported for the pipeline and
test steps - so the experiment is created through the proxy from the start.
`docker-compose.yml`'s `mlflow` service now uses `--artifacts-destination
./mlruns` instead of `--default-artifact-root ./mlruns` (the latter, per
MLflow's own docs, "does not impact already-created experiments" - it would
have been a no-op fix).

**Why local dev (non-Docker) was never affected:** running `uvicorn
src.serving.api:app` directly on the same machine that registered the model
has no container boundary to cross - the absolute `file:///D:/...` path
resolves fine because it's the same filesystem. The bug is specific to
crossing a host/container or container/container boundary, which is exactly
why it only showed up once CI actually exercised Docker for real.

**Resolved:** after three rounds of fixes (Host header allowlist, port
qualification, artifact proxying), the `docker` job passed fully: build,
`compose up`, `/health`, `/predict`, and the GHCR image push. ADR-011's
"unverified" status is closed - this is now a real, repeatable check on
every push.

## ADR-012: Multi-bookmaker baseline (Bet365 + Bet&Win), with fallback

**Decision:** `odds_implied_*` features (and the bookmaker baseline model)
average Bet365's and Bet&Win's own overround-normalized implied probabilities
(a linear pool), instead of using Bet365 alone. When Bet&Win's odds are
missing for a match, fall back to Bet365 alone rather than dropping the row.

**Why:** Raised directly by Mateusz - if the model's strongest feature is one
bookmaker's own odds, and the model still can't beat that bookmaker, is the
comparison meaningful, or just "can you reproduce the input you were given"?
Answer: comparing against a single bookmaker actually understates the
baseline's real strength, because bookmaker consensus (averaging independent
odds-setters) is generally a *more* efficient estimate than any one
bookmaker - so it's a fairer, harder baseline to be measured against, not a
weaker one. football-data.co.uk has odds from several bookmakers (B365, BW,
WH, IW, VC, PS, plus its own Max/Avg aggregate columns), but only Bet365 and
Bet&Win have (near-)complete coverage across all 16 seasons in scope - the
rest are missing entirely for either the earliest 1-2 seasons or the most
recent 1-2 (see the season-by-season bookmaker coverage table this ADR is
based on, checked directly against the raw CSVs).

**The fallback matters because of a real data gap:** Bet&Win is missing for
141 of 380 matches (37%) in the 2024/25 season specifically (likely delisted
by football-data.co.uk for part of that season, since it's fully present
again in 2025/26). Requiring both bookmakers strictly would have dropped
those 141 matches outright. Falling back to Bet365 alone for just those rows
keeps every match in the dataset, at the cost of those specific rows using a
single-bookmaker signal instead of the two-bookmaker average.

**Not pursued:** using football-data.co.uk's own `Max`/`Avg` columns (already
computed across many bookmakers) as an even stronger consensus baseline -
these only exist from 2019/20 onward (7 of 16 seasons), which would cut the
walk-forward validation window roughly in half. Left as a possible follow-up
analysis on that shorter window, not a replacement for the main result.

**Result:** rerun in `vault/Evaluation-Log.md` Run 002.

---

## ADR-013: Manually forced a model promotion despite the automatic gate saying no

**Decision:** After retraining Random Forest on the new B365+BW odds feature
(ADR-012), `register_and_promote()` correctly refused to promote the new
version (v6, log-loss 0.9685) because it scored marginally worse than the
already-promoted version (v?, log-loss 0.9683 - trained on the old
Bet365-only odds feature). The `production` alias was moved to v6 anyway, via
a direct `MlflowClient.set_registered_model_alias` call, bypassing the gate.

**Why this override was correct, not a violation of the promote-if-better
rule:** the automatic gate compares *model quality* (log-loss), which is the
right check when the feature *definition* hasn't changed. Here it had - the
old promoted version was fit on `odds_implied_*` meaning "Bet365 alone";
`src/features/build_features.py` now computes `odds_implied_*` as the
Bet365+Bet&Win average. Any live `/predict` call recomputes features with
the *current* pipeline, so keeping the old model promoted would silently
have fed it inputs from a different distribution than it was trained on -
real train/serve skew, not a quality trade-off the log-loss gate can see.

**How to apply generally:** the promote-if-better check in
`src/models/registry.py` / the Phase 2 retraining pipeline is only a valid
gate between versions of the *same* feature schema. A feature-schema change
requires promoting the new version manually (or extending the gate to also
compare a feature-schema version/hash and force promotion on mismatch,
without a manual step) - not always accept the old model just because its
recorded log-loss looks marginally better.
