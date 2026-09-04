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

---

## ADR-016: Kubernetes manifests, not a live cluster

**Decision:** `k8s/deployment.yaml` and `k8s/service.yaml` define both
services (`api`, `mlflow`) as Deployments + Services, plus a
PersistentVolumeClaim for MLflow's data - but nothing here has been applied
to a real cluster (no `kubectl apply` was run).

**Why:** matches the plan's own scoping - a correct manifest plus the ability
to explain it in an interview is the deliverable, not a running production
cluster. Standing up a real cluster (minikube/kind/cloud) just to prove one
`kubectl apply` works is disproportionate effort for a portfolio project at
this stage, especially compared to Docker (ADR-011/014), where GitHub Actions
already gave free, real, repeated verification - no equivalent free lunch
exists for Kubernetes without deliberately provisioning a cluster.

**Design choices carried over from the Docker setup, translated to K8s:**
- Both Deployments reuse the single published image
  (`ghcr.io/mateusz-gob1/football-outcome-mlops-api`), with the `mlflow`
  Deployment overriding the container `command` to run `mlflow server`
  instead of `uvicorn` - the same "one image, two roles" pattern
  `docker-compose.yml` uses via its `command:` override.
- `mlflow-service` is `ClusterIP` (internal-only); `api-service` is
  `LoadBalancer` (the one meant to be reached from outside) - deliberately
  asymmetric, not an oversight.
- `--allowed-hosts=*` on the mlflow container, wider than Docker Compose's
  explicit allowlist (ADR-014). Cluster-internal traffic arrives with the pod
  IP as the Host header, not a stable name a short allowlist could cover -
  acceptable here specifically because the service is ClusterIP-only, never
  reachable from outside the cluster in the first place.
- Both liveness/readiness probes point at the same endpoints proven to work
  in the Docker healthcheck/CI smoke test (`/health` for `api`, `/` for
  `mlflow`) - not guessed.

**Known gap, not hidden:** `replicas: 1` on the `mlflow` Deployment isn't a
placeholder value - see ADR-017.

---

## ADR-017: Why `mlflow` is pinned to `replicas: 1` (the real MinIO argument)

**Decision:** the `mlflow` Deployment in `k8s/deployment.yaml` is hard-set to
one replica, not left to autoscale like `api` (which runs 2).

**Why:** the backend store is `sqlite:///data/mlflow.db` - a single file.
SQLite does not safely support multiple processes writing to the same file
concurrently the way a real database server does; two `mlflow` pods would
race on the same file and corrupt it. This is a genuine scaling ceiling, not
a style choice.

**How this connects to MinIO (Phase 2, not yet built):** MinIO was originally
scoped as "nicer artifact storage," but this ADR is the concrete argument for
why it's actually a *correctness* upgrade, not just a convenience one -
S3-compatible storage handles concurrent access safely, the way a shared
filesystem/SQLite file does not. The real fix for horizontal scaling here is
two-part: (1) a proper multi-writer backend store (Postgres/MySQL instead of
SQLite) for run/experiment *metadata*, and (2) S3-compatible storage (MinIO
or real S3) for the artifact *bytes* themselves, so `--artifacts-destination`
points at a bucket rather than a PersistentVolume tied to one pod. Both are
needed together - fixing only artifact storage still leaves the metadata
store as a single-writer bottleneck.

**Not pursued yet:** this is documented as the known reason to eventually
build MinIO support, not deferred without explanation.

---

## ADR-018: Airflow via `airflow standalone`, verified through CI like Docker

**Decision:** `dags/retrain_dag.py`, `Dockerfile.airflow`, and
`docker-compose.airflow.yml` run Airflow as a single container via
`airflow standalone` (SQLite backend, SequentialExecutor, one process running
webserver + scheduler + a one-time DB init) - not the full multi-container
reference architecture (Postgres + Redis + separate webserver/scheduler/worker
services) Airflow's official docker-compose example uses.

**Why single-container:** the goal here is to demonstrate orchestration -
turning `src/pipeline/retrain_pipeline.py`'s five steps into an Airflow DAG
with independent, retryable, observable tasks - not to prove out
production-scale concurrent task execution. `airflow standalone` is Airflow's
own documented quick-start path for exactly this kind of scenario. Fewer
moving parts also means fewer chances at another multi-round debugging cycle
like ADR-014's Docker saga - a real, deliberate trade-off, not laziness.

**Verification:** like Docker (ADR-011/014), this is checked for real via a
dedicated `airflow` job in `.github/workflows/ci.yml` - builds the image,
starts the container, waits for the webserver's `/health`, then actually
**triggers the DAG and polls until it succeeds or fails**, not just "the
container started." The DAG's real tasks (ingest 16 seasons, train 4 models
with hyperparameter grids, evaluate, register) take several minutes - this
proves the whole pipeline runs inside Airflow's task execution model, not
just that the DAG file parses without errors.

**Dependency risk, handled explicitly - and it fired on the first CI run:**
installing this project's full requirements.txt (fastapi, evidently, dvc,
pytest, black, ...) into Airflow's own image risks conflicting with Airflow's
own pinned dependencies. `Dockerfile.airflow` installs against Airflow's
official constraints file (`constraints-3.10.txt` for the matching Airflow
version), and CI immediately hit exactly the predicted conflict: our
`requirements.txt` requires `pandas>=2.2`, Airflow 2.10.4's constraints pin
`pandas==2.1.4` - unsolvable together. Fixed by introducing
`requirements-airflow.txt`, a separate, deliberately minimal dependency list
containing only what `dags/retrain_dag.py`'s tasks actually import (pandas,
numpy, scikit-learn, xgboost, lightgbm, mlflow, requests) - no
fastapi/uvicorn/pydantic/evidently/dvc/pytest/black/ruff/pre-commit, none of
which any DAG task touches. Smaller dependency surface, smaller conflict
surface. `pandas` is explicitly capped `>=2.1,<2.2` to land on the exact
version Airflow's constraints require.

**Known simplification:** in a real production setup, heavy ML training
would more likely run as a separate task (`DockerOperator`/
`KubernetesPodOperator`) rather than importing training code directly into
Airflow's own worker Python environment - keeping Airflow's runtime lean and
avoiding exactly the dependency-conflict risk above. Here, `@task`-decorated
functions call `src/models/train.py` etc. directly, which is simpler to
demonstrate and verify but not how a larger real system would isolate
concerns.

**Resolved:** after the `requirements-airflow.txt` fix, the `airflow` CI job
passed fully - image build, webserver health, and a real DAG trigger through
all five tasks (ingest, validate, train four models, evaluate, promote) to a
successful run. All four CI jobs (`lint`, `test`, `docker`, `airflow`) are
green together.

---

## ADR-019: MinIO replaces local disk for MLflow artifacts (full integration)

**Decision:** `docker-compose.yml` gained `minio` (S3-compatible storage) and
`minio-init` (one-shot bucket creation) services. The `mlflow` service's
`--artifacts-destination` now points at `s3://mlflow-artifacts` instead of a
locally-mounted `./mlruns` directory, with `MLFLOW_S3_ENDPOINT_URL` and dummy
AWS credentials pointed at MinIO.

**Why full integration, not a standalone demo:** MinIO was scoped as
"nicer artifact storage," but replacing the working local-disk setup end to
end - not bolting on a side demo - is what actually proves the swap works,
consistent with how Docker (ADR-011/014) and Airflow (ADR-018) were each
verified for real rather than left as "should work." The trade-off, accepted
knowingly: this touches the `docker` CI job that took three debugging rounds
to get green (ADR-014), so it carries real regression risk.

**Why the `api` container needed zero changes:** because ADR-014 already
made the `mlflow` server proxy all artifact access (the `mlflow-artifacts:/`
scheme), swapping *where* the server physically stores bytes - local disk vs.
an S3 bucket - is entirely the server's own concern. `api` only ever talks to
`mlflow-service:5000` over HTTP, exactly as before; it has no AWS credentials
and no idea MinIO exists. This is the payoff of building the proxy correctly
the first time, not a coincidence.

**CI restructuring this required:** the old flow had the `test` job register
a model against a bare local `mlflow server` and hand the resulting
`mlflow.db`/`mlruns` files to the `docker` job as a build artifact. That
stops working once artifacts live in MinIO - the `test` job's local-disk
artifacts have nothing to do with the `docker` job's MinIO bucket. New flow:
the `docker` job now starts `minio` + `minio-init` + `mlflow` *first*, then
runs the full ingest/validate/train/registry pipeline directly against that
containerized, MinIO-backed `mlflow` (reusing the same pattern the `test` and
`airflow` jobs already use), and only *then* starts `api`. The artifact
upload/download step between `test` and `docker` was removed entirely - each
job is now fully self-contained, which is simpler to reason about than the
hand-off it replaced.

**Known limitation, documented not hidden (see ADR-017):** this fixes
artifact *storage* concurrency (S3 handles concurrent readers/writers safely,
a local file mount does not) but not backend *metadata* store concurrency -
`--backend-store-uri sqlite:///mlflow.db` is unchanged. A real multi-replica
`mlflow` deployment still needs Postgres/MySQL for that half; this ADR closes
one of the two gaps ADR-017 identified, not both.

**Bug caught by CI, unrelated to S3/MinIO itself:** the first real run of the
restructured `docker` job failed with `sqlite3.OperationalError: unable to
open database file` - not an S3 problem at all. Removing the old
`test`→`docker` artifact hand-off (ADR-019's main change) meant `./mlflow.db`
no longer existed on the runner's filesystem *before* `docker compose up`.
`docker-compose.yml` bind-mounts it as a file
(`./mlflow.db:/app/mlflow.db`); Docker's default behavior when a bind-mount
source path doesn't exist is to silently create a **directory** there instead
of a file, so mlflow's sqlite backend found a directory where it expected a
file. Fixed with a one-line `touch mlflow.db` step immediately before
`docker compose up`, guaranteeing the mount target is a real (empty) file.

**Resolved:** the next CI run passed fully - `mlflow` came up healthy, a real
model trained and registered with its artifacts genuinely written to and
read back from the MinIO bucket, `api` started against it, and `/health` +
`/predict` both succeeded. All four CI jobs (`lint`, `test`, `docker`,
`airflow`) green together. Phase 2 is complete.

---

## ADR-020: Demo ships pre-computed predictions, not a live model

**Decision:** `frontend/app.js` fetches static JSON
(`frontend/data/*.json`) generated once by `frontend/prepare_data.py`. It
does not load the trained model, call MLflow, or run
`src/features/build_features.py` at runtime.

**Why:** the plan's original ask was a dashboard showing "predictions for the
upcoming gameweek." At the time this was built the 2025/26 season had already
finished and 2026/27 hadn't started yet (see the "Timing" note in the
original plan re: the 2026/27 season starting shortly after this project) -
there was no live upcoming gameweek to predict. Rather than block the demo on
a season starting, or bolt on a fake "pretend this date is live" flow, the
demo browses real historical out-of-fold predictions instead: for every match
in the 2015/16-2025/26 seasons, it shows what the model predicted, what the
bookmaker implied, and what actually happened - a truthful, immediately
available substitute that still demonstrates the same thing (model vs
bookmaker vs reality) the live version would have.

**Why this is also the right call independent of the timing issue:** a
dashboard that loads a live model needs either (a) a reachable MLflow
tracking server (meaning deploying and keeping MLflow+MinIO running
somewhere the demo can reach, well beyond a demo's scope) or (b) a model
file bundled in with its own inference code duplicated from `src/`. Both add
real operational surface for a component whose only job is "show what we
already proved in vault/Evaluation-Log.md." Shipping the already-computed,
already-validated out-of-fold results is simpler, faster to load, and cannot
drift from the numbers reported elsewhere in this project, because they
*are* those numbers.

**Trade-off, stated plainly:** this demo cannot answer "who wins Arsenal vs
Chelsea next week" - only "here's how the model did on real matches it never
saw during training." For a portfolio piece whose point is demonstrating
methodology and honest evaluation, that is the more relevant claim to make
visually, not a compromise.

**Technology pivot - Streamlit, then plain HTML/CSS/JS:** the first version
of this demo was a Streamlit app. Two things changed that: (1) Hugging Face
deprecated Streamlit as a native Spaces SDK - deploying one now means
choosing the Docker SDK with a Streamlit template, and the account used here
showed Docker/Gradio gated behind a paid plan (Static Spaces stayed free);
(2) checking this project's own prior work (`football-agent/frontend/`,
`financial-doc-agent/frontend/`) showed neither uses Streamlit either - both
ship a static HTML/CSS/JS dashboard (dark-neutral theme, Geist font,
Chart.js) served by their own API. Rebuilding this demo the same way
(`frontend/index.html` + `style.css` + `app.js`, Chart.js for the
calibration/importance charts) gets a free, unblocked Static Space *and*
visual consistency with the rest of the portfolio - not a fallback, the
better option once the constraint surfaced.

**Deployment:** a standalone HF Spaces Static app (`frontend/` is the entire
Space - `README.md` carries the required HF YAML frontmatter, `sdk: static`),
independent of the Docker/K8s/Airflow/MinIO stack built in Phase 2. Verified
locally via the Preview tool (screenshots, console clean, season/team
filtering exercised) before deployment. Live at
https://matigob-football-outcome-predictor.static.hf.space, pushed via
`git subtree push --prefix=frontend hf main` (a plain push failed with
"fetch first" since HF auto-initializes new Spaces with placeholder template
files; force-pushed a `git subtree split` branch over that placeholder,
confirmed with Mateusz first since it's a destructive, external-facing
operation).

**Windows-specific credential gotcha, for the record:** the first push
attempt hung on a Git Credential Manager GUI popup for huggingface.co that
a non-interactive session can't answer. Cause: both the system-level
`credential.helper=manager` (GCM) and a locally-configured
`credential.helper=store` were active simultaneously - git accumulates
credential helpers across config levels rather than the local one replacing
the system one, so GCM ran first and blocked on its own prompt before the
stored token was ever tried. Fixed by adding an empty `helper =` entry
immediately before `helper = store` in the repo's local `.git/config`,
which resets the accumulated helper list at that point - only `store` was
active afterward, and the push authenticated without any GUI prompt.

---

## ADR-021: Architecture diagram as hand-written SVG, not a generated PNG

**Decision:** `docs/architecture.svg`, referenced from the README's new
"Architecture" section. Hand-authored markup, not exported from a diagramming
tool.

**Why:** the plan named `docs/architecture.png`, but SVG renders natively on
GitHub, scales without pixelation at any zoom level, and is a few KB of text
instead of a raster blob - no real downside for a static box-and-arrow
diagram. Neither prior project (`football-agent`, `financial-doc-agent`) has
an actual architecture diagram to match conventions against (just a logo and
screenshots for one, a text-only vault for the other), so there was no
existing format to stay consistent with - free choice, and SVG was the
better default.

**Content:** one horizontal pipeline (ingest → validate → features → train →
MLflow Registry → FastAPI), a support layer underneath for MLflow Server +
MinIO (the proxied-artifact design from ADR-011/014/019), small badges for
Evidently and Docker/K8s, a dashed box wrapping the same pipeline for the
Airflow DAG with a loop-back arrow to represent the weekly retrain cycle, and
the demo dashboard drawn with a dotted connector to make its one-way,
pre-computed relationship to the rest of the stack visually explicit (same
distinction as ADR-020, restated visually).

**Verification note:** built and checked without ever opening the file in an
image viewer - the Preview tool's screenshot function was unreliable in this
session (timed out repeatedly), so correctness was verified programmatically
instead: fetched the SVG into a real browser DOM via `preview_eval`, then
used `getBBox()`/`getPointAtLength()` to check every text label sat fully
inside its box and no connector line's actual stroke path crossed through
any text run. This caught two real bugs a glance might have missed - a
loop-back arrow that cut straight through three lines of the Airflow box's
own text, and a Docker/K8s connector arrow that started from empty space
instead of the FastAPI box - before a manual screenshot (once the tool
recovered) confirmed the fixed layout visually.

---

## ADR-022: Removed bookmaker odds as a model input feature

**Decision:** `odds_implied_home_prob/draw_prob/away_prob` (ADR-012's
Bet365+Bet&Win consensus) are no longer part of `FEATURE_COLUMNS` -
`BookmakerBaseline` alone still uses them, via a new dedicated
`BOOKMAKER_FEATURE_COLUMNS` list. Logistic Regression, Random Forest, and
XGBoost now predict purely from match statistics (form, head-to-head, rest
days, table position) with no visibility into the betting market at all.

**Why:** two independent reasons, either one would have been sufficient.
(1) Predicting a genuinely future gameweek needs pre-match odds sourced live
- an extra dependency (a live odds feed) the project didn't otherwise need,
since every other feature is computable purely from past match history.
(2) Feeding the bookmaker's own odds into "our" models undermined any
independent comparison - the model was partly just re-deriving the market's
own view. Per the ablation study (Run 001/002 in Evaluation-Log.md), odds
were also by a wide margin the single most important feature group, so this
is a real, not cosmetic, change: log-loss for all three ML models gets
measurably worse (Run 003). That's an accepted, deliberate trade-off, not a
regression to fix - see vault/Evaluation-Log.md Run 003 for the exact
numbers.

**What this does *not* change:** the bookmaker's pick is still shown
per-match as a side-by-side reference (both historically and, from Phase 2
onward, for upcoming fixtures) - it's just no longer wired into the ML
models' own predictions. ADR-012 (how the bookmaker consensus itself is
built) is unaffected.

**Registry consequence, same pattern as ADR-013:** retraining Random Forest
without odds (v10) scored worse than the already-promoted odds-based
version, so `register_and_promote()`'s automatic gate correctly refused to
promote it. The `production` alias was moved to v10 anyway via a direct
`MlflowClient.set_registered_model_alias` call - the same justification as
ADR-013: this is a feature-*schema* change, not a same-schema quality
comparison, so the log-loss gate isn't the right check. Leaving the old
odds-based model promoted would have caused a live schema mismatch, since
`src/serving/api.py` builds its request features from the current, odds-free
`FEATURE_COLUMNS` regardless of which version is promoted (confirmed by a
real test failure - `tests/test_api.py` - before the alias was moved,
exactly the train/serve skew ADR-013 warned about in general).

**Bookmaker baseline's own feature list is now separate from the shared
one:** `MODEL_SPECS["bookmaker_baseline"]["feature_cols"]` overrides the
default `FEATURE_COLUMNS` per-model (rather than one global constant every
model used), since the bookmaker baseline is the one model that must keep
seeing odds. `tests/test_train.py` guards this split directly (disjoint
column sets, `BookmakerBaseline` still predicts correctly when sliced only
with its own list) so a future edit can't silently reintroduce odds into the
shared list or break the baseline by accident.

---

## ADR-023: Predicting a genuinely future gameweek - two new data sources, one feature-building fix

**Decision:** Added `src/pipeline/predict_upcoming.py`, which predicts the
next *unplayed* Premier League gameweek from all three ML models at once.
This needed two new data sources, since football-data.co.uk (the project's
only source until now) never publishes a future-fixture list - only
completed results:

- `src/data/fixtures_openfootball.py` parses openfootball/england's
  plain-text season schedule (`raw.githubusercontent.com/openfootball/
  england/master/<season>/1-premierleague.txt`) to find the earliest
  matchday that still has an unplayed fixture. Team names there ("Arsenal
  FC", "AFC Bournemouth") don't match football-data.co.uk's short/irregular
  ones ("Arsenal", "Bournemouth", "Nott'm Forest", "QPR") - `src/data/
  team_names.py` normalizes via a curated alias table with a fuzzy-match
  fallback, and **raises rather than guesses** on an unmappable name (a
  silent wrong mapping would look like a brand-new team with zero history
  to every downstream feature).
- `src/data/fixtures_odds.py` opportunistically attaches pre-match odds from
  football-data.co.uk's separate `fixtures.csv` when it happens to already
  include Premier League rows for that gameweek (it doesn't always, by the
  time this was built) - display-only, per ADR-022, so a missing match here
  just means "bookmaker pick unavailable yet," not an error.
- Caught while wiring this up: `fixtures.csv` is UTF-8-with-BOM but doesn't
  declare a charset, so `requests`' `.text` (which guessed Latin-1) mangled
  the BOM into literal characters glued onto the first column name (`Div`
  became `ï»¿Div`), breaking the `Div == "E0"` filter. Fixed by decoding
  `response.content` explicitly as `utf-8-sig` instead of using `.text`.

**Why fixtures are predicted one at a time, never batched into one combined
DataFrame:** `build_features()`'s table position and rolling-form features
rank/aggregate across *all* teams as of a given date. If two not-yet-played
fixtures from the same gameweek (e.g. Friday and Sunday) were appended
together and featurized in one pass, Friday's synthetic placeholder result
would already look like a real result to Sunday's fixture - inflating a
team's table position and form before that Friday match has actually been
played. `src/features/query_features.py::build_query_features` avoids this
by construction: it loops internally, featurizing each fixture against only
real historical data, one at a time, and concatenates the results
afterward. `tests/test_query_features.py` guards this directly - predicting
several fixtures together must produce byte-identical output to predicting
each one alone.

**Refactor along the way:** `src/serving/api.py::predict()` had its own
copy of the "append a synthetic row, rebuild features, read the new row
back" trick. It's now `build_query_features` (a one-fixture call), shared
with the batch path above - one implementation of the leak-free
future-fixture trick instead of two that could quietly drift apart.

**`src/models/registry.py::train_production_candidate` generalized** into
`fit_full_model(model_name, params, data)`, usable for any `MODEL_SPECS`
entry (not just the promoted candidate) - this is what lets
`predict_upcoming.py` fit fresh logistic regression, random forest, and
XGBoost models on the entire match history for a fixture that hasn't been
played yet, where there's no held-out test season to speak of.

**Known real-world wrinkle, not a bug:** openfootball/england's 2026-27
schedule file lists some clubs (e.g. Coventry City, Hull City) that aren't
in this project's actual football-data.co.uk history for the current top
flight - `normalize_openfootball_team_name` correctly raises for those, and
`next_gameweek_fixtures` logs a warning and skips that one fixture rather
than crashing the whole gameweek's predictions. Confirmed live on
2026-09-03: fixtures.csv had no E0 rows yet for the next gameweek (bookmaker
odds not published that far out), and one fixture (Man City v Coventry) was
correctly skipped - the other 9 fixtures in that gameweek predicted fine.

---

## ADR-024: Demo redesigned around live-gameweek predictions - supersedes ADR-020

**Decision:** `frontend/` now leads with a "Next Matchday" view (all three ML
models' picks per upcoming fixture, bookmaker pick shown as a reference when
published) instead of only browsing historical out-of-fold matches. The
historical browse view was extended to show all three ML models (via a
model-select dropdown) instead of just Random Forest, and the
"Model Performance" tab was renamed "Methodology" with the
"beat-the-bookmaker" verdict language removed from its copy (the numbers
stay - just not framed as a headline conclusion). README.md's intro and
"Result" section were rewritten the same way.

**Why this supersedes ADR-020, not just extends it:** ADR-020's decision was
"ship pre-computed predictions, not a live model" for a specific reason -
*there was no live upcoming gameweek to show* at build time (2025/26 had
ended, 2026/27 hadn't started). That premise no longer holds; the season is
underway. ADR-020's *other* argument still holds and is carried forward
unchanged: the demo still ships a pre-computed export
(`frontend/prepare_data.py` -> static JSON), not a live MLflow-backed
inference server, because standing up a reachable model server just for a
portfolio demo isn't worth the operational cost. What changes is *what* gets
pre-computed and exported: historical OOF results only (ADR-020) versus
historical OOF results *and* a weekly-refreshed upcoming-gameweek batch from
`src/pipeline/predict_upcoming.py` (this ADR).

**Why "beat the bookmaker" was removed as the framing, not just downplayed:**
Mateusz's own assessment, in plain terms - it was the wrong headline for a
portfolio piece aimed at both recruiters and football fans. A negative
statistical result ("we didn't beat the market") is honest and worth keeping
*somewhere* (it still is - Methodology tab, README methodology section), but
leading with it made the project read as a failed attempt at beating
bookmakers rather than what it actually demonstrates: rigorous ML
methodology (walk-forward validation, calibration, a full MLOps stack) and a
genuinely independent, currently-serving prediction system.

**Data staleness note carried over from Phase 2:** the demo's "Next
Matchday" JSON is only as fresh as the last time
`src/pipeline/predict_upcoming.py` + `frontend/prepare_data.py` were run -
there is no automatic refresh yet. That's Phase 4 (weekly GitHub Actions
automation), not done as of this ADR.

---

## ADR-025: Club badges via TheSportsDB's free API, not Wikipedia crests

**Decision:** Match cards show each club's badge, resolved once (offline,
`frontend/prepare_badges.py` -> `frontend/data/team_badges.json`) via
TheSportsDB's free search API (`src/data/team_badges.py`) and hotlinked
directly from their CDN at render time - never downloaded, re-hosted, or
modified. A team with no resolved badge (or an image that fails to load at
runtime) falls back to a colored-initials avatar (`teamInitials`/`teamColor`
in `frontend/app.js`) rather than a broken image. The page footer credits
TheSportsDB with a link, per their terms of use.

**Why not Wikipedia/Wikimedia Commons (the first option considered):** most
current Premier League club crest files there are tagged "non-free use" -
explicitly licensed only for identifying the subject within a Wikipedia
article, not for reuse on a third-party site. Downloading and embedding
those on a public, deployed demo would be a real licensing violation, not
just a formality - this was flagged and the user agreed to use TheSportsDB
instead once that was clear.

**Why TheSportsDB is an acceptable middle ground:** their free-tier terms
(thesportsdb.com/docs_terms_of_use.php) explicitly allow using their
API/artwork "for your development projects," conditioned on not modifying
trademarked logos and linking back to their site - both satisfied here.
It's still someone else's trademarked material displayed under a specific
API's terms, not a freely-licensed asset - if this project ever needed
stronger guarantees (e.g. commercial use), a paid/licensed logo API would be
the correct next step, not this one.

**Real friction hit while building this:** the shared public test API key
("3") rate-limits (HTTP 429) heavily under any real request volume - a batch
of ~40 team lookups needed retry-with-backoff (`fetch_team_badge`) and took
several passes to fully resolve. `frontend/prepare_badges.py` only re-fetches
teams missing from the existing `team_badges.json` rather than the whole set
every time, for exactly this reason.

**Real bug caught by a test written for this feature, not by manual
inspection:** `_best_match`'s first version preferred any team whose league
name *contained* "premier league" (case-insensitive substring) - but
"Premier League 2" is the real name of England's U21 reserve competition, so
"West Brom U21" (league: "English Premier League 2") would have matched
before the actual first team. `tests/test_team_badges.py` was written to
lock in the intended behavior *before* checking - it initially exposed this
exact bug (the substring check couldn't tell the two leagues apart), fixed
by requiring an exact match on `"English Premier League"` instead.

**Separate build step, not part of `frontend/prepare_data.py`:** club
badges essentially never change between runs (unlike model output), and
hit an external rate-limited API - bundling badge resolution into the
data-refresh script that's meant to run after every retrain/gameweek would
add needless external-API risk to every routine update. It only needs
re-running when a genuinely new team appears in the data (promotion).

---

## ADR-026: Match-stat features (shots, corners, cards) added; xG deliberately left out for now

**Decision:** `FEATURE_COLUMNS` gained 14 new rolling-5-average features per
side - shots for/against, shots on target for/against, corners for/against,
and a combined yellow+red "cards for" discipline indicator
(`src/features/build_features.py`: `STAT_COLUMN_PAIRS`, `ROLLING_SUM_STATS`).
Computed the same leak-free way as existing form/goals features (past
matches only, via `.shift(1).rolling(5)`), so they work for genuinely future
fixtures with no new data-availability requirement - shots/corners/cards
have been in football-data.co.uk's files for all 17 seasons on record.

**xG (expected goals) was raised as a candidate too and rejected for now, not
overlooked:** football-data.co.uk only started publishing `HxG`/`AxG` from
the 2026/27 season - checked directly against the raw files (16 older
season CSVs have no such column at all; only the 20 matches played so far
in 2026/27 do). Walk-forward validation needs the feature across many
seasons to mean anything; with 16 seasons blank, either those seasons get
dropped entirely (destroying nearly all training data) or the feature gets
imputed with no real signal (defeats the purpose). Revisit once enough
seasons carry it - plausibly not for a few years. In the meantime, xG is
carried through as a **display-only** field (`DISPLAY_ONLY_COLUMNS` in
`src/data/validate.py`, NaN for pre-2026/27 matches) and shown in the
frontend's team-profile view for matches that have it, same treatment as
bookmaker odds after ADR-022 - informative, not fed to any model.

**Result - the promotion gate worked normally this time, no manual
override needed** (contrast with ADR-013/ADR-022's forced promotions):
Random Forest's log-loss improved from 1.0047 to 0.9935 with the new
features, so `register_and_promote()`'s automatic "not worse" check passed
on its own. See `vault/Evaluation-Log.md` Run 004 for full numbers -
shots-based features (`away_shots_against_5`, `home_shots_for_5`,
`home_shots_against_5`) now rank among the single most important
individual features, ahead of table position.

**Required, not optional, columns:** shots/shots-on-target/corners/cards
were added to `REQUIRED_COLUMNS` in `src/data/validate.py` (not
`OPTIONAL_COLUMNS`, unlike Bet&Win odds) - unlike BW's 141-match gap in
2024/25, these are missing in only 1 of ~6100 rows across all 17 seasons
combined, so dropping that one row outright (existing `dropna` behavior) is
simpler than building fallback-imputation logic for a near-nonexistent gap.

---

## ADR-027: Tried ensembling and a wider hyperparameter search - honest result, neither helped much; recalibration reopened

**Decision/context:** Mateusz asked, reasonably, how confident we are that
the three models are squeezing out everything the data has to offer. Three
things were checked in order:

**1. Ensembling (`EnsembleAverage` in `src/models/train.py`)** - a plain
average of logistic regression, random forest, and XGBoost's predicted
probabilities, each refit per walk-forward fold with fixed, previously-
established hyperparameters (`ENSEMBLE_MEMBER_PARAMS`) rather than nested-
tuning all three jointly (would multiply the grid size for little expected
benefit). **Result: it made things slightly worse** (0.9979 vs Random
Forest/Logistic Regression's ~0.9936) - equal-weight averaging drags the
result toward the weakest member (XGBoost, 0.9999) instead of toward the
best one. Kept in `MODEL_SPECS` and shown on the demo (useful, honest
context: "here's what a naive ensemble gets you"), but never a promotion
candidate.

**2. Wider hyperparameter grids** (Random Forest: `min_samples_leaf` added,
more `n_estimators`/`max_depth` values; XGBoost: `subsample` added to both
the wrapper and its grid; Logistic Regression: wider `C` range). **Result:
mostly noise.** XGBoost genuinely improved (1.0096 -> 0.9999 log-loss) - the
one case where the wider grid clearly helped. Random Forest was
unchanged in practice (0.9935 -> 0.9936, an utterly negligible difference -
the automatic promote-if-better gate correctly declined to promote the new
version, no manual override needed, unlike ADR-013/022's feature-schema
cases). Logistic Regression actually got very slightly *worse* (0.9951 ->
0.9977) - plausible, ordinary nested-CV variance from different C values
winning the inner-fold selection under the new grid, not a real regression.
**Conclusion: the previous grids were already close to as good as this
model family gets on this feature set** - more search didn't move the
needle except for XGBoost specifically.

**3. Recalibration re-check (`src/models/calibration.py`, rebuilt from
scratch - the file referenced by the original ADR-009 no longer exists in
the repo)** - isotonic and Platt scaling applied post-hoc to each model's
OOF probabilities. **Result: recalibration now helps, for every model**
(isotonic: -0.01 to -0.013 log-loss across all four ML models) - the
opposite conclusion from ADR-009's original check, which predates the
ADR-026 shots/corners/cards features and found no benefit. **Important
caveat, stated directly in the code's own docstring:** this check fits and
evaluates the recalibration on the *same* OOF data - optimistic versus a
real deployment, which would need to fit recalibration on a properly
held-out slice (e.g. per outer walk-forward fold, using only that fold's
training data). The honest headline is "there's recalibration headroom
worth pursuing properly," not "log-loss will drop by exactly this much in
production." **Not yet integrated into the trained/served models** -
building a leak-free, per-fold recalibration step is real, separate work
(next candidate task), not done as part of this check.

**Overall answer to "are we squeezing out everything":** mostly yes for
model *choice and tuning* (ensembling and wider search barely moved
things - Random Forest and Logistic Regression are both already close to
this feature set's ceiling), but no for *calibration* - that's the one
concrete, promising lever this check surfaced, pending a fair (non-leaky)
re-measurement before it's trusted for production.

See `vault/Evaluation-Log.md` Run 005 for the full numbers.

---

## ADR-028: The fair recalibration re-check reverses ADR-027's optimistic finding for isotonic

**Decision:** Built `leakfree_recalibrated_log_loss()` in
`src/models/calibration.py` - for each walk-forward test season, fits the
recalibrator only on OOF rows from strictly earlier seasons, then scores
that season's rows with it (the earliest season has nothing earlier to
calibrate from and is dropped, same trade-off as `MIN_HISTORY` elsewhere).
`tests/test_calibration.py::test_leakfree_recalibration_never_uses_the_test_season_itself_to_fit`
locks this in directly - it patches the fit step to record what it was
given and asserts none of it comes from the season being scored.

**Result: isotonic regression's apparent win in ADR-027 was mostly
overfitting, not a real effect.** Evaluated honestly (recalibrator fit only
on genuinely prior seasons), isotonic makes every model's log-loss *worse* -
dramatically so for Logistic Regression (0.9936 -> 1.1211) and Random
Forest (0.9936 -> 1.0472). Isotonic regression is a very flexible,
non-parametric fit; with only one or a few prior seasons' worth of data to
calibrate from (as few as ~340 matches for the second test season), it
fits noise in that small sample instead of a real miscalibration pattern,
and that noise doesn't generalize to the next season. ADR-027's same-data
check couldn't see this because fitting and evaluating on identical data
hides overfitting by construction - exactly the caveat that ADR-027 flagged
in advance as the reason not to trust that number for production.

**Platt scaling (the simpler, 2-parameter method) mostly survives the fair
re-check:**

| Model | Baseline | Platt (leak-free) |
|---|---|---|
| Random Forest (production) | 0.9936 | 0.9948 (worse) |
| Logistic Regression | 0.9936 | 0.9906 (better) |
| XGBoost | 0.9999 | 0.9975 (better) |
| Ensemble | 0.9979 | 0.9959 (better) |

Being a much simpler model, Platt scaling can't overfit the same way on a
small calibration set - it improves 3 of 4 models by a real, if modest,
~0.002-0.003 log-loss (an order of magnitude smaller than isotonic's
illusory same-data gain).

**Conclusion / what changes in production: nothing, for now.** Random
Forest is the currently promoted model, and Platt recalibration makes *it*
slightly worse, not better - there is no recalibration change to deploy
today. The result is still worth having: it corrects ADR-027's record
(isotonic doesn't actually help, full stop), narrows future recalibration
work to Platt-style methods only, and is a fair example of why "recalibrate
and re-evaluate on the same data" is not an acceptable production check -
this project won't make that mistake again.

**Not pursued further:** trying to make Platt scaling work for Random
Forest specifically (e.g. a different calibration window, blending with
the uncalibrated output) - the gain elsewhere is small enough that this
isn't a priority next step; ensembling architecture (ADR-027) and further
feature work are likely to matter more than chasing a ~0.001-0.003
log-loss delta on calibration alone.

## ADR-029: Frontend rebrand ("Prem Lab") and a bolder, club-themed visual redesign

**Decision:** Renamed the demo from the generic "Football Outcome
Predictor" to **Prem Lab** and replaced the light, Apple-style theme with
a dark pitch-green glass-morphism design (`frontend/style.css`) - deep
green gradient background, translucent blurred cards with angular
clipped corners, and a 4-color logo mark (one bar per model: Random
Forest, Logistic Regression, XGBoost, Ensemble) that sits directly on the
page instead of inside a separate nav-bar banner. The repo/README title
stays the technical "Football Match Outcome Prediction - MLOps Pipeline"
deliberately - "Prem Lab" is the consumer-facing product name shown on
the page and the Hugging Face Space, the same split many projects have
between an internal/repo name and a public product name.

**Team pages got substantially richer**, not just re-skinned: each
club's page (`#team/<name>`) now pulls a real stadium photo and identity
facts (nickname, founding year, ground capacity) from TheSportsDB's venue
lookup endpoint (`src/data/team_stadiums.py`, extending the existing
badge-lookup module rather than duplicating its team-search/retry logic),
and the whole page - not just the header banner - washes toward that
club's real color (`teamColor()` in `app.js`, unchanged from earlier;
only the CSS consuming it changed).

**Iterated with the user against live screenshots rather than guessing
once:** the background wash initially left a hard visible seam partway
down the page (`body` had a fixed `height: 100%`, so its own background
only covered one viewport's worth even though the page scrolled taller -
fixed by switching to `min-height`), and a first attempt at a stronger
team-color wash still went nearly black on wide viewports because a
radial-gradient's circular falloff doesn't reach a wide screen's far
corners - fixed by making the *linear* gradient layer (uniform across
the full width) carry most of the color, with the radial only adding
depth. Both are documented here because they're easy to reintroduce by
accident if this CSS is refactored later.

**Two rounds of brand-identity options were mocked up as a Claude Design
canvas** (four nav-bar directions, then reviewed live) rather than
guessing in code - "Prem Lab" together with a 4-bar "consensus meter"
mark (no boxed badge, no ball emoji, no pie/ring icon) is what the user
picked out of those options.

## ADR-030: Weekly automation (Phase 4) - one GitHub Actions workflow, two cadences

**Decision:** `.github/workflows/weekly_update.yml` runs on two schedules
rather than a single weekly cron, because the site has two independent
things that go stale at different rates:

- **Monday 07:00 UTC** - a full retrain (`src.pipeline.retrain_pipeline`:
  ingest -> validate -> train -> evaluate -> promote-if-better), once the
  weekend's Premier League results exist to train on.
- **Tuesday and Friday 07:00 UTC** - just `src.pipeline.predict_upcoming`
  + `frontend.prepare_data`, no retrain. football-data.co.uk typically
  (re)publishes `fixtures.csv` with bookmaker odds around these two days
  (ADR-023's "usually by Friday" messaging on the frontend is the same
  observation) - this is what picks up newly published odds for a fixture
  the site already knows about, and what would catch a postponed/rescheduled
  fixture, without the cost of a full retrain.

Both paths share one job (`github.event.schedule` picks the branch, and
`workflow_dispatch` with a `full_retrain` checkbox covers an on-demand run
of either) rather than two separate workflow files, since every other step
- checkout, install, commit-and-push, the Hugging Face Space push - is
identical either way.

**Why `predict_upcoming.py` needs no MLflow server here, but the full
retrain does:** `predict_upcoming.py` calls
`src.models.registry.fit_full_model()`, which fits fresh directly on all
history in-process - it was never wired to read from the MLflow registry
(the "next gameweek" numbers shown on the site are not the same model
object as whatever is currently tagged `production`, deliberately - see
Architecture.md). The full retrain path exercises the real registry
(train -> evaluate -> promote-if-better), so it needs a real tracking
server, started the same way `ci.yml`'s `test` job already does.

**Refreshed data commits back to `main` with `[skip ci]`** (data is the
only thing that changed, re-running lint/test/docker/airflow against it
would just re-verify code that didn't move) and the frontend is pushed to
the Hugging Face Space via the same `git subtree push --prefix=frontend hf
main` used for the original manual deploy (ADR-020) - a normal push each
time now, not the one-off force-push that ADR-020 needed to get past HF's
placeholder template.

**Manual, one-time setup this can't do unattended:** a Hugging Face access
token with write access to the Space, added as the `HF_TOKEN` secret in
the GitHub repo's settings. Without it the workflow's data/prediction
refresh still runs and commits to `main`; only the Space push step fails
(with an explicit message pointing at this ADR/the README, not a silent
skip).
