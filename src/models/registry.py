"""Register the production candidate model in MLflow Model Registry and promote it.

Uses MLflow's "alias" mechanism (`production`) rather than the older "stage"
API (Stages are deprecated as of MLflow 2.9+ in favor of aliases; see
vault/Architecture-Decisions.md ADR-010). Serving code loads the model via
`models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}`.

Promotion rule: a new model version is promoted to the `production` alias
only if its walk-forward mean log-loss is not worse than the currently
promoted version's - the same "promote only if better" guard the Phase 2
retraining pipeline will reuse.
"""

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from src.models.evaluate import load_oof, log_loss_of
from src.models.train import (
    FEATURE_COLUMNS,
    MODEL_SPECS,
    TARGET_COLUMN,
    prepare_dataset,
)

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "football_outcome_predictor"
PRODUCTION_ALIAS = "production"
CANDIDATE_MODEL = (
    "random_forest"  # best-performing ML model, see vault/Evaluation-Log.md Run 001
)
CANDIDATE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
}  # max_depth=6 won every walk-forward fold


def fit_full_model(model_name: str, params: dict | None, data):
    """Fit one MODEL_SPECS entry on the entire dataset - no walk-forward split.

    Used both for the promoted production candidate (train_production_candidate,
    below) and for predicting genuinely future fixtures
    (src/pipeline/predict_upcoming.py), where there's no held-out test season
    to speak of - the whole match history becomes the training set.
    """
    spec = MODEL_SPECS[model_name]
    feature_cols = spec.get("feature_cols", FEATURE_COLUMNS)
    model = spec["factory"](params or {})
    model.fit(data[feature_cols], data[TARGET_COLUMN])
    return model


def train_production_candidate():
    """Fit the final model on the full dataset, using the walk-forward-selected hyperparameters."""
    data = prepare_dataset()
    model = fit_full_model(CANDIDATE_MODEL, CANDIDATE_PARAMS, data)
    return model, data


def current_production_log_loss(client: MlflowClient) -> float | None:
    """Log-loss recorded for the currently promoted version, or None if nothing is promoted yet."""
    try:
        version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
        )
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return run.data.metrics.get("candidate_mean_log_loss")


def should_promote(
    candidate_log_loss: float, current_best_log_loss: float | None
) -> bool:
    """Promote if there's no production model yet, or the candidate is not worse.

    Extracted as a pure function so the "reject a worse model" behavior can be
    tested directly, without retraining a real bad model - see
    tests/test_retrain_pipeline.py.
    """
    return current_best_log_loss is None or candidate_log_loss <= current_best_log_loss


def register_and_promote(candidate_log_loss: float | None = None) -> dict:
    """Register a new candidate version and promote it only if it's not worse.

    `candidate_log_loss` can be overridden (bypassing the real oof computation)
    to test the promotion decision without needing an actually-worse trained
    model - the registration/promotion mechanics that run are the real ones.
    """
    client = MlflowClient()
    mlflow.set_experiment("football-outcome-mlops")

    if candidate_log_loss is None:
        candidate_log_loss = log_loss_of(load_oof(CANDIDATE_MODEL))
    current_best = current_production_log_loss(client)
    model, data = train_production_candidate()

    with mlflow.start_run(run_name=f"{CANDIDATE_MODEL}_production_candidate"):
        mlflow.log_param("model", CANDIDATE_MODEL)
        for key, value in CANDIDATE_PARAMS.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("candidate_mean_log_loss", candidate_log_loss)

        signature = infer_signature(
            data[FEATURE_COLUMNS], model.predict_proba(data[FEATURE_COLUMNS])
        )
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=data[FEATURE_COLUMNS].head(3),
        )

    new_version = str(model_info.registered_model_version)
    promoted = should_promote(candidate_log_loss, current_best)

    if promoted:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, new_version
        )
        logger.info(
            "Promoted %s v%s to '%s' (log-loss %.4f, previous best %s)",
            REGISTERED_MODEL_NAME,
            new_version,
            PRODUCTION_ALIAS,
            candidate_log_loss,
            f"{current_best:.4f}" if current_best is not None else "none",
        )
    else:
        logger.info(
            "Did NOT promote %s v%s: log-loss %.4f is worse than current production %.4f",
            REGISTERED_MODEL_NAME,
            new_version,
            candidate_log_loss,
            current_best,
        )
    return {
        "version": new_version,
        "promoted": promoted,
        "candidate_log_loss": candidate_log_loss,
        "previous_best": current_best,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    register_and_promote()
