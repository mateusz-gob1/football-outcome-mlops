"""Automated retraining pipeline: the same cycle a scheduled job (Phase 2:
Airflow) would run after every new gameweek of results comes in.

ingest -> validate -> train (walk-forward CV, all models) -> evaluate ->
promote-if-better. Steps reuse the existing modules directly rather than
duplicating logic - this file is orchestration only.
"""

import logging

from src.data import ingest, validate
from src.models import evaluate, registry, train

logger = logging.getLogger(__name__)


def run_retrain_pipeline() -> dict:
    logger.info("Step 1/5: ingest")
    ingest.ingest()

    logger.info("Step 2/5: validate")
    validate.save_validated()

    logger.info("Step 3/5: train (walk-forward CV, all models)")
    train.train_all()

    logger.info("Step 4/5: evaluate")
    evaluate.run_evaluation()

    logger.info("Step 5/5: promote if better")
    result = registry.register_and_promote()

    if result["promoted"]:
        logger.info("Retrain cycle complete: promoted v%s", result["version"])
    else:
        logger.info(
            "Retrain cycle complete: v%s NOT promoted (log-loss %.4f vs current %.4f)",
            result["version"],
            result["candidate_log_loss"],
            result["previous_best"],
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_retrain_pipeline()
