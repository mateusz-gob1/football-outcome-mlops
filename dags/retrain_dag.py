"""Airflow DAG for the weekly retraining cycle: ingest -> validate -> train ->
evaluate -> promote-if-better.

This is the same five steps as src/pipeline/retrain_pipeline.py, but split
into separate Airflow tasks (rather than one script calling
run_retrain_pipeline() directly) - the point of using an orchestrator is that
each step is independently visible, retryable, and alertable in the Airflow
UI, not just a log line in one long-running process.
"""

from __future__ import annotations

import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="football_retrain_pipeline",
    description="Ingest new results, retrain, and promote the model if it's not worse.",
    schedule="@weekly",
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    tags=["football-outcome-mlops"],
)
def retrain_dag():
    @task
    def ingest_data():
        from src.data.ingest import ingest

        ingest()

    @task
    def validate_data():
        from src.data.validate import save_validated

        save_validated()

    @task
    def train_models():
        from src.models.train import train_all

        train_all()

    @task
    def evaluate_models():
        from src.models.evaluate import run_evaluation

        run_evaluation()

    @task
    def promote_if_better():
        from src.models.registry import register_and_promote

        result = register_and_promote()
        return result

    (
        ingest_data()
        >> validate_data()
        >> train_models()
        >> evaluate_models()
        >> promote_if_better()
    )


retrain_dag()
