"""Tests for the promote-if-better guard in src/models/registry.py.

The integration test uses the real local MLflow registry (already has a
production-aliased model from earlier in this project's setup) and actually
registers a new version, to prove the alias genuinely doesn't move - not just
that a pure function returns the right boolean.
"""

from mlflow import MlflowClient

from src.models import registry


def test_should_promote_pure_logic():
    assert (
        registry.should_promote(candidate_log_loss=0.9, current_best_log_loss=1.0)
        is True
    )
    assert (
        registry.should_promote(candidate_log_loss=1.0, current_best_log_loss=0.9)
        is False
    )
    assert (
        registry.should_promote(candidate_log_loss=1.0, current_best_log_loss=1.0)
        is True
    )
    assert (
        registry.should_promote(candidate_log_loss=5.0, current_best_log_loss=None)
        is True
    )


def test_retrain_pipeline_rejects_a_deliberately_worse_model():
    client = MlflowClient()
    before = client.get_model_version_by_alias(
        registry.REGISTERED_MODEL_NAME, registry.PRODUCTION_ALIAS
    )

    # Deliberately much worse than anything a real model could score (log-loss
    # for 3 balanced classes is bounded well below 5.0), simulating the
    # "substitute a worse model" scenario the plan asks to test explicitly.
    result = registry.register_and_promote(candidate_log_loss=5.0)

    assert result["promoted"] is False

    after = client.get_model_version_by_alias(
        registry.REGISTERED_MODEL_NAME, registry.PRODUCTION_ALIAS
    )
    assert (
        after.version == before.version
    ), "a worse candidate must not move the production alias"

    # The candidate was still registered as a new version (kept for history),
    # just not promoted.
    assert int(result["version"]) > int(before.version)
