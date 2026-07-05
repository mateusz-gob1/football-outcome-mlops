"""Data drift monitoring: compare feature distributions between an older
reference period and the most recent seasons, using Evidently AI.

Scope: feature distribution drift only (see README known limitations) - not
full prediction/performance drift, which would need real outcomes for the
most recent matches to compare against.
"""

import logging
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.models.train import FEATURE_COLUMNS, prepare_dataset

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "reports"


def run_drift_report(reference_end_season: int, out_dir: Path = REPORTS_DIR) -> dict:
    """Compare feature distributions before vs from `reference_end_season` onward."""
    data = prepare_dataset()
    reference = data[data["season_start_year"] < reference_end_season]
    current = data[data["season_start_year"] >= reference_end_season]

    definition = DataDefinition(numerical_columns=FEATURE_COLUMNS)
    reference_ds = Dataset.from_pandas(
        reference[FEATURE_COLUMNS], data_definition=definition
    )
    current_ds = Dataset.from_pandas(
        current[FEATURE_COLUMNS], data_definition=definition
    )

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current_ds, reference_data=reference_ds)

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "drift_report.html"
    json_path = out_dir / "drift_report.json"
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    logger.info("Drift report saved to %s and %s", html_path, json_path)
    logger.info(
        "Reference: %d matches (seasons < %d), current: %d matches (seasons >= %d)",
        len(reference),
        reference_end_season,
        len(current),
        reference_end_season,
    )
    return snapshot.dict()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_drift_report(reference_end_season=2023)
