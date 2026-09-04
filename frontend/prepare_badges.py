"""One-off script: resolve every known team's badge URL via TheSportsDB.

Separate from prepare_data.py (which regenerates every time the model/data
changes) because club badges essentially never change and this hits an
external, rate-limited free API - no reason to re-fetch them on every data
refresh. Run this once, or whenever a new team appears in the data
(promotion/relegation) that isn't in frontend/data/team_badges.json yet.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.data.team_badges import build_badge_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = Path(__file__).resolve().parent / "data" / "team_badges.json"

logger = logging.getLogger(__name__)


def main() -> None:
    matches = pd.read_csv(PROJECT_ROOT / "data/processed/matches_validated.csv")
    known_teams = set(matches["HomeTeam"]) | set(matches["AwayTeam"])

    existing = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    still_missing = known_teams - existing.keys()

    if still_missing:
        badges = {**existing, **build_badge_map(still_missing)}
    else:
        badges = existing

    missing = known_teams - badges.keys()
    if missing:
        logger.warning("No badge found for: %s", sorted(missing))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(badges, indent=2))
    print(f"Saved {OUT_PATH} ({len(badges)}/{len(known_teams)} teams matched)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
