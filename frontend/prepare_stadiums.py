"""One-off script: resolve every known team's stadium photo via TheSportsDB.

Same rationale as prepare_badges.py: stadium venues essentially never change,
so this is run once (or whenever a new team appears in the data that isn't
in frontend/data/team_stadiums.json yet), not on every data refresh.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.data.team_stadiums import build_stadium_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = Path(__file__).resolve().parent / "data" / "team_stadiums.json"

logger = logging.getLogger(__name__)


def main() -> None:
    matches = pd.read_csv(PROJECT_ROOT / "data/processed/matches_validated.csv")
    known_teams = set(matches["HomeTeam"]) | set(matches["AwayTeam"])

    existing = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    still_missing = known_teams - existing.keys()

    if still_missing:
        stadiums = {**existing, **build_stadium_map(still_missing)}
    else:
        stadiums = existing

    missing = known_teams - stadiums.keys()
    if missing:
        logger.warning("No stadium found for: %s", sorted(missing))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(stadiums, indent=2))
    print(f"Saved {OUT_PATH} ({len(stadiums)}/{len(known_teams)} teams matched)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
