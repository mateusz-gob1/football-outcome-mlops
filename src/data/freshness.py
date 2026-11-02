"""Fail loudly when our results data has stopped keeping up with the fixture list.

A green Daily Update run only proves the steps ran, not that new matches
arrived (ADR-043: "next matchday" sat on October fixtures for three weeks
while every run was green). This compares openfootball's schedule - which is
authoritative about *when* matches are due, even when its scores lag - with
our own results history: fixtures that should have been played a few days ago
but have no result here mean ingestion is stale. Checking against the
schedule, rather than "newest match is N days old", keeps international
breaks and the summer gap from tripping it.
"""

import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from src.data.fixtures_openfootball import fetch_season_text, parse_fixtures
from src.data.team_names import normalize_openfootball_team_name

logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "matches_validated.csv"
)

GRACE_DAYS = 3
# One or two postponed matches are normal; a stale data source misses a
# whole matchday at once.
MIN_OVERDUE_TO_FAIL = 3


def overdue_fixtures(
    fixtures: list[dict],
    played: set[tuple[dt.date, str, str]],
    known_teams: set,
    today: dt.date,
    grace_days: int = GRACE_DAYS,
) -> list[dict]:
    """Fixtures dated at least `grace_days` ago that our results don't contain."""
    cutoff = today - dt.timedelta(days=grace_days)
    overdue = []
    for f in fixtures:
        if f["date"] > cutoff:
            continue
        try:
            key = (
                f["date"],
                normalize_openfootball_team_name(f["home"], known_teams),
                normalize_openfootball_team_name(f["away"], known_teams),
            )
        except ValueError:
            continue
        if key not in played:
            overdue.append(f)
    return overdue


def main() -> int:
    today = dt.datetime.now(tz=dt.timezone.utc).date()
    start_year = today.year - 1 if today.month < 8 else today.year

    history = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
    known_teams = set(history["HomeTeam"]) | set(history["AwayTeam"])
    played = {
        (d.date(), h, a)
        for d, h, a in zip(history["Date"], history["HomeTeam"], history["AwayTeam"])
    }

    try:
        fixtures = parse_fixtures(fetch_season_text(start_year), start_year)
    except requests.RequestException as exc:
        # Can't tell either way - a different problem from stale results.
        logger.warning("Could not load the fixture list, skipping check: %s", exc)
        return 0

    overdue = overdue_fixtures(fixtures, played, known_teams, today)
    newest = history["Date"].max().date()
    if len(overdue) >= MIN_OVERDUE_TO_FAIL:
        logger.error(
            "STALE DATA: %d fixtures dated %d+ days ago have no result in our "
            "data (newest match we have: %s). First few:",
            len(overdue),
            GRACE_DAYS,
            newest,
        )
        for f in overdue[:10]:
            logger.error("  %s  %s v %s", f["date"], f["home"], f["away"])
        return 1

    logger.info(
        "Data fresh: %d overdue fixture(s), newest match we have is %s",
        len(overdue),
        newest,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sys.exit(main())
