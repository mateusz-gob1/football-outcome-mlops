"""Attach pre-match bookmaker odds to upcoming fixtures, when published.

football-data.co.uk's separate fixtures.csv (unlike its per-season history
files) lists not-yet-played matches across many leagues with pre-match odds,
refreshed a few times a week. At any given moment it may or may not yet
include Premier League (Div == "E0") rows for the gameweek being predicted -
this module treats that as normal, not an error: odds are display-only
(see ADR-022), never required for a prediction to exist.
"""

import logging
from io import BytesIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
LEAGUE_CODE = "E0"
ODDS_COLUMNS = ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA"]


def fetch_fixtures_odds() -> pd.DataFrame:
    """Return published pre-match odds for upcoming Premier League fixtures, if any.

    Empty DataFrame (not an exception) both when the request fails and when
    football-data.co.uk simply hasn't published E0 rows yet for this window.
    """
    try:
        response = requests.get(FIXTURES_CSV_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch %s (%s) - proceeding without pre-match odds", FIXTURES_CSV_URL, exc)
        return pd.DataFrame(columns=["HomeTeam", "AwayTeam", *ODDS_COLUMNS])

    # football-data.co.uk serves this file as UTF-8 with a BOM, but doesn't
    # declare a charset - requests then guesses Latin-1 for .text, which
    # mangles the BOM into literal characters glued onto the first column
    # name ("Div" becomes "﻿Div"). Decoding the raw bytes explicitly as
    # utf-8-sig strips the BOM correctly.
    all_fixtures = pd.read_csv(BytesIO(response.content), encoding="utf-8-sig")
    pl_fixtures = all_fixtures[all_fixtures["Div"] == LEAGUE_CODE].copy()
    if pl_fixtures.empty:
        logger.info("football-data.co.uk fixtures.csv has no %s rows yet", LEAGUE_CODE)
        return pd.DataFrame(columns=["HomeTeam", "AwayTeam", *ODDS_COLUMNS])

    keep = ["HomeTeam", "AwayTeam"] + [c for c in ODDS_COLUMNS if c in pl_fixtures.columns]
    return pl_fixtures[keep]


def attach_odds(fixtures: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Left-join pre-match odds onto `fixtures` by (home team, away team).

    Matched on team names only, not date - both this project's historical
    data and football-data.co.uk's own fixtures.csv use the same short
    naming for the same clubs, and matching on names alone is more robust
    than also requiring an exact date match against an independently
    scraped source. Fixtures with no matching odds row keep NaN odds.
    """
    fixtures = fixtures.copy()

    if odds.empty:
        for col in ODDS_COLUMNS:
            fixtures[col] = float("nan")
        return fixtures

    for col in ODDS_COLUMNS:
        if col not in odds.columns:
            fixtures[col] = float("nan")

    merged = fixtures.merge(
        odds, on=["HomeTeam", "AwayTeam"], how="left", suffixes=("", "_odds")
    )
    return merged
