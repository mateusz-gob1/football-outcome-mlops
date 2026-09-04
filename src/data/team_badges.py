"""Resolve club badge images from TheSportsDB's free API.

Free tier terms (thesportsdb.com/docs_terms_of_use.php): usable for
"development projects", trademarked logos must be used as-is (never
modified), and the site should be linked back as the source - the frontend
carries that attribution (see frontend/index.html footer). This module only
resolves each of our known team names to a badge URL once, offline - the
frontend then hotlinks directly to TheSportsDB's own CDN, so we never
download, store, or modify the images ourselves.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php"

# TheSportsDB's search wants full club names; our short/irregular
# football-data.co.uk names ("Man City", "Nott'm Forest", "QPR") don't match
# directly. Same aliasing problem as src/data/team_names.py, inverted.
SEARCH_NAME_OVERRIDES = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
    "Tottenham": "Tottenham Hotspur",
    "Brighton": "Brighton and Hove Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Swansea": "Swansea City",
    "Stoke": "Stoke City",
    "Hull": "Hull City",
    "Cardiff": "Cardiff City",
    "Leeds": "Leeds United",
    "Ipswich": "Ipswich Town",
    "Luton": "Luton Town",
    "Coventry": "Coventry City",
}


def _best_match(teams: list[dict]) -> dict | None:
    """Prefer the actual Premier League club over a same-named youth/foreign team.

    Exact match on "English Premier League", not a substring check - "Premier
    League 2" is the real name of England's U21 reserve competition, and a
    naive `"premier league" in strLeague.lower()` check would match it too,
    defeating the whole point of preferring the first team (e.g. "West Brom
    U21" is listed with strLeague "English Premier League 2").
    """
    if not teams:
        return None
    for team in teams:
        if team.get("strLeague") == "English Premier League":
            return team
    return teams[0]


def _search_team(short_name: str, retries: int = 3) -> dict | None:
    """Resolve one team's TheSportsDB record, retried with backoff on rate limiting.

    The public test API key ("3") is shared across everyone using it without
    their own registration, so it rate-limits (429) under any real request
    volume - retried with backoff rather than treated as "team not found".
    Shared by badge and stadium lookups since both start from the same team
    search.
    """
    query = SEARCH_NAME_OVERRIDES.get(short_name, short_name)
    for attempt in range(retries):
        try:
            response = requests.get(SEARCH_URL, params={"t": query}, timeout=20)
            if response.status_code == 429:
                wait = 2 * (attempt + 1)
                logger.info("Rate limited on '%s', retrying in %ss", short_name, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("TheSportsDB lookup failed for '%s': %s", short_name, exc)
            return None

        match = _best_match(response.json().get("teams") or [])
        if match is None:
            logger.warning("No TheSportsDB match for '%s' (query '%s')", short_name, query)
        return match

    logger.warning("Gave up on '%s' after %d retries (still rate limited)", short_name, retries)
    return None


def fetch_team_badge(short_name: str, retries: int = 3) -> str | None:
    """Return a badge image URL for one team, or None if no confident match is found."""
    team = _search_team(short_name, retries)
    return team.get("strBadge") if team else None


def build_badge_map(known_teams: set, delay_seconds: float = 1.2) -> dict:
    """Resolve a badge URL for every team in `known_teams`; silently skip misses.

    `delay_seconds` between requests keeps this well under the shared test
    key's rate limit for a one-off batch of ~40 teams.
    """
    badges = {}
    for name in sorted(known_teams):
        badge = fetch_team_badge(name)
        if badge:
            badges[name] = badge
        time.sleep(delay_seconds)
    return badges
