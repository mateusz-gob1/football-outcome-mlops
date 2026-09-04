"""Resolve club stadium photos from TheSportsDB's free API.

Same free-tier terms as src/data/team_badges.py (see that module's
docstring): usable for development projects, with attribution and no
modification of the image itself. This module resolves each known team to
its home venue's photo URL once, offline - the frontend hotlinks directly to
TheSportsDB's own CDN, so we never download, store, or modify the images.
"""

import logging
import time

import requests

from src.data.team_badges import _search_team

logger = logging.getLogger(__name__)

VENUE_URL = "https://www.thesportsdb.com/api/v1/json/3/lookupvenue.php"


def _club_nickname(raw_keywords: str | None) -> str | None:
    """TheSportsDB's strKeywords is a comma-separated list ("Gunners, Gooners");
    take the first, and prefix "The " when it isn't already ("The Reds" stays
    as-is, "Gunners" becomes "The Gunners") to read as a nickname, not a tag.
    """
    if not raw_keywords:
        return None
    first = raw_keywords.split(",")[0].strip()
    if not first:
        return None
    return first if first.lower().startswith("the ") else f"The {first}"


def fetch_team_stadium(short_name: str, retries: int = 3) -> dict | None:
    """Return {"name", "image", "location", "capacity", "nickname", "founded"}
    for one team's home ground and a few identity facts, or None."""
    team = _search_team(short_name, retries)
    venue_id = team.get("idVenue") if team else None
    if not venue_id:
        logger.warning("No venue id for '%s'", short_name)
        return None

    nickname = _club_nickname(team.get("strKeywords"))
    founded = team.get("intFormedYear")

    for attempt in range(retries):
        try:
            response = requests.get(VENUE_URL, params={"id": venue_id}, timeout=20)
            if response.status_code == 429:
                wait = 2 * (attempt + 1)
                logger.info("Rate limited on venue for '%s', retrying in %ss", short_name, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("TheSportsDB venue lookup failed for '%s': %s", short_name, exc)
            return None

        venues = response.json().get("venues") or []
        if not venues:
            logger.warning("No venue record for '%s' (id %s)", short_name, venue_id)
            return None

        venue = venues[0]
        image = venue.get("strFanart1") or venue.get("strThumb")
        if not image:
            logger.warning("Venue '%s' has no photo on record", venue.get("strVenue"))
            return None
        return {
            "name": venue.get("strVenue"),
            "image": image,
            "location": venue.get("strLocation"),
            "capacity": venue.get("intCapacity"),
            "nickname": nickname,
            "founded": founded,
        }

    logger.warning("Gave up on venue for '%s' after %d retries (still rate limited)", short_name, retries)
    return None


def build_stadium_map(known_teams: set, delay_seconds: float = 1.2) -> dict:
    """Resolve a stadium photo for every team in `known_teams`; silently skip misses.

    `delay_seconds` between requests keeps this well under the shared test
    key's rate limit for a one-off batch of ~40 teams.
    """
    stadiums = {}
    for name in sorted(known_teams):
        stadium = fetch_team_stadium(name)
        if stadium:
            stadiums[name] = stadium
        time.sleep(delay_seconds)
    return stadiums
