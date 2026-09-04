"""Fetch the next unplayed Premier League gameweek from openfootball/england.

football-data.co.uk (this project's main data source) never publishes a
future-fixture list for the Premier League - only completed results, updated
as the season goes. openfootball/england publishes a full season schedule
per matchday, in a plain-text format, including fixtures that haven't been
played yet (no score). This module parses just enough of that format to
answer one question: which matches make up the next Premier League gameweek
that hasn't finished yet?
"""

import datetime as dt
import logging
import re

import requests

from src.data.team_names import normalize_openfootball_team_name

logger = logging.getLogger(__name__)

RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/openfootball/england/master/"
    "{season}/1-premierleague.txt"
)

MATCHDAY_RE = re.compile(r"^▪\s*Matchday\s+(\d+)\s*$")
DAY_HEADER_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$"
)
TIME_RE = re.compile(r"^\s*(\d{1,2}:\d{2})\s+(.*)$")
SCORE_SUFFIX_RE = re.compile(r"\s{2,}(\d+-\d+(?:\s*\(\d+-\d+\))?)\s*$")

MONTHS = {
    name: i + 1
    for i, name in enumerate(
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    )
}


def season_code(start_year: int) -> str:
    """"2026-27" style folder name openfootball uses for a season starting in `start_year`."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def fetch_season_text(start_year: int) -> str:
    url = RAW_URL_TEMPLATE.format(season=season_code(start_year))
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _parse_match_line(line: str) -> tuple[str, str, bool] | None:
    """Parse one fixture line into (home, away, played). None if the line isn't a fixture."""
    time_match = TIME_RE.match(line)
    rest = time_match.group(2) if time_match else line.strip()
    if " v " not in rest:
        return None
    home, remainder = rest.split(" v ", 1)
    score_match = SCORE_SUFFIX_RE.search(remainder)
    if score_match:
        away = remainder[: score_match.start()].strip()
        played = True
    else:
        away = remainder.strip()
        played = False
    return home.strip(), away, played


def parse_fixtures(text: str, start_year: int) -> list[dict]:
    """Parse every match in an openfootball season file into fixture records.

    Each record: {matchday, date (datetime.date), home, away, played}. The
    season spans two calendar years (e.g. Aug 2026 - May 2027); the file
    doesn't always repeat the year on every day header, so month >= August
    is treated as `start_year` and month < August as `start_year + 1`.
    """
    end_year = start_year + 1
    fixtures: list[dict] = []
    current_date: dt.date | None = None
    current_matchday: int | None = None

    for line in text.splitlines():
        stripped = line.strip()
        matchday_match = MATCHDAY_RE.match(stripped)
        if matchday_match:
            current_matchday = int(matchday_match.group(1))
            continue

        day_match = DAY_HEADER_RE.match(line)
        if day_match:
            month_abbr, day_str, year_str = day_match.groups()
            month = MONTHS[month_abbr]
            year = int(year_str) if year_str else (start_year if month >= 8 else end_year)
            current_date = dt.date(year, month, int(day_str))
            continue

        parsed = _parse_match_line(line)
        if parsed is None or current_date is None or current_matchday is None:
            continue
        home, away, played = parsed
        fixtures.append(
            {
                "matchday": current_matchday,
                "date": current_date,
                "home": home,
                "away": away,
                "played": played,
            }
        )

    return fixtures


def select_next_gameweek(fixtures: list[dict]) -> list[dict]:
    """Pure selection logic: the earliest matchday that still has an unplayed fixture.

    Separated from the network/parsing step so it can be tested directly
    against hand-built fixture lists, without hitting GitHub.
    """
    unplayed_matchdays = sorted({f["matchday"] for f in fixtures if not f["played"]})
    if not unplayed_matchdays:
        return []
    target = unplayed_matchdays[0]
    return [f for f in fixtures if f["matchday"] == target and not f["played"]]


def next_gameweek_fixtures(known_teams: set, start_year: int | None = None) -> list[dict]:
    """Return the next unplayed Premier League gameweek's fixtures.

    Team names are normalized onto football-data.co.uk's naming (see
    src/data/team_names.py) so the rest of the pipeline can look up each
    team's history without a separate mapping step downstream. Each record:
    {"Date": datetime.date, "HomeTeam": str, "AwayTeam": str}.
    """
    if start_year is None:
        today = dt.date.today()
        start_year = today.year - 1 if today.month < 8 else today.year

    text = fetch_season_text(start_year)
    fixtures = parse_fixtures(text, start_year)
    gameweek = select_next_gameweek(fixtures)

    if not gameweek:
        logger.warning(
            "No unplayed fixtures found in the %s-%s season file", start_year, start_year + 1
        )
        return []

    result = []
    for f in gameweek:
        try:
            home = normalize_openfootball_team_name(f["home"], known_teams)
            away = normalize_openfootball_team_name(f["away"], known_teams)
        except ValueError as exc:
            logger.warning("Skipping fixture %s v %s: %s", f["home"], f["away"], exc)
            continue
        result.append(
            {"Date": f["date"], "HomeTeam": home, "AwayTeam": away, "Matchday": f["matchday"]}
        )
    return result
