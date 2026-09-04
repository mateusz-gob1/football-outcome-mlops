"""Normalize openfootball's full club names to football-data.co.uk's short names.

openfootball writes names like "Manchester City FC" or "AFC Bournemouth";
football-data.co.uk (the historical dataset every model in this project is
trained on) uses short, sometimes irregular abbreviations ("Man City",
"Nott'm Forest", "QPR"). Team names must match exactly for the leak-free
feature pipeline (build_features.py) to find a team's prior match history -
a silent mismatch would look like a brand-new team with zero history.
"""

import difflib
import logging

logger = logging.getLogger(__name__)

# Known irregular abbreviations a generic suffix-strip can't produce on its
# own. Keyed by the openfootball name with "FC"/"AFC" already stripped.
KNOWN_ALIASES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Coventry City": "Coventry",
    "Luton Town": "Luton",
    "Sheffield United": "Sheffield United",
}


def _strip_suffix(name: str) -> str:
    name = name.strip()
    if name.startswith("AFC "):
        return name[len("AFC ") :]
    for suffix in (" FC", " AFC"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def normalize_openfootball_team_name(name: str, known_teams: set) -> str:
    """Map an openfootball club name onto football-data.co.uk's name for the same club.

    Raises ValueError if no confident match is found, rather than silently
    returning a name build_features.py would treat as a brand-new team with
    no history (which would badly distort that team's features).
    """
    stripped = _strip_suffix(name)
    if stripped in known_teams:
        return stripped

    alias = KNOWN_ALIASES.get(stripped)
    if alias and alias in known_teams:
        return alias

    close = difflib.get_close_matches(stripped, list(known_teams), n=1, cutoff=0.6)
    if close:
        logger.warning("Fuzzy-matched openfootball team '%s' -> '%s'", name, close[0])
        return close[0]

    raise ValueError(
        f"Could not map openfootball team name '{name}' onto a known "
        "football-data.co.uk team name"
    )
