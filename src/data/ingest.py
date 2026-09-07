"""Download raw Premier League match history.

Originally fetched season-by-season directly from football-data.co.uk.
That source started returning 503 for every request from this project's
environments - this session's sandbox, the Claude Browser pane, and (per
CI logs) GitHub Actions runners - while a normal residential-IP browser
loads the same pages fine (confirmed live, screenshot in hand - see
ADR-034). That points at an IP/ASN-based block on cloud/datacenter
traffic, not a real outage, and no amount of retrying fixes a block.

Source moved to xgabora/Club-Football-Match-Data (MIT-licensed, actively
updated), a GitHub-hosted mirror of the same football-data.co.uk numbers
(results, shots, corners, cards, Bet365 odds) - hosted on GitHub's own
CDN, which GitHub Actions obviously has no trouble reaching. See ADR-034.
"""

import datetime as dt
import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LEAGUE_CODE = "E0"  # Premier League
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

MATCHES_SOURCE_URL = (
    "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data"
    "/main/data/Matches.csv"
)

# xgabora's combined table uses its own column names across all 38 leagues;
# renamed back to football-data.co.uk's original names so nothing downstream
# (src/data/validate.py's REQUIRED_COLUMNS, src/features/build_features.py)
# needs to know the source changed.
COLUMN_RENAME = {
    "Division": "Div",
    "MatchDate": "Date",
    "FTHome": "FTHG",
    "FTAway": "FTAG",
    "FTResult": "FTR",
    "HomeShots": "HS",
    "AwayShots": "AS",
    "HomeTarget": "HST",
    "AwayTarget": "AST",
    "HomeFouls": "HF",
    "AwayFouls": "AF",
    "HomeCorners": "HC",
    "AwayCorners": "AC",
    "HomeYellow": "HY",
    "AwayYellow": "AY",
    "HomeRed": "HR",
    "AwayRed": "AR",
    "OddHome": "B365H",
    "OddDraw": "B365D",
    "OddAway": "B365A",
}
OUTPUT_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] + [
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
    "B365H",
    "B365D",
    "B365A",
]
# xgabora doesn't carry Bet&Win odds at all - src/data/validate.py's
# REQUIRED_COLUMNS + OPTIONAL_COLUMNS check wants the columns present even
# when every value in them is null (that's the documented "falls back to
# B365 alone" path, already exercised for seasons with partial BW gaps).
OPTIONAL_OUTPUT_COLUMNS = ["BWH", "BWD", "BWA"]


def season_code(start_year: int) -> str:
    """Convert a season start year (2010 for 2010/11) to football-data.co.uk's code ("1011")."""
    end_year = start_year + 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def season_start_year_for_date(d: dt.date) -> int:
    """Start year of the Premier League season that `d` falls in (season starts in August)."""
    return d.year - 1 if d.month < 8 else d.year


def latest_season_start_year(today: dt.date | None = None) -> int:
    """Start year of the most recently started season, as of `today`."""
    return season_start_year_for_date(
        today or dt.datetime.now(tz=dt.timezone.utc).date()
    )


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER_SECONDS = 90  # cap how long one request can hold up the whole ingest


def _get_with_retry(url: str, retries: int = 3) -> bytes:
    """GET a URL, retried with backoff on transient errors.

    Honors the server's own `Retry-After` header when it sends one (capped
    at MAX_RETRY_AFTER_SECONDS - a multi-minute outage shouldn't hang the
    whole pipeline waiting on one file), falling back to a short
    exponential backoff when it doesn't.
    """
    for attempt in range(retries):
        response = requests.get(url, timeout=60)
        if response.status_code in TRANSIENT_STATUS_CODES and attempt < retries - 1:
            retry_after = response.headers.get("Retry-After")
            wait = (
                min(int(retry_after), MAX_RETRY_AFTER_SECONDS)
                if retry_after and retry_after.isdigit()
                else 2 * (attempt + 1)
            )
            logger.info("%s for %s, retrying in %ss", response.status_code, url, wait)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.content
    raise RuntimeError(f"unreachable: retries={retries}")  # pragma: no cover


_matches_cache: pd.DataFrame | None = None


def fetch_combined_matches(retries: int = 3, use_cache: bool = True) -> pd.DataFrame:
    """Fetch and cache the full multi-league match table, filtered to the Premier League.

    One ~45MB download regardless of how many seasons `ingest()` asks for -
    cached at module level so a 16-season backfill doesn't refetch it 16
    times in the same process.
    """
    global _matches_cache
    if use_cache and _matches_cache is not None:
        return _matches_cache

    content = _get_with_retry(MATCHES_SOURCE_URL, retries=retries)
    df = pd.read_csv(StringIO(content.decode("utf-8")), low_memory=False)
    df = df[df["Division"] == LEAGUE_CODE].copy()
    df = df.rename(columns=COLUMN_RENAME)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in OPTIONAL_OUTPUT_COLUMNS:
        df[col] = pd.NA

    if use_cache:
        _matches_cache = df
    return df


def download_season_csv(start_year: int, retries: int = 3) -> bytes:
    """Return one season's raw CSV bytes, sliced from the combined match table."""
    matches = fetch_combined_matches(retries=retries)
    season_start = pd.Timestamp(year=start_year, month=8, day=1)
    season_end = pd.Timestamp(year=start_year + 1, month=7, day=31)
    season_rows = matches[
        (matches["Date"] >= season_start) & (matches["Date"] <= season_end)
    ].sort_values("Date")
    out = season_rows[OUTPUT_COLUMNS + OPTIONAL_OUTPUT_COLUMNS].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")


def ingest(
    start_year: int = 2010,
    end_year: int | None = None,
    out_dir: Path = RAW_DATA_DIR,
) -> list[Path]:
    """Download and save raw CSVs for every season in [start_year, end_year]."""
    end_year = end_year if end_year is not None else latest_season_start_year()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for year in range(start_year, end_year + 1):
        content = download_season_csv(year)
        path = out_dir / f"{LEAGUE_CODE}_{season_code(year)}.csv"
        path.write_bytes(content)
        saved.append(path)
        logger.info("Saved %s (%d bytes)", path, len(content))
    return saved


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ingest()
