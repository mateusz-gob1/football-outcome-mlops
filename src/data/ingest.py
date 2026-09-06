"""Download raw match history CSVs from football-data.co.uk."""

import datetime as dt
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUE_CODE = "E0"  # Premier League
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


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


def download_season_csv(
    start_year: int, league_code: str = LEAGUE_CODE, retries: int = 3
) -> bytes:
    """Fetch one season's raw CSV bytes from football-data.co.uk.

    Retried with backoff on transient errors (rate limiting, momentary
    outages) - this runs unattended on a schedule now (daily_update.yml),
    where a single 503 on any one of 16+ season requests would otherwise
    fail the whole run for a reason that would have gone away on its own a
    few seconds later. Honors the server's own `Retry-After` header when it
    sends one (capped at MAX_RETRY_AFTER_SECONDS - a multi-minute site-wide
    outage shouldn't hang the whole pipeline waiting on one file), falling
    back to a short exponential backoff when it doesn't.
    """
    url = f"{BASE_URL}/{season_code(start_year)}/{league_code}.csv"
    for attempt in range(retries):
        response = requests.get(url, timeout=30)
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
