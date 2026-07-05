"""Download raw match history CSVs from football-data.co.uk."""

import datetime as dt
import logging
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
    return season_start_year_for_date(today or dt.date.today())


def download_season_csv(start_year: int, league_code: str = LEAGUE_CODE) -> bytes:
    """Fetch one season's raw CSV bytes from football-data.co.uk."""
    url = f"{BASE_URL}/{season_code(start_year)}/{league_code}.csv"
    response = requests.get(url, timeout=30)
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
