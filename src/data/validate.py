"""Validate raw match history CSVs: schema, missing values, anomalies."""

import logging
import re
from pathlib import Path

import pandas as pd

from src.data.ingest import LEAGUE_CODE, RAW_DATA_DIR

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Columns every season file must have, regardless of how many betting markets
# football-data.co.uk added over the years (see Architecture-Decisions ADR-008).
REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
    # Match stats (shots, shots on target, corners, cards) - present for every
    # season back to 2010/11 (unlike xG, only introduced 2026/27 - see
    # ADR-026), used for rolling-average features in build_features.py.
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

# BW (Bet&Win) is present in 14/16 seasons but has a large gap in 2024/25
# (141/380 matches missing - see ADR-012). Carried through when available and
# averaged with B365 for a sturdier baseline/feature; falls back to B365 alone
# for rows where it's missing, rather than dropping those matches entirely.
OPTIONAL_COLUMNS = ["BWH", "BWD", "BWA"]

# xG (expected goals) - display-only, not a model feature. Only present from
# 2026/27 onward (football-data.co.uk added it then); 16 seasons of training
# history have none, so it can't be walk-forward validated yet. Optional so
# older season files (without the columns at all) still validate.
DISPLAY_ONLY_COLUMNS = ["HxG", "AxG"]

FILENAME_RE = re.compile(rf"{LEAGUE_CODE}_(\d{{2}})(\d{{2}})\.csv")


class ValidationError(Exception):
    """Raised when a raw file fails a structural check (missing columns, unparseable dates)."""


def _season_start_year_from_filename(path: Path) -> int:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValidationError(f"Cannot parse season from filename: {path.name}")
    start_yy = int(match.group(1))
    # 2010-2025 raw files use 2-digit years in the 2000s; safe for this project's range.
    return 2000 + start_yy


def load_raw_season(path: Path) -> pd.DataFrame:
    """Load one raw season CSV and parse its date column."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"{path.name} is missing required columns: {missing}")
    for col in DISPLAY_ONLY_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df["season_start_year"] = _season_start_year_from_filename(path)
    return df


def check_season(df: pd.DataFrame, season_start_year: int) -> list[str]:
    """Return a list of human-readable anomalies found in one season's data."""
    issues = []

    null_counts = df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS].isnull().sum()
    for col, count in null_counts[null_counts > 0].items():
        issues.append(f"{count} missing value(s) in '{col}'")

    valid_scores = df["FTHG"].notna() & df["FTAG"].notna()
    expected_result = pd.Series(pd.NA, index=df.index, dtype="string")
    expected_result[valid_scores & (df["FTHG"] > df["FTAG"])] = "H"
    expected_result[valid_scores & (df["FTHG"] < df["FTAG"])] = "A"
    expected_result[valid_scores & (df["FTHG"] == df["FTAG"])] = "D"
    mismatch = valid_scores & (df["FTR"].astype("string") != expected_result)
    if mismatch.any():
        issues.append(f"{mismatch.sum()} row(s) where FTR does not match FTHG/FTAG")

    negative_scores = (df["FTHG"] < 0) | (df["FTAG"] < 0)
    if negative_scores.any():
        issues.append(f"{negative_scores.sum()} row(s) with a negative score")

    bad_odds = (df[["B365H", "B365D", "B365A", "BWH", "BWD", "BWA"]] <= 1.0).any(axis=1)
    if bad_odds.any():
        issues.append(f"{bad_odds.sum()} row(s) with implausible odds (<=1.0)")

    season_start = pd.Timestamp(year=season_start_year, month=7, day=1)
    season_end = pd.Timestamp(year=season_start_year + 1, month=7, day=31)
    out_of_range = (df["Date"] < season_start) | (df["Date"] > season_end)
    if out_of_range.any():
        issues.append(
            f"{out_of_range.sum()} row(s) dated outside the expected "
            f"{season_start_year}/{season_start_year + 1} season window"
        )

    return issues


def validate_all(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load, validate and combine every raw season file into one dataset.

    Raises ValidationError on structural problems (missing columns, bad filenames).
    Row-level anomalies (nulls, scoreline mismatches, bad odds, out-of-window dates)
    are logged as warnings rather than raising, so a handful of bad rows in one
    season don't block ingestion of the other fifteen.
    """
    files = sorted(raw_dir.glob(f"{LEAGUE_CODE}_*.csv"))
    if not files:
        raise ValidationError(f"No raw files found in {raw_dir}")

    frames = []
    for path in files:
        season_start_year = _season_start_year_from_filename(path)
        df = load_raw_season(path)
        issues = check_season(df, season_start_year)
        for issue in issues:
            logger.warning("%s: %s", path.name, issue)

        before = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        dropped = before - len(df)
        if dropped:
            logger.warning(
                "%s: dropped %d row(s) with missing required fields", path.name, dropped
            )

        frames.append(
            df[
                REQUIRED_COLUMNS
                + OPTIONAL_COLUMNS
                + DISPLAY_ONLY_COLUMNS
                + ["season_start_year"]
            ]
        )

    combined = (
        pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    )
    return combined


def save_validated(out_dir: Path = PROCESSED_DATA_DIR) -> Path:
    combined = validate_all()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "matches_validated.csv"
    combined.to_csv(path, index=False)
    logger.info("Saved %s (%d matches)", path, len(combined))
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    save_validated()
