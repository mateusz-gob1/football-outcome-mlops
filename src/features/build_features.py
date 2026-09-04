"""Leak-free feature engineering for match outcome prediction.

All features for a given match are computed using only information available
strictly before that match's date. See tests/test_features.py for the
leakage, as-of, and cold-start checks that guard this invariant.
"""

import numpy as np
import pandas as pd

MIN_HISTORY = (
    5  # matches a team must have played before its data counts toward evaluation
)
H2H_WINDOW = 5
FORM_WINDOWS = (5, 10)


# Match-stat columns (football-data.co.uk) that mirror goals: known only
# after the match, safe as rolling-average-of-past-matches features (never
# as a same-match feature). xG isn't here - see ADR-026, only available from
# 2026/27 onward, so it can't be walk-forward validated across 16 seasons of
# history yet; kept display-only.
STAT_COLUMN_PAIRS = {
    "goals": ("FTHG", "FTAG"),
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
    "corners": ("HC", "AC"),
}

# (source column in the long-format table, output rolling-5 feature name).
# goals keeps its original names (goals_scored_5/goals_conceded_5, predating
# this generalization) rather than renaming to goals_for_5/goals_against_5
# and touching every existing caller.
ROLLING_SUM_STATS = [
    ("goals_for", "goals_scored_5"),
    ("goals_against", "goals_conceded_5"),
    ("shots_for", "shots_for_5"),
    ("shots_against", "shots_against_5"),
    ("shots_on_target_for", "shots_on_target_for_5"),
    ("shots_on_target_against", "shots_on_target_against_5"),
    ("corners_for", "corners_for_5"),
    ("corners_against", "corners_against_5"),
    ("cards_for", "cards_for_5"),
]


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home and away), sorted by team then date."""
    base_cols = ["match_id", "Date", "season_start_year"]

    home = matches[base_cols + ["HomeTeam"]].rename(columns={"HomeTeam": "team"})
    away = matches[base_cols + ["AwayTeam"]].rename(columns={"AwayTeam": "team"})

    for stat, (home_col, away_col) in STAT_COLUMN_PAIRS.items():
        home[f"{stat}_for"] = matches[home_col]
        home[f"{stat}_against"] = matches[away_col]
        away[f"{stat}_for"] = matches[away_col]
        away[f"{stat}_against"] = matches[home_col]

    # Yellow + red combined into one "cards_for" discipline indicator per
    # team, rather than also tracking the opponent's cards - a much weaker
    # signal that would double the number of new columns for little benefit.
    home["cards_for"] = matches["HY"] + matches["HR"]
    away["cards_for"] = matches["AY"] + matches["AR"]

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select(
        [
            long["goals_for"] > long["goals_against"],
            long["goals_for"] == long["goals_against"],
        ],
        [3, 1],
        default=0,
    )
    long["result"] = np.select(
        [
            long["goals_for"] > long["goals_against"],
            long["goals_for"] == long["goals_against"],
        ],
        ["W", "D"],
        default="L",
    )
    return long.sort_values(["team", "Date"]).reset_index(drop=True)


def _streak_sequence(results: list[str]) -> list[int]:
    """Streak value *after* each result: positive run of wins, negative run of losses, 0 on a draw."""
    streaks = []
    current = 0
    for r in results:
        if r == "W":
            current = current + 1 if current > 0 else 1
        elif r == "L":
            current = current - 1 if current < 0 else -1
        else:
            current = 0
        streaks.append(current)
    return streaks


def _add_rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    grp = long.groupby("team")
    long["matches_played_prior"] = grp.cumcount()

    for window in FORM_WINDOWS:
        long[f"form_pts_{window}"] = grp["points"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).sum()
        )
    for source_col, output_col in ROLLING_SUM_STATS:
        long[output_col] = grp[source_col].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).sum()
        )

    long["streak_after"] = grp["result"].transform(
        lambda s: pd.Series(_streak_sequence(s.tolist()), index=s.index)
    )
    long["streak_before"] = grp["streak_after"].shift(1).fillna(0).astype(int)

    long["rest_days"] = grp["Date"].transform(lambda s: (s - s.shift(1)).dt.days)
    return long


def build_standings_asof(long: pd.DataFrame) -> pd.DataFrame:
    """Table position for every team, as of every match date, within each season.

    Position at date D uses only matches with Date < D in the same season -
    never the end-of-season or end-of-matchday standings.
    """
    rows = []
    for season, group in long.groupby("season_start_year"):
        g = group.sort_values("Date").copy()
        g["cum_points"] = g.groupby("team")["points"].cumsum()
        g["cum_gd"] = (
            g.groupby("team")["goals_for"].cumsum()
            - g.groupby("team")["goals_against"].cumsum()
        )

        for match_date in sorted(g["Date"].unique()):
            prior = g[g["Date"] < match_date]
            if prior.empty:
                continue
            snapshot = (
                prior.sort_values("Date")
                .groupby("team")
                .last()[["cum_points", "cum_gd"]]
                .sort_values(["cum_points", "cum_gd"], ascending=False)
            )
            snapshot["position"] = range(1, len(snapshot) + 1)
            for team, row in snapshot.iterrows():
                rows.append(
                    {
                        "season_start_year": season,
                        "Date": match_date,
                        "team": team,
                        "position": row["position"],
                    }
                )
    return pd.DataFrame(rows)


def _add_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """Points earned by today's home/away team in their last up-to-5 meetings, regardless of venue then."""
    matches = matches.sort_values("Date").reset_index(drop=True)
    h2h_home_pts = np.full(len(matches), np.nan)
    h2h_away_pts = np.full(len(matches), np.nan)
    h2h_count = np.zeros(len(matches), dtype=int)

    # One entry per past meeting (not per team), so "last 5 meetings" means 5 matches.
    history: dict[frozenset, list[dict]] = {}
    for i, row in matches.iterrows():
        pair = frozenset([row["HomeTeam"], row["AwayTeam"]])
        recent = history.get(pair, [])[-H2H_WINDOW:]
        if recent:
            h2h_home_pts[i] = sum(
                m["home_pts"] if m["home_team"] == row["HomeTeam"] else m["away_pts"]
                for m in recent
            )
            h2h_away_pts[i] = sum(
                m["home_pts"] if m["home_team"] == row["AwayTeam"] else m["away_pts"]
                for m in recent
            )
            h2h_count[i] = len(recent)

        home_pts = (
            3 if row["FTHG"] > row["FTAG"] else (1 if row["FTHG"] == row["FTAG"] else 0)
        )
        away_pts = (
            3 if row["FTAG"] > row["FTHG"] else (1 if row["FTHG"] == row["FTAG"] else 0)
        )
        history.setdefault(pair, []).append(
            {
                "home_team": row["HomeTeam"],
                "home_pts": home_pts,
                "away_team": row["AwayTeam"],
                "away_pts": away_pts,
            }
        )

    matches["h2h_home_pts"] = h2h_home_pts
    matches["h2h_away_pts"] = h2h_away_pts
    matches["h2h_matches_count"] = h2h_count
    return matches


def _normalized_implied_probs(
    matches: pd.DataFrame, prefix: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """One bookmaker's implied probabilities, normalized to remove the overround."""
    raw_home = 1 / matches[f"{prefix}H"]
    raw_draw = 1 / matches[f"{prefix}D"]
    raw_away = 1 / matches[f"{prefix}A"]
    overround = raw_home + raw_draw + raw_away
    return raw_home / overround, raw_draw / overround, raw_away / overround


def _add_odds_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Average Bet365 and Bet&Win's own overround-normalized probabilities (a linear
    pool - the standard way to combine independent probability forecasts), falling
    back to Bet365 alone where Bet&Win odds are missing (see ADR-012: BW has a large
    gap in the 2024/25 season; falling back keeps every match instead of dropping ~37%
    of that one season).
    """
    b365_home, b365_draw, b365_away = _normalized_implied_probs(matches, "B365")
    bw_home, bw_draw, bw_away = _normalized_implied_probs(matches, "BW")
    has_bw = matches[["BWH", "BWD", "BWA"]].notna().all(axis=1)

    matches["odds_implied_home_prob"] = np.where(
        has_bw, (b365_home + bw_home) / 2, b365_home
    )
    matches["odds_implied_draw_prob"] = np.where(
        has_bw, (b365_draw + bw_draw) / 2, b365_draw
    )
    matches["odds_implied_away_prob"] = np.where(
        has_bw, (b365_away + bw_away) / 2, b365_away
    )
    return matches


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build the full leak-free feature set for match outcome prediction.

    `matches` must contain: Date, season_start_year, HomeTeam, AwayTeam,
    FTHG, FTAG, FTR, B365H, B365D, B365A (see src/data/validate.py).
    """
    matches = matches.copy().sort_values("Date").reset_index(drop=True)
    matches["match_id"] = matches.index

    long = _long_format(matches)
    long = _add_rolling_form(long)

    form_cols = (
        [f"form_pts_{w}" for w in FORM_WINDOWS]
        + [output_col for _, output_col in ROLLING_SUM_STATS]
        + [
            "streak_before",
            "rest_days",
            "matches_played_prior",
        ]
    )
    home_side = (
        long.merge(
            matches[["match_id", "HomeTeam"]],
            left_on=["match_id", "team"],
            right_on=["match_id", "HomeTeam"],
        )[["match_id"] + form_cols]
        .add_prefix("home_")
        .rename(columns={"home_match_id": "match_id"})
    )
    away_side = (
        long.merge(
            matches[["match_id", "AwayTeam"]],
            left_on=["match_id", "team"],
            right_on=["match_id", "AwayTeam"],
        )[["match_id"] + form_cols]
        .add_prefix("away_")
        .rename(columns={"away_match_id": "match_id"})
    )

    matches = matches.merge(home_side, on="match_id", how="left")
    matches = matches.merge(away_side, on="match_id", how="left")

    standings = build_standings_asof(long)
    matches = matches.merge(
        standings.rename(
            columns={"team": "HomeTeam", "position": "home_table_position"}
        ),
        on=["season_start_year", "Date", "HomeTeam"],
        how="left",
    )
    matches = matches.merge(
        standings.rename(
            columns={"team": "AwayTeam", "position": "away_table_position"}
        ),
        on=["season_start_year", "Date", "AwayTeam"],
        how="left",
    )

    matches = _add_h2h(matches)
    matches = _add_odds_features(matches)

    matches["has_min_history"] = (
        matches["home_matches_played_prior"] >= MIN_HISTORY
    ) & (matches["away_matches_played_prior"] >= MIN_HISTORY)

    return matches
