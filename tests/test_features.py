import pandas as pd

from src.features.build_features import (
    MIN_HISTORY,
    _long_format,
    build_features,
    build_standings_asof,
)


def _matches(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    defaults = [
        ("B365H", 2.5),
        ("B365D", 3.2),
        ("B365A", 2.9),
        ("BWH", 2.5),
        ("BWD", 3.2),
        ("BWA", 2.9),
        ("HS", 10),
        ("AS", 10),
        ("HST", 5),
        ("AST", 5),
        ("HC", 5),
        ("AC", 5),
        ("HY", 1),
        ("AY", 1),
        ("HR", 0),
        ("AR", 0),
    ]
    for col, default in defaults:
        if col not in df.columns:
            df[col] = default
    return df


def _two_team_season() -> pd.DataFrame:
    """A and B play each other 7 times, alternating venue, on weekly dates."""
    rows = [
        {
            "Date": "2020-08-01",
            "season_start_year": 2020,
            "HomeTeam": "A",
            "AwayTeam": "B",
            "FTHG": 2,
            "FTAG": 0,
        },
        {
            "Date": "2020-08-08",
            "season_start_year": 2020,
            "HomeTeam": "B",
            "AwayTeam": "A",
            "FTHG": 1,
            "FTAG": 1,
        },
        {
            "Date": "2020-08-15",
            "season_start_year": 2020,
            "HomeTeam": "A",
            "AwayTeam": "B",
            "FTHG": 0,
            "FTAG": 1,
        },
        {
            "Date": "2020-08-22",
            "season_start_year": 2020,
            "HomeTeam": "B",
            "AwayTeam": "A",
            "FTHG": 2,
            "FTAG": 2,
        },
        {
            "Date": "2020-08-29",
            "season_start_year": 2020,
            "HomeTeam": "A",
            "AwayTeam": "B",
            "FTHG": 3,
            "FTAG": 1,
        },
        {
            "Date": "2020-09-05",
            "season_start_year": 2020,
            "HomeTeam": "B",
            "AwayTeam": "A",
            "FTHG": 0,
            "FTAG": 0,
        },
        {
            "Date": "2020-09-12",
            "season_start_year": 2020,
            "HomeTeam": "A",
            "AwayTeam": "B",
            "FTHG": 1,
            "FTAG": 0,
        },
    ]
    return _matches(rows)


def test_rolling_form_matches_hand_computed_values():
    matches = _two_team_season()
    features = build_features(matches)

    match6 = features.iloc[5]  # 2020-09-05, Home=B, Away=A
    assert match6["home_matches_played_prior"] == 5
    assert match6["away_matches_played_prior"] == 5
    # A's points from matches 1-5: 3, 1, 0, 1, 3 = 8
    assert match6["away_form_pts_5"] == 8
    # B's points from matches 1-5: 0, 1, 3, 1, 0 = 5
    assert match6["home_form_pts_5"] == 5
    # A's results before match 6: W, D, L, D, W -> last was a win -> streak +1
    assert match6["away_streak_before"] == 1


def test_rolling_features_do_not_leak_into_the_past():
    """Changing a later match's result must not change earlier matches' features."""
    matches = _two_team_season()
    features_original = build_features(matches)

    mutated = matches.copy()
    mutated.loc[mutated.index[-1], ["FTHG", "FTAG"]] = [
        5,
        0,
    ]  # flip the last match's scoreline
    features_mutated = build_features(mutated)

    feature_cols = [
        c for c in features_original.columns if c.startswith(("home_", "away_"))
    ]
    pd.testing.assert_frame_equal(
        features_original.iloc[:-1][feature_cols],
        features_mutated.iloc[:-1][feature_cols],
    )


def test_cold_start_flag_before_min_history_threshold():
    matches = _two_team_season()
    features = build_features(matches)

    # Both teams reach MIN_HISTORY (5) prior matches at match index 5 (the 6th match).
    assert not features.iloc[:MIN_HISTORY]["has_min_history"].any()
    assert features.iloc[MIN_HISTORY:]["has_min_history"].all()


def test_table_position_is_as_of_match_date_not_final_standings():
    matches = _matches(
        [
            {
                "Date": "2021-08-01",
                "season_start_year": 2021,
                "HomeTeam": "X",
                "AwayTeam": "Y",
                "FTHG": 1,
                "FTAG": 0,
            },
            {
                "Date": "2021-08-08",
                "season_start_year": 2021,
                "HomeTeam": "Y",
                "AwayTeam": "X",
                "FTHG": 3,
                "FTAG": 0,
            },
        ]
    )
    matches["match_id"] = matches.index
    long = _long_format(matches)
    standings = build_standings_asof(long)

    # As of 2021-08-08 (before match 2 is played), only match 1 counts: X won, so X leads.
    as_of_second_match = standings[standings["Date"] == pd.Timestamp("2021-08-08")]
    x_position = as_of_second_match.loc[
        as_of_second_match["team"] == "X", "position"
    ].item()
    y_position = as_of_second_match.loc[
        as_of_second_match["team"] == "Y", "position"
    ].item()
    assert x_position == 1
    assert y_position == 2

    # The season opener has no prior matches at all - no as-of standing should exist for it.
    assert (standings["Date"] == pd.Timestamp("2021-08-01")).sum() == 0


def test_h2h_uses_only_past_meetings_within_window():
    rows = []
    date = pd.Timestamp("2019-08-01")
    # Team P beats Team Q six times in a row, alternating venue.
    for i in range(6):
        home, away = ("P", "Q") if i % 2 == 0 else ("Q", "P")
        fthg, ftag = (2, 0) if home == "P" else (0, 2)
        rows.append(
            {
                "Date": date + pd.Timedelta(days=7 * i),
                "season_start_year": 2019,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": fthg,
                "FTAG": ftag,
            }
        )
    matches = _matches(rows)
    features = build_features(matches)

    # Before the 1st meeting there is no history.
    assert features.iloc[0]["h2h_matches_count"] == 0
    assert pd.isna(features.iloc[0]["h2h_home_pts"])

    # Before the 6th meeting, only the last 5 (the H2H window) should count, all won by P.
    sixth = features.iloc[5]
    assert sixth["h2h_matches_count"] == 5
    p_is_home = sixth["HomeTeam"] == "P"
    assert (sixth["h2h_home_pts"] == 15) if p_is_home else (sixth["h2h_away_pts"] == 15)


def test_odds_implied_probabilities_sum_to_one():
    matches = _two_team_season()
    features = build_features(matches)
    total = (
        features["odds_implied_home_prob"]
        + features["odds_implied_draw_prob"]
        + features["odds_implied_away_prob"]
    )
    assert (total.round(6) == 1.0).all()
