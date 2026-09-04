import datetime as dt

from src.data.fixtures_openfootball import parse_fixtures, select_next_gameweek

SAMPLE_TEXT = """\
= English Premier League 2026/27

# Date       Fri Aug 21 2026 - Sun May 30 2027 (282d)
# Teams      20
# Matches    380



▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC              v Coventry City FC         3-0 (2-0)
  Sat Aug 22
    12:30  Hull City AFC           v Manchester United FC     2-0 (2-0)
           Everton FC              v Crystal Palace FC        2-0 (1-0)


▪ Matchday 2
  Fri Aug 28
    20:00  Crystal Palace FC       v Manchester City FC       1-4 (0-1)
  Sat Aug 29
    12:30  Liverpool FC            v Nottingham Forest FC     2-2 (0-1)


▪ Matchday 3
  Fri Sep 4
    20:00  Ipswich Town FC         v Liverpool FC
  Sat Sep 5
    12:30  Newcastle United FC     v AFC Bournemouth
    15:00  Nottingham Forest FC    v Tottenham Hotspur FC
"""


def test_parse_fixtures_extracts_matchday_date_teams_and_played_flag():
    fixtures = parse_fixtures(SAMPLE_TEXT, start_year=2026)

    assert len(fixtures) == 8
    first = fixtures[0]
    assert first == {
        "matchday": 1,
        "date": dt.date(2026, 8, 21),
        "home": "Arsenal FC",
        "away": "Coventry City FC",
        "played": True,
    }

    unplayed = [f for f in fixtures if not f["played"]]
    assert len(unplayed) == 3
    assert all(f["matchday"] == 3 for f in unplayed)


def test_parse_fixtures_handles_day_headers_without_a_year():
    fixtures = parse_fixtures(SAMPLE_TEXT, start_year=2026)
    saturday_fixtures = [f for f in fixtures if f["date"] == dt.date(2026, 8, 22)]
    assert len(saturday_fixtures) == 2


def test_parse_fixtures_rolls_the_year_over_for_spring_months():
    text = """\
▪ Matchday 30
  Sat Jan 3
    15:00  Arsenal FC              v Chelsea FC               1-1 (0-0)
"""
    fixtures = parse_fixtures(text, start_year=2026)
    assert fixtures[0]["date"] == dt.date(2027, 1, 3)


def test_select_next_gameweek_picks_earliest_matchday_with_an_unplayed_fixture():
    fixtures = parse_fixtures(SAMPLE_TEXT, start_year=2026)
    gameweek = select_next_gameweek(fixtures)

    assert len(gameweek) == 3
    assert all(f["matchday"] == 3 and not f["played"] for f in gameweek)
    assert {(f["home"], f["away"]) for f in gameweek} == {
        ("Ipswich Town FC", "Liverpool FC"),
        ("Newcastle United FC", "AFC Bournemouth"),
        ("Nottingham Forest FC", "Tottenham Hotspur FC"),
    }


def test_select_next_gameweek_returns_empty_when_everything_is_played():
    text = """\
▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC              v Chelsea FC               1-0 (1-0)
"""
    fixtures = parse_fixtures(text, start_year=2026)
    assert select_next_gameweek(fixtures) == []
