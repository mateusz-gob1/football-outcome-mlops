from src.data.team_badges import _best_match


def test_best_match_prefers_premier_league_team_over_same_named_youth_side():
    teams = [
        {"strTeam": "West Brom U21", "strLeague": "English Premier League 2", "strBadge": "u21.png"},
        {"strTeam": "West Bromwich Albion", "strLeague": "English Premier League", "strBadge": "main.png"},
    ]
    assert _best_match(teams)["strBadge"] == "main.png"


def test_best_match_falls_back_to_first_result_when_no_premier_league_match():
    teams = [{"strTeam": "Some Club", "strLeague": "Belgian First Division", "strBadge": "x.png"}]
    assert _best_match(teams)["strBadge"] == "x.png"


def test_best_match_handles_no_results():
    assert _best_match([]) is None
