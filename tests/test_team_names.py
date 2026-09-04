import pytest

from src.data.team_names import normalize_openfootball_team_name

KNOWN_TEAMS = {
    "Arsenal",
    "Man City",
    "Man United",
    "Nott'm Forest",
    "QPR",
    "West Brom",
    "Tottenham",
    "Brighton",
    "Wolves",
    "Newcastle",
    "West Ham",
    "Leicester",
    "Bournemouth",
    "Ipswich",
    "Chelsea",
}


@pytest.mark.parametrize(
    "openfootball_name,expected",
    [
        ("Arsenal FC", "Arsenal"),
        ("Chelsea FC", "Chelsea"),
        ("AFC Bournemouth", "Bournemouth"),
        ("Manchester City FC", "Man City"),
        ("Manchester United FC", "Man United"),
        ("Nottingham Forest FC", "Nott'm Forest"),
        ("Queens Park Rangers FC", "QPR"),
        ("West Bromwich Albion FC", "West Brom"),
        ("Tottenham Hotspur FC", "Tottenham"),
        ("Brighton & Hove Albion FC", "Brighton"),
        ("Wolverhampton Wanderers FC", "Wolves"),
        ("Newcastle United FC", "Newcastle"),
        ("Ipswich Town FC", "Ipswich"),
    ],
)
def test_known_openfootball_names_map_correctly(openfootball_name, expected):
    assert normalize_openfootball_team_name(openfootball_name, KNOWN_TEAMS) == expected


def test_unmappable_name_raises_instead_of_silently_passing_through():
    with pytest.raises(ValueError):
        normalize_openfootball_team_name("Totally Unknown Rovers FC", KNOWN_TEAMS)
