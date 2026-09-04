import numpy as np
import pandas as pd

from src.data.fixtures_odds import ODDS_COLUMNS, attach_odds


def _fixtures():
    return pd.DataFrame(
        {
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Liverpool", "Man City"],
        }
    )


def test_attach_odds_joins_on_team_names():
    odds = pd.DataFrame(
        {
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Liverpool"],
            "B365H": [2.1],
            "B365D": [3.4],
            "B365A": [3.2],
        }
    )
    result = attach_odds(_fixtures(), odds)

    arsenal_row = result[result["HomeTeam"] == "Arsenal"].iloc[0]
    assert arsenal_row["B365H"] == 2.1

    chelsea_row = result[result["HomeTeam"] == "Chelsea"].iloc[0]
    assert np.isnan(chelsea_row["B365H"])


def test_attach_odds_handles_no_published_odds_at_all():
    empty_odds = pd.DataFrame(columns=["HomeTeam", "AwayTeam", *ODDS_COLUMNS])
    result = attach_odds(_fixtures(), empty_odds)

    assert len(result) == 2
    for col in ODDS_COLUMNS:
        assert result[col].isna().all()
