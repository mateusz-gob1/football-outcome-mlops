import src.data.team_stadiums as team_stadiums


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_team_stadium_returns_none_without_a_venue_id(monkeypatch):
    monkeypatch.setattr(team_stadiums, "_search_team", lambda name, retries=3: {"idVenue": None})
    assert team_stadiums.fetch_team_stadium("Coventry") is None


def test_fetch_team_stadium_returns_none_when_team_not_found(monkeypatch):
    monkeypatch.setattr(team_stadiums, "_search_team", lambda name, retries=3: None)
    assert team_stadiums.fetch_team_stadium("Made Up FC") is None


def test_fetch_team_stadium_prefers_fanart_over_thumb(monkeypatch):
    monkeypatch.setattr(
        team_stadiums,
        "_search_team",
        lambda name, retries=3: {"idVenue": "15407", "strKeywords": "The Reds", "intFormedYear": "1892"},
    )
    monkeypatch.setattr(
        team_stadiums.requests,
        "get",
        lambda *a, **k: FakeResponse(
            {
                "venues": [
                    {
                        "strVenue": "Anfield",
                        "strFanart1": "fanart.jpg",
                        "strThumb": "thumb.jpg",
                        "strLocation": "Anfield, Liverpool",
                        "intCapacity": "55000",
                    }
                ]
            }
        ),
    )
    result = team_stadiums.fetch_team_stadium("Liverpool")
    assert result == {
        "name": "Anfield",
        "image": "fanart.jpg",
        "location": "Anfield, Liverpool",
        "capacity": "55000",
        "nickname": "The Reds",
        "founded": "1892",
    }


def test_club_nickname_prefers_first_keyword_and_adds_the_prefix():
    assert team_stadiums._club_nickname("Gunners, Gooners") == "The Gunners"
    assert team_stadiums._club_nickname("The Reds") == "The Reds"
    assert team_stadiums._club_nickname(None) is None
    assert team_stadiums._club_nickname("") is None


def test_fetch_team_stadium_falls_back_to_thumb_without_fanart(monkeypatch):
    monkeypatch.setattr(team_stadiums, "_search_team", lambda name, retries=3: {"idVenue": "15407"})
    monkeypatch.setattr(
        team_stadiums.requests,
        "get",
        lambda *a, **k: FakeResponse(
            {"venues": [{"strVenue": "Anfield", "strFanart1": "", "strThumb": "thumb.jpg", "strLocation": "x"}]}
        ),
    )
    assert team_stadiums.fetch_team_stadium("Liverpool")["image"] == "thumb.jpg"


def test_fetch_team_stadium_returns_none_when_venue_has_no_photo(monkeypatch):
    monkeypatch.setattr(team_stadiums, "_search_team", lambda name, retries=3: {"idVenue": "15407"})
    monkeypatch.setattr(
        team_stadiums.requests,
        "get",
        lambda *a, **k: FakeResponse({"venues": [{"strVenue": "Anfield", "strFanart1": "", "strThumb": ""}]}),
    )
    assert team_stadiums.fetch_team_stadium("Liverpool") is None
