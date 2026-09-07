import pandas as pd
import pytest

from src.data import ingest


class FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ingest.requests.exceptions.HTTPError(f"{self.status_code} error")


def test_get_with_retry_retries_on_a_transient_error(monkeypatch):
    responses = [FakeResponse(503), FakeResponse(200, b"hello")]
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: responses.pop(0))

    content = ingest._get_with_retry("https://example.test/x")

    assert content == b"hello"
    assert responses == []


def test_get_with_retry_gives_up_after_repeated_transient_errors(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResponse(503))

    with pytest.raises(ingest.requests.exceptions.HTTPError):
        ingest._get_with_retry("https://example.test/x", retries=3)


def test_get_with_retry_honors_retry_after_header(monkeypatch):
    waits = []
    responses = [
        FakeResponse(503, headers={"Retry-After": "285"}),
        FakeResponse(200, b"hello"),
    ]
    monkeypatch.setattr(ingest.time, "sleep", waits.append)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: responses.pop(0))

    content = ingest._get_with_retry("https://example.test/x")

    assert content == b"hello"
    # Capped at MAX_RETRY_AFTER_SECONDS rather than waiting the full 285s -
    # one slow/broken response shouldn't stall the rest of the pipeline.
    assert waits == [ingest.MAX_RETRY_AFTER_SECONDS]


def test_get_with_retry_does_not_retry_a_non_transient_error(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ingest.requests,
        "get",
        lambda *a, **k: calls.append(1) or FakeResponse(404),
    )

    with pytest.raises(ingest.requests.exceptions.HTTPError):
        ingest._get_with_retry("https://example.test/x")

    assert len(calls) == 1


def _fake_combined_csv() -> bytes:
    rows = [
        # In range for the 2010/11 season slice (Aug 2010 - Jul 2011).
        "E0,2010-08-14,Arsenal,Liverpool,1,1,D,10,8,4,3,12,9,10,8,1,1,0,0,2.0,3.2,3.5",
        # A different league - must be filtered out.
        "I1,2010-08-14,Juventus,Roma,2,0,H,12,6,5,2,11,10,8,6,2,0,0,0,1.8,3.4,4.0",
        # Outside the 2010/11 window (belongs to 2011/12) - must not leak in.
        "E0,2011-08-20,Chelsea,Everton,2,1,H,9,7,3,2,10,13,9,10,1,2,0,0,1.5,4.0,6.0",
    ]
    header = (
        "Division,MatchDate,HomeTeam,AwayTeam,FTHome,FTAway,FTResult,"
        "HomeShots,AwayShots,HomeTarget,AwayTarget,HomeFouls,AwayFouls,"
        "HomeCorners,AwayCorners,HomeYellow,AwayYellow,HomeRed,AwayRed,"
        "OddHome,OddDraw,OddAway"
    )
    return ("\n".join([header, *rows]) + "\n").encode("utf-8")


def test_download_season_csv_filters_to_premier_league_and_the_right_season(
    monkeypatch,
):
    monkeypatch.setattr(ingest, "_matches_cache", None)
    monkeypatch.setattr(ingest, "_get_with_retry", lambda *a, **k: _fake_combined_csv())

    content = ingest.download_season_csv(2010)
    df = pd.read_csv(pd.io.common.BytesIO(content))

    assert len(df) == 1
    assert df.iloc[0]["HomeTeam"] == "Arsenal"
    assert df.iloc[0]["Date"] == "2010-08-14"
    # BW columns must exist (even empty) - validate.py's REQUIRED_COLUMNS +
    # OPTIONAL_COLUMNS check wants them present regardless of value.
    assert {"BWH", "BWD", "BWA"}.issubset(df.columns)
    assert df["BWH"].isna().all()


def test_fetch_combined_matches_is_cached_across_calls(monkeypatch):
    monkeypatch.setattr(ingest, "_matches_cache", None)
    calls = []
    monkeypatch.setattr(
        ingest,
        "_get_with_retry",
        lambda *a, **k: calls.append(1) or _fake_combined_csv(),
    )

    ingest.fetch_combined_matches()
    ingest.fetch_combined_matches()

    assert len(calls) == 1
