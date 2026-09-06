import pytest

from src.data import ingest


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ingest.requests.exceptions.HTTPError(f"{self.status_code} error")


def test_download_season_csv_retries_on_a_transient_error(monkeypatch):
    responses = [FakeResponse(503), FakeResponse(200, b"date,home,away\n")]
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: responses.pop(0))

    content = ingest.download_season_csv(2010)

    assert content == b"date,home,away\n"
    assert responses == []


def test_download_season_csv_gives_up_after_repeated_transient_errors(monkeypatch):
    monkeypatch.setattr(ingest.time, "sleep", lambda _: None)
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResponse(503))

    with pytest.raises(ingest.requests.exceptions.HTTPError):
        ingest.download_season_csv(2010, retries=3)


def test_download_season_csv_does_not_retry_a_non_transient_error(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ingest.requests,
        "get",
        lambda *a, **k: calls.append(1) or FakeResponse(404),
    )

    with pytest.raises(ingest.requests.exceptions.HTTPError):
        ingest.download_season_csv(2010)

    assert len(calls) == 1
