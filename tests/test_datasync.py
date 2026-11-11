"""Host-side data sync: download the published archive only when it changed, and never break good data."""
import tarfile
import threading

import pytest
from fastapi.testclient import TestClient

from cryptosense import service
from cryptosense.api import create_app
from cryptosense.datasync import DEFAULT_SYNC_MINUTES, MIN_SYNC_MINUTES, SYNC_MINUTES_ENV, DataSync, sync_minutes
from cryptosense.refresh import pack_data, update_all
from test_refresh import FakeExchange, make_archive, now_after
from test_service import NOW

URL = "https://example.test/releases/download/market-data/market-data-processed.tar.gz"


class FakeReply:
    def __init__(self, content=b"", headers=None, status=200):
        self.content, self.headers, self.status = content, headers or {}, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class FakeSession:
    """Serves one archive; counts HEAD and GET requests."""

    def __init__(self, content, headers=None, head_status=200, get_status=200):
        self.content, self.headers = content, headers if headers is not None else {"ETag": '"v1"'}
        self.head_status, self.get_status = head_status, get_status
        self.heads, self.gets = 0, 0

    def head(self, url, allow_redirects=True, timeout=None):
        self.heads += 1
        return FakeReply(headers=self.headers, status=self.head_status)

    def get(self, url, allow_redirects=True, timeout=None):
        self.gets += 1
        return FakeReply(self.content, self.headers, self.get_status)


def archive(text):
    return make_archive([("processed/btc_usdt_1d.csv", text.encode(), tarfile.REGTYPE)])


def read(data_dir):
    return (data_dir / "processed" / "btc_usdt_1d.csv").read_text(encoding="utf-8")


def test_first_check_downloads_and_unpacks(tmp_path):
    session = FakeSession(archive("first"))
    sync = DataSync(URL, tmp_path, session=session)
    assert sync.check() == "updated"
    assert read(tmp_path) == "first" and session.gets == 1


def test_unchanged_archive_is_not_downloaded_again(tmp_path):
    session = FakeSession(archive("first"))
    sync = DataSync(URL, tmp_path, session=session)
    sync.check()
    assert sync.check() == "unchanged" and sync.check() == "unchanged"
    assert session.gets == 1 and session.heads == 3


def test_changed_archive_replaces_the_files(tmp_path):
    session = FakeSession(archive("first"))
    sync = DataSync(URL, tmp_path, session=session)
    sync.check()
    session.content, session.headers = archive("second"), {"ETag": '"v2"'}
    assert sync.check() == "updated"
    assert read(tmp_path) == "second" and session.gets == 2


@pytest.mark.parametrize("headers, head_status", [({}, 200), ({"ETag": '"v1"'}, 405)])
def test_without_usable_headers_the_content_decides(tmp_path, headers, head_status):
    session = FakeSession(archive("first"), headers=headers, head_status=head_status)
    sync = DataSync(URL, tmp_path, session=session)
    assert sync.check() == "updated"
    target = tmp_path / "processed" / "btc_usdt_1d.csv"
    stamp = target.stat().st_mtime_ns
    assert sync.check() == "unchanged"            # downloaded again, same bytes: files are left alone
    assert session.gets == 2 and target.stat().st_mtime_ns == stamp
    session.content = archive("second")
    assert sync.check() == "updated" and read(tmp_path) == "second"


def test_failed_download_keeps_the_data_on_disk(tmp_path):
    session = FakeSession(archive("first"))
    sync = DataSync(URL, tmp_path, session=session)
    sync.check()
    session.headers, session.get_status = {"ETag": '"v2"'}, 503
    with pytest.raises(RuntimeError):
        sync.check()
    assert sync.check_quietly() == "failed"
    assert read(tmp_path) == "first"
    session.get_status, session.content = 200, archive("second")
    assert sync.check_quietly() == "updated" and read(tmp_path) == "second"     # recovers on the next check


def test_bad_archive_is_rejected_and_writes_nothing(tmp_path):
    bad = make_archive([("processed/btc_usdt_1d.csv", b"x", tarfile.REGTYPE), ("../evil.csv", b"x", tarfile.REGTYPE)])
    sync = DataSync(URL, tmp_path / "data", session=FakeSession(bad))
    assert sync.check_quietly() == "failed"
    assert not (tmp_path / "data").exists() and not (tmp_path / "evil.csv").exists()
    not_an_archive = DataSync(URL, tmp_path / "data", session=FakeSession(b"<html>Not Found</html>"))
    assert not_an_archive.check_quietly() == "failed"


def test_only_https_addresses_are_accepted(tmp_path):
    for url in ["http://example.test/a.tar.gz", "file:///etc/passwd", ""]:
        with pytest.raises(ValueError):
            DataSync(url, tmp_path, session=FakeSession(b""))


def test_sync_minutes_setting():
    assert sync_minutes({}) == DEFAULT_SYNC_MINUTES
    assert sync_minutes({SYNC_MINUTES_ENV: "30"}) == 30
    assert sync_minutes({SYNC_MINUTES_ENV: "0.1"}) == MIN_SYNC_MINUTES
    assert sync_minutes({SYNC_MINUTES_ENV: "soon"}) == DEFAULT_SYNC_MINUTES


def test_run_checks_until_stopped_and_survives_failures(tmp_path):
    session = FakeSession(archive("first"), get_status=503)

    class Stop:
        """Lets the loop run three times, then stops it; records the waits."""
        def __init__(self):
            self.waits = []

        def wait(self, seconds):
            self.waits.append(seconds)
            if len(self.waits) == 2:
                session.get_status = 200          # the server recovers before the second check
            return len(self.waits) > 3

    stop = Stop()
    DataSync(URL, tmp_path, session=session).run(stop, 1800)
    assert stop.waits == [1800] * 4
    assert session.gets == 2 and read(tmp_path) == "first"      # failed once, then updated, then unchanged


def test_api_fills_an_empty_data_folder_at_startup_and_serves_it(tmp_path):
    source = tmp_path / "source"
    update_all(FakeExchange({"1d": 400, "15m": 200}), source, now_after(400))
    published = pack_data(source, tmp_path / "publish")["processed"].read_bytes()
    host_data = tmp_path / "host-data"
    session = FakeSession(published)
    service.clear_cache()
    app = create_app(data_dir=host_data, web_dir=tmp_path / "no-web", get_client=lambda: None, now=lambda: NOW,
                     data_sync=DataSync(URL, host_data, session=session), sync_interval_s=3600)
    with TestClient(app) as client:
        assets = client.get("/api/assets").json()
        assert [asset["has_data"] for asset in assets] == [True, True, True]
        assert client.get("/api/assets/btc_usdt/prices?range=all").json()["points"][-1]["t"] == "2025-02-03"
        assert any(thread.name == "data-sync" for thread in threading.enumerate())
    assert session.gets == 1
    assert not any(thread.name == "data-sync" and thread.is_alive() for thread in threading.enumerate())
    service.clear_cache()


def test_api_starts_even_if_the_first_download_fails(tmp_path):
    session = FakeSession(b"", get_status=503)
    app = create_app(data_dir=tmp_path / "data", web_dir=tmp_path / "no-web", get_client=lambda: None,
                     data_sync=DataSync(URL, tmp_path / "data", session=session), sync_interval_s=3600)
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert [asset["has_data"] for asset in client.get("/api/assets").json()] == [False, False, False]


def test_no_sync_without_the_setting(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTOSENSE_DATA_URL", raising=False)
    app = create_app(data_dir=tmp_path, web_dir=tmp_path / "no-web", get_client=lambda: None)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert not any(thread.name == "data-sync" for thread in threading.enumerate())
