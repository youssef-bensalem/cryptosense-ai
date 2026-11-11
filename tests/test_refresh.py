"""Refreshing all datasets, and packing and unpacking them for publishing. No network."""
import io
import tarfile

import pandas as pd
import pytest

from cryptosense import refresh
from cryptosense.data import DAY_MS, load_processed
from cryptosense.refresh import (
    BINANCE_URL_ENV, PROCESSED_ARCHIVE, RAW_ARCHIVE, DataShrank, datasets, format_results, main, make_exchange,
    pack_data, unpack_data, update_all,
)

START_MS = 1_704_067_200_000  # 2024-01-01 00:00 UTC
M15_MS = 15 * 60 * 1000
STEP = {"1d": DAY_MS, "15m": M15_MS}
DATA_URL = "https://data-api.binance.vision/api/v3"


class FakeExchange:
    """Serves ``count[timeframe]`` consecutive candles per symbol; ``since`` is inclusive, as on Binance."""

    def __init__(self, count, max_limit=1000):
        self.count, self.max_limit, self.requests = dict(count), max_limit, []

    def parse8601(self, text):
        return pd.Timestamp(text).value // 1_000_000

    def milliseconds(self):
        return START_MS + 400 * DAY_MS

    def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None):
        self.requests.append((symbol, timeframe, since))
        offset = 1 + ord(symbol[0]) % 7  # a different price level per symbol
        candles = [[START_MS + i * STEP[timeframe], 100.0 * offset, 110.0 * offset, 90.0 * offset, 105.0 * offset, 50.0]
                   for i in range(self.count[timeframe])]
        return [c for c in candles if c[0] >= since][: min(limit, self.max_limit)]


def now_after(days):
    return START_MS + days * DAY_MS


# --- exchange ----------------------------------------------------------------

def spy_on(exchange):
    """Replace the exchange's network call; return the list that collects the addresses it asks for."""
    calls = []

    def fake_fetch(url, method="GET", headers=None, body=None):
        calls.append(url)
        if "exchangeInfo" in url:
            return {"symbols": [{"symbol": "SOLUSDT", "status": "TRADING", "baseAsset": "SOL", "quoteAsset": "USDT",
                                 "baseAssetPrecision": 8, "quoteAssetPrecision": 8, "quotePrecision": 8,
                                 "orderTypes": [], "isSpotTradingAllowed": True, "isMarginTradingAllowed": False,
                                 "filters": [], "permissions": ["SPOT"], "permissionSets": [["SPOT"]]}]}
        return [[START_MS, "1", "2", "0.5", "1.5", "10", 2, "15", 3, "5", "7", "0"]]

    exchange.fetch = fake_fetch
    return calls


def test_exchange_uses_the_market_data_address_and_nothing_else():
    exchange = make_exchange(env={BINANCE_URL_ENV: DATA_URL + "/"})
    calls = spy_on(exchange)
    assert exchange.fetch_ohlcv("SOL/USDT", timeframe="1d", since=START_MS, limit=5) == [
        [START_MS, 1.0, 2.0, 0.5, 1.5, 10.0]]
    assert calls and all(url.startswith(DATA_URL + "/") for url in calls)


def test_exchange_defaults_to_the_main_address_and_spot_only():
    exchange = make_exchange(env={})
    calls = spy_on(exchange)
    exchange.fetch_ohlcv("SOL/USDT", timeframe="1d", since=START_MS, limit=5)
    assert all(url.startswith("https://api.binance.com/api/v3/") for url in calls)   # no futures addresses
    assert make_exchange(public_url=DATA_URL, env={}).urls["api"]["public"] == DATA_URL


@pytest.mark.parametrize("url", ["http://data-api.binance.vision/api/v3", "ftp://x", "data-api.binance.vision"])
def test_exchange_rejects_addresses_that_are_not_https(url):
    with pytest.raises(ValueError):
        make_exchange(env={BINANCE_URL_ENV: url})


# --- update ------------------------------------------------------------------

def test_datasets_are_daily_for_all_and_15m_for_har_assets():
    assert datasets() == [("SOL/USDT", "1d"), ("BTC/USDT", "1d"), ("ETH/USDT", "1d"),
                          ("BTC/USDT", "15m"), ("SOL/USDT", "15m")]


def test_update_all_downloads_every_dataset_and_drops_incomplete_candles(tmp_path):
    exchange = FakeExchange({"1d": 50, "15m": 2500})
    seen = []
    results = update_all(exchange, tmp_path, now_after(40), notify=lambda *call: seen.append(call))
    assert [(r["symbol"], r["timeframe"], r["unit"]) for r in results] == [
        ("SOL/USDT", "1d", "days"), ("BTC/USDT", "1d", "days"), ("ETH/USDT", "1d", "days"),
        ("BTC/USDT", "15m", "15-minute candles"), ("SOL/USDT", "15m", "15-minute candles")]
    assert [r["rows"] for r in results] == [40, 40, 40, 2500, 2500]       # day 40 onward is not complete yet
    assert all(r["problems"] == {} for r in results)
    assert seen == [(symbol, timeframe, True) for symbol, timeframe in datasets()]
    assert len(load_processed("BTC/USDT", tmp_path, timeframe="15m")) == 2500
    assert load_processed("SOL/USDT", tmp_path)["close"].iloc[0] != load_processed("BTC/USDT", tmp_path)["close"].iloc[0]


def test_second_update_fetches_only_new_candles(tmp_path):
    update_all(FakeExchange({"1d": 40, "15m": 96 * 40}), tmp_path, now_after(40))
    later = FakeExchange({"1d": 43, "15m": 96 * 43})
    seen = []
    results = update_all(later, tmp_path, now_after(43), notify=lambda *call: seen.append(call))
    assert [r["rows"] for r in results] == [43, 43, 43, 96 * 43, 96 * 43]
    assert [r["new_candles"] for r in results] == [3, 3, 3, 96 * 3, 96 * 3]
    assert all(first_time is False for _, _, first_time in seen)
    assert all(since > START_MS + 39 * DAY_MS for _, _, since in later.requests)   # nothing old is re-read


def test_update_all_stops_if_a_dataset_shrinks(tmp_path):
    update_all(FakeExchange({"1d": 40, "15m": 200}), tmp_path, now_after(40))
    (tmp_path / "raw" / "sol_usdt_1d_raw.csv").unlink()      # forces a full download of SOL daily ...
    with pytest.raises(DataShrank, match="SOL/USDT 1d: 40 candles before the update, 10 after"):
        update_all(FakeExchange({"1d": 10, "15m": 200}), tmp_path, now_after(40))   # ... which comes back short


def test_format_results_lists_every_dataset(tmp_path):
    lines = format_results(update_all(FakeExchange({"1d": 40, "15m": 200}), tmp_path, now_after(40)))
    assert lines[0] == "SOL/USDT 1d: 40 days to 2024-02-09, 40 fetched (clean)"
    assert lines[3] == "BTC/USDT 15m: 200 15-minute candles to 2024-01-03, 200 fetched (clean)"


# --- pack and unpack ---------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path):
    path = tmp_path / "data"
    update_all(FakeExchange({"1d": 40, "15m": 200}), path, now_after(40))
    return path


def names_in(archive):
    with tarfile.open(archive) as tar:
        return sorted(tar.getnames())


def test_pack_holds_exactly_the_market_data_files(data_dir, tmp_path):
    news = data_dir / "news"
    news.mkdir()
    (news / "labels.csv").write_text("secret study data", encoding="utf-8")
    (data_dir / "processed" / "notes.csv").write_text("not a dataset", encoding="utf-8")
    paths = pack_data(data_dir, tmp_path / "out")
    assert paths == {"processed": tmp_path / "out" / PROCESSED_ARCHIVE, "raw": tmp_path / "out" / RAW_ARCHIVE}
    assert names_in(paths["processed"]) == [
        "processed/btc_usdt_15m.csv", "processed/btc_usdt_1d.csv", "processed/eth_usdt_1d.csv",
        "processed/sol_usdt_15m.csv", "processed/sol_usdt_1d.csv"]
    assert names_in(paths["raw"]) == [
        "raw/btc_usdt_15m_raw.csv", "raw/btc_usdt_1d_raw.csv", "raw/eth_usdt_1d_raw.csv",
        "raw/sol_usdt_15m_raw.csv", "raw/sol_usdt_1d_raw.csv"]


def test_pack_refuses_to_write_an_incomplete_archive(data_dir, tmp_path):
    (data_dir / "processed" / "eth_usdt_1d.csv").unlink()
    with pytest.raises(FileNotFoundError, match="processed/eth_usdt_1d.csv"):
        pack_data(data_dir, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unpack_restores_identical_files_and_the_next_update_is_incremental(data_dir, tmp_path):
    paths = pack_data(data_dir, tmp_path / "out")
    restored = tmp_path / "restored"
    written = unpack_data(paths["processed"], restored) + unpack_data(paths["raw"].read_bytes(), restored)
    assert len(written) == 10
    for name in written:
        assert (restored / name).read_bytes() == (data_dir / name).read_bytes()
    assert not list(restored.rglob("*.part"))
    later = FakeExchange({"1d": 42, "15m": 400})
    results = update_all(later, restored, now_after(42))
    assert [r["new_candles"] for r in results] == [2, 2, 2, 200, 200]


def make_archive(entries):
    """A .tar.gz in memory from (name, content or None for a directory/link) entries."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.SYMTYPE:
                info.linkname = content
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.mark.parametrize("name, content, kind", [
    ("../evil.csv", b"x", tarfile.REGTYPE),
    ("processed/../../evil.csv", b"x", tarfile.REGTYPE),
    ("/tmp/evil.csv", b"x", tarfile.REGTYPE),
    ("news/labels.csv", b"x", tarfile.REGTYPE),
    ("processed/script.py", b"x", tarfile.REGTYPE),
    ("processed/sub/btc_usdt_1d.csv", b"x", tarfile.REGTYPE),
    ("processed/BTC_USDT_1d.csv", b"x", tarfile.REGTYPE),
    ("processed/link.csv", "/etc/passwd", tarfile.SYMTYPE),
])
def test_unpack_rejects_anything_but_the_expected_files_and_writes_nothing(tmp_path, name, content, kind):
    good = ("processed/btc_usdt_1d.csv", b"timestamp,open\n1,2\n", tarfile.REGTYPE)
    archive = make_archive([good, (name, content, kind)])
    with pytest.raises(ValueError, match="Unexpected entry"):
        unpack_data(archive, tmp_path / "data")
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "evil.csv").exists()


def test_unpack_replaces_existing_files(tmp_path):
    target = tmp_path / "processed" / "btc_usdt_1d.csv"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")
    unpack_data(make_archive([("processed/btc_usdt_1d.csv", b"new", tarfile.REGTYPE)]), tmp_path)
    assert target.read_text(encoding="utf-8") == "new"


# --- command line ------------------------------------------------------------

def test_command_line_update_pack_unpack(tmp_path, capsys):
    data, out, copy = tmp_path / "data", tmp_path / "out", tmp_path / "copy"
    exchange = FakeExchange({"1d": 40, "15m": 200})
    assert main(["--data-dir", str(data), "update"], exchange=exchange, now_ms=now_after(40)) == 0
    printed = capsys.readouterr().out
    assert "SOL/USDT 1d: downloading the whole history" in printed
    assert "ETH/USDT 1d: 40 days to 2024-02-09, 40 fetched (clean)" in printed
    assert main(["--data-dir", str(data), "pack", "--out", str(out)]) == 0
    assert main(["--data-dir", str(copy), "unpack", str(out / PROCESSED_ARCHIVE), str(out / RAW_ARCHIVE)]) == 0
    assert "restored raw/sol_usdt_15m_raw.csv" in capsys.readouterr().out
    assert (copy / "processed" / "sol_usdt_15m.csv").read_bytes() == (data / "processed" / "sol_usdt_15m.csv").read_bytes()


def test_command_line_reports_a_failed_update_without_a_traceback(tmp_path, capsys):
    class Refused(FakeExchange):
        def fetch_ohlcv(self, *args, **kwargs):
            raise RuntimeError("451 restricted location")

    assert main(["--data-dir", str(tmp_path), "update"], exchange=Refused({}), now_ms=now_after(1)) == 1
    assert "Update failed: RuntimeError: 451 restricted location" in capsys.readouterr().err
    assert refresh.DEFAULT_DATA_DIR.name == "data"
