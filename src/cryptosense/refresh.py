"""Refresh every asset's market data, and pack or unpack it for publishing.

Usage (from the project root)::

    python -m cryptosense.refresh update            # download new candles into data/
    python -m cryptosense.refresh pack --out DIR    # write the two archives into DIR
    python -m cryptosense.refresh unpack FILE...    # restore archives into data/

``update`` runs ``data.update_symbol`` for the daily data of every watchlist
asset and the 15-minute data of the assets that use it. It is what the
dashboard's refresh button does and what the scheduled GitHub job runs.

Binance refuses its main address from some locations (for example GitHub's
servers). Set ``CRYPTOSENSE_BINANCE_URL`` to Binance's market-data address,
``https://data-api.binance.vision/api/v3``, to download from there instead.
Only public prices are read: no account, no API key, and no trading.

The archives hold only the market-data CSV files of the watchlist
(``raw/`` and ``processed/``). Nothing else in ``data/`` is ever packed; in
particular the news study's files are not.
"""

import argparse
import io
import os
import re
import sys
import tarfile
from pathlib import Path

import ccxt

from cryptosense.data import symbol_to_slug, update_symbol
from cryptosense.service import WATCHLIST
from cryptosense.volatility import HAR_SYMBOLS

BINANCE_URL_ENV = "CRYPTOSENSE_BINANCE_URL"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_ARCHIVE = "market-data-processed.tar.gz"
RAW_ARCHIVE = "market-data-raw.tar.gz"
ARCHIVES = {"processed": PROCESSED_ARCHIVE, "raw": RAW_ARCHIVE}
UNITS = {"1d": "days", "15m": "15-minute candles"}
# The only names an archive may contain, e.g. "processed/btc_usdt_15m.csv".
MEMBER_NAME = re.compile(r"(raw|processed)/[a-z0-9_]+\.csv")


class DataShrank(RuntimeError):
    """An update left fewer processed candles than before, so the result must not be published."""


def make_exchange(public_url=None, env=None):
    """A Binance exchange object for public spot prices.

    Only spot markets are loaded, so the futures addresses are never contacted.

    Parameters
    ----------
    public_url : str, optional
        Address to download from instead of Binance's main one. Defaults to
        the ``CRYPTOSENSE_BINANCE_URL`` environment variable, and to the main
        address when that is not set.
    env : mapping, optional
        Environment to read (default: the process environment).

    Raises
    ------
    ValueError
        If the address does not start with ``https://``.
    """
    env = os.environ if env is None else env
    url = (public_url or env.get(BINANCE_URL_ENV, "")).strip().rstrip("/")
    exchange = ccxt.binance({"options": {"fetchMarkets": ["spot"], "fetchCurrencies": False}})
    if url:
        if not url.startswith("https://"):
            raise ValueError(f"{BINANCE_URL_ENV} must start with https://; got {url!r}.")
        exchange.urls["api"]["public"] = url
    return exchange


def datasets():
    """Every (symbol, timeframe) the project keeps: daily for the watchlist, 15-minute for HAR assets."""
    return [(symbol, "1d") for symbol in WATCHLIST] + [(symbol, "15m") for symbol in HAR_SYMBOLS]


def _processed_rows(data_dir, symbol, timeframe):
    """Number of candles in a processed file, or 0 if it does not exist."""
    path = Path(data_dir) / "processed" / f"{symbol_to_slug(symbol)}_{timeframe}.csv"
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def update_all(exchange, data_dir, now_ms, notify=None):
    """Update every dataset with ``data.update_symbol``.

    Parameters
    ----------
    exchange : ccxt.Exchange
        Exchange object, e.g. from ``make_exchange``.
    data_dir : pathlib.Path
        Root data folder.
    now_ms : int
        Current time in milliseconds (UTC), used to drop incomplete candles.
    notify : callable, optional
        ``notify(symbol, timeframe, first_time)`` is called before each
        dataset is updated; ``first_time`` is True when its whole history is
        about to be downloaded.

    Returns
    -------
    list of dict
        One ``update_symbol`` summary per dataset, plus ``timeframe`` and
        ``unit`` ("days" or "15-minute candles").

    Raises
    ------
    DataShrank
        If a dataset ends up with fewer processed candles than it had. The
        files on disk are already rewritten at that point, so the caller must
        not publish them.
    """
    data_dir = Path(data_dir)
    results = []
    for symbol, timeframe in datasets():
        raw_path = data_dir / "raw" / f"{symbol_to_slug(symbol)}_{timeframe}_raw.csv"
        before = _processed_rows(data_dir, symbol, timeframe)
        if notify is not None:
            notify(symbol, timeframe, not raw_path.exists())
        result = update_symbol(exchange, symbol, data_dir, now_ms, timeframe=timeframe)
        if result["rows"] < before:
            raise DataShrank(f"{symbol} {timeframe}: {before} candles before the update, {result['rows']} after.")
        results.append({**result, "timeframe": timeframe, "unit": UNITS[timeframe]})
    return results


def format_results(results):
    """One plain line per dataset: symbol, timeframe, candles, last date, new candles and any problems."""
    lines = []
    for r in results:
        problems = "clean" if not r["problems"] else f"problems: {r['problems']}"
        lines.append(
            f"{r['symbol']} {r['timeframe']}: {r['rows']} {r['unit']} to {r['last_date']}, "
            f"{r['new_candles']} fetched ({problems})"
        )
    return lines


def pack_data(data_dir, out_dir):
    """Write the market-data archives for publishing.

    ``market-data-processed.tar.gz`` holds the processed CSV of every dataset
    (what the web interface reads) and ``market-data-raw.tar.gz`` the raw CSVs
    (what the next update starts from). Only those files are packed.

    Returns
    -------
    dict
        ``{"processed": path, "raw": path}``.

    Raises
    ------
    FileNotFoundError
        If a dataset's file is missing, so an incomplete archive is never written.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    members = {"processed": [], "raw": []}
    for symbol, timeframe in datasets():
        slug = symbol_to_slug(symbol)
        members["processed"].append(f"processed/{slug}_{timeframe}.csv")
        members["raw"].append(f"raw/{slug}_{timeframe}_raw.csv")
    missing = [name for names in members.values() for name in names if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot pack: missing {', '.join(missing)} in {data_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, names in members.items():
        paths[kind] = out_dir / ARCHIVES[kind]
        with tarfile.open(paths[kind], "w:gz") as archive:
            for name in names:
                archive.add(data_dir / name, arcname=name, recursive=False)
    return paths


def unpack_data(archive, data_dir):
    """Restore an archive written by ``pack_data`` into ``data_dir``.

    Every member must be a regular file named ``raw/<name>.csv`` or
    ``processed/<name>.csv``; anything else (other folders, links, paths that
    climb out of the data folder) rejects the whole archive before a single
    file is written. Each file is written next to its destination and then
    moved into place, so a reader never sees a half-written file.

    Parameters
    ----------
    archive : path-like or bytes
        The ``.tar.gz`` file, or its content.
    data_dir : pathlib.Path
        Root data folder.

    Returns
    -------
    list of str
        The member names written, e.g. ``"processed/btc_usdt_1d.csv"``.

    Raises
    ------
    ValueError
        If the archive holds anything other than the expected CSV files.
    """
    data_dir = Path(data_dir)
    source = {"fileobj": io.BytesIO(archive)} if isinstance(archive, (bytes, bytearray)) else {"name": str(archive)}
    with tarfile.open(mode="r:gz", **source) as tar:
        members = tar.getmembers()
        for member in members:
            if not member.isfile() or not MEMBER_NAME.fullmatch(member.name):
                raise ValueError(f"Unexpected entry in the data archive: {member.name!r}.")
        written = []
        for member in members:
            target = data_dir / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with tar.extractfile(member) as content, partial.open("wb") as handle:
                while chunk := content.read(1 << 20):
                    handle.write(chunk)
            os.replace(partial, target)
            written.append(member.name)
    return written


def main(argv=None, exchange=None, now_ms=None):
    """Command-line entry point. ``exchange`` and ``now_ms`` can be injected in tests.

    Returns the process exit code: 0 on success, 1 if the update failed.
    """
    parser = argparse.ArgumentParser(description="Refresh, pack or unpack the market data.")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="project data directory (default: %(default)s)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("update", help="download new candles for every dataset")
    pack = commands.add_parser("pack", help="write the market-data archives")
    pack.add_argument("--out", type=Path, required=True, help="folder to write the archives into")
    unpack = commands.add_parser("unpack", help="restore archives into the data directory")
    unpack.add_argument("archives", type=Path, nargs="+", help="archive files written by pack")
    args = parser.parse_args(argv)

    if args.command == "pack":
        for path in pack_data(args.data_dir, args.out).values():
            print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return 0
    if args.command == "unpack":
        for archive in args.archives:
            for name in unpack_data(archive, args.data_dir):
                print(f"restored {name}")
        return 0

    exchange = exchange or make_exchange()
    now_ms = exchange.milliseconds() if now_ms is None else now_ms

    def notify(symbol, timeframe, first_time):
        if first_time:
            print(f"{symbol} {timeframe}: downloading the whole history; this takes a few minutes.", flush=True)

    try:
        results = update_all(exchange, args.data_dir, now_ms, notify)
    except Exception as error:  # network errors, a refused address, or DataShrank
        print(f"Update failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print("\n".join(format_results(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
