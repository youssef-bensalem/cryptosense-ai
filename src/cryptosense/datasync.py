"""Keep a hosted copy of the market data up to date from the published archive.

The "Refresh market data" GitHub workflow publishes
``market-data-processed.tar.gz`` every day. A host has no data folder of its
own, so the web API downloads that archive when it starts and checks again
from time to time. Set ``CRYPTOSENSE_DATA_URL`` to the archive's address to
turn this on; without it nothing is downloaded and the local ``data/`` folder
is used as it is.

Only the archive is read (one public file over https); nothing is uploaded.
The archive is checked by ``refresh.unpack_data`` before any file is written,
and a failed check leaves the data already on disk untouched.
"""

import hashlib
import logging
from pathlib import Path

from cryptosense.refresh import unpack_data

DATA_URL_ENV = "CRYPTOSENSE_DATA_URL"
SYNC_MINUTES_ENV = "CRYPTOSENSE_DATA_SYNC_MINUTES"
DEFAULT_SYNC_MINUTES = 60
MIN_SYNC_MINUTES = 5
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
VALIDATOR_HEADERS = ("ETag", "Last-Modified", "Content-Length")

log = logging.getLogger(__name__)


def sync_minutes(env):
    """Minutes between checks, from ``CRYPTOSENSE_DATA_SYNC_MINUTES`` (default 60, at least 5)."""
    raw = env.get(SYNC_MINUTES_ENV, "").strip()
    try:
        minutes = float(raw) if raw else DEFAULT_SYNC_MINUTES
    except ValueError:
        minutes = DEFAULT_SYNC_MINUTES
    return max(minutes, MIN_SYNC_MINUTES)


class DataSync:
    """Download the published data archive into ``data_dir`` whenever it has changed.

    Parameters
    ----------
    url : str
        Address of ``market-data-processed.tar.gz``; must start with ``https://``.
    data_dir : path-like
        Root data folder to unpack into.
    session : object, optional
        Something with ``head`` and ``get`` like ``requests.Session``; tests pass a fake.
    timeout : float, default 120
        Seconds to wait for the server.
    """

    def __init__(self, url, data_dir, session=None, timeout=120):
        if not str(url).startswith("https://"):
            raise ValueError(f"{DATA_URL_ENV} must start with https://; got {url!r}.")
        if session is None:
            import requests

            session = requests.Session()
        self.url, self.data_dir, self.session, self.timeout = url, Path(data_dir), session, timeout
        self._fingerprint = None  # the server's description of the archive last unpacked
        self._digest = None  # SHA-256 of the archive last unpacked

    def _remote_fingerprint(self):
        """What the server says about the archive without sending it, or None if it says nothing useful."""
        try:
            reply = self.session.head(self.url, allow_redirects=True, timeout=self.timeout)
            reply.raise_for_status()
        except Exception:
            return None
        values = tuple(reply.headers.get(name) for name in VALIDATOR_HEADERS)
        return values if any(values) else None

    def check(self):
        """Fetch and unpack the archive if it changed since the last successful check.

        Returns
        -------
        str
            ``"updated"`` if files were written, ``"unchanged"`` otherwise.

        Raises
        ------
        Exception
            If the download fails, the archive is too large, or it holds
            anything other than the expected CSV files. Nothing is written then.
        """
        fingerprint = self._remote_fingerprint()
        if fingerprint is not None and fingerprint == self._fingerprint:
            return "unchanged"
        reply = self.session.get(self.url, allow_redirects=True, timeout=self.timeout)
        reply.raise_for_status()
        content = reply.content
        if len(content) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"The data archive is {len(content)} bytes, above the {MAX_ARCHIVE_BYTES} limit.")
        digest = hashlib.sha256(content).hexdigest()
        if digest == self._digest:
            self._fingerprint = fingerprint
            return "unchanged"
        written = unpack_data(content, self.data_dir)
        self._fingerprint, self._digest = fingerprint, digest
        log.info("Market data updated: %d files from %s", len(written), self.url)
        return "updated"

    def check_quietly(self):
        """Run ``check`` and return its result, or ``"failed"`` (logged) instead of raising."""
        try:
            return self.check()
        except Exception as error:
            log.warning("Market data check failed (%s: %s); keeping the data on disk.", type(error).__name__, error)
            return "failed"

    def run(self, stop, interval_s):
        """Check every ``interval_s`` seconds until ``stop`` (a ``threading.Event``) is set."""
        while not stop.wait(interval_s):
            self.check_quietly()
