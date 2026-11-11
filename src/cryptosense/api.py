"""Web API for the CryptoSense interface.

Run locally with ``uvicorn cryptosense.api:app --reload`` (from the project
root). Every endpoint returns what ``cryptosense.service`` builds from the
tested analysis modules; nothing is computed here. The API is read-only: it
never fetches market data and never places trades. If the built front end
exists in ``web/dist``, it is served from the same address.

Settings (environment variables, all optional):

``CRYPTOSENSE_DATA_DIR``
    Data folder; defaults to ``data/`` in the project root.
``CRYPTOSENSE_WEB_DIR``
    Built front end; defaults to ``web/dist`` in the project root.
``CRYPTOSENSE_DATA_URL``
    Address of the published ``market-data-processed.tar.gz``. When set, the
    data folder is filled from it at start-up and re-checked every
    ``CRYPTOSENSE_DATA_SYNC_MINUTES`` minutes (default 60); see ``datasync``.
``ASSISTANT_API_URL``, ``ASSISTANT_API_KEY``, ``ASSISTANT_MODEL``
    A hosted chat model for the assistant (see ``assistant.get_hosted_client``).
    Without them the assistant uses a local Ollama server if one is running,
    and otherwise gives only its fixed, checked answers.
"""

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from cryptosense import service
from cryptosense.assistant import MAX_HISTORY_MESSAGES, get_hosted_client, get_local_client
from cryptosense.datasync import DATA_URL_ENV, DataSync, sync_minutes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_QUESTION_CHARS = 500
MAX_MESSAGE_CHARS = 4000
ASK_LIMIT = 8  # questions per client ...
ASK_WINDOW_S = 60  # ... per this many seconds
ASK_DAILY_LIMIT = 500  # questions per day across all clients, to protect a free model tier
LOCAL_CLIENT_TTL_S = 60

RangeName = Literal["1m", "3m", "1y", "all"]
PeriodName = Literal["full", "heldout"]


class RateLimiter:
    """Allow at most ``limit`` events per ``window_s`` seconds for each key.

    Old events are forgotten as time passes. ``clock`` returns the current
    time in seconds and can be replaced in tests.
    """

    def __init__(self, limit, window_s, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._events = defaultdict(deque)

    def allow(self, key):
        """Record an event for ``key`` and return True, or return False if the key is over its limit."""
        now = self.clock()
        events = self._events[key]
        while events and now - events[0] >= self.window_s:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        return True


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)


class _ClientProvider:
    """Pick the assistant's model: the hosted one if configured, else a local Ollama if running."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._local, self._checked_at = None, None

    def __call__(self):
        hosted = get_hosted_client()
        if hosted is not None:
            return hosted
        now = self.clock()
        if self._checked_at is None or now - self._checked_at >= LOCAL_CLIENT_TTL_S:
            self._local, self._checked_at = get_local_client(), now
        return self._local


def create_app(
    data_dir=None,
    web_dir=None,
    get_client=None,
    now=None,
    limiter=None,
    daily_limiter=None,
    data_sync=None,
    sync_interval_s=None,
):
    """Build the FastAPI application.

    Parameters
    ----------
    data_dir : path-like, optional
        Data folder. Defaults to ``CRYPTOSENSE_DATA_DIR`` or ``data/`` in the project root.
    web_dir : path-like, optional
        Folder with the built front end. Defaults to ``CRYPTOSENSE_WEB_DIR`` or ``web/dist``.
    get_client : callable, optional
        ``get_client() -> client or None`` for the assistant's model. Tests pass a fake.
    now : callable, optional
        ``now() -> pandas.Timestamp`` (UTC) used for the age of the data. Tests pass a fixed clock.
    limiter, daily_limiter : RateLimiter, optional
        Limits on assistant questions per client and in total.
    data_sync : DataSync, optional
        Keeps the data folder up to date from the published archive. Defaults to one built
        from ``CRYPTOSENSE_DATA_URL``, or none when that is not set. It is checked once when
        the server starts and then every ``sync_interval_s`` seconds in the background.
    sync_interval_s : float, optional
        Seconds between checks. Defaults to ``CRYPTOSENSE_DATA_SYNC_MINUTES`` (60 minutes).
    """
    data_dir = Path(data_dir or os.environ.get("CRYPTOSENSE_DATA_DIR") or PROJECT_ROOT / "data")
    web_dir = Path(web_dir or os.environ.get("CRYPTOSENSE_WEB_DIR") or PROJECT_ROOT / "web" / "dist")
    get_client = get_client or _ClientProvider()
    current_time = now or (lambda: None)
    limiter = limiter or RateLimiter(ASK_LIMIT, ASK_WINDOW_S)
    daily_limiter = daily_limiter or RateLimiter(ASK_DAILY_LIMIT, 86_400)
    data_url = os.environ.get(DATA_URL_ENV, "").strip()
    if data_sync is None and data_url:
        data_sync = DataSync(data_url, data_dir)
    interval_s = sync_minutes(os.environ) * 60 if sync_interval_s is None else sync_interval_s

    @asynccontextmanager
    async def lifespan(app):
        """Fill the data folder before serving, then keep it fresh until the server stops."""
        stop, worker = threading.Event(), None
        if data_sync is not None:
            data_sync.check_quietly()
            worker = threading.Thread(target=data_sync.run, args=(stop, interval_s), daemon=True, name="data-sync")
            worker.start()
        try:
            yield
        finally:
            stop.set()
            if worker is not None:
                worker.join(timeout=5)

    app = FastAPI(
        title="CryptoSense AI",
        description=service.DISCLAIMER,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    @app.exception_handler(service.UnknownAsset)
    async def unknown_asset(request, error):
        return JSONResponse({"detail": "Unknown asset."}, status_code=404)

    @app.exception_handler(service.NoData)
    async def no_data(request, error):
        return JSONResponse({"detail": "No data has been downloaded for this asset yet."}, status_code=404)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def symbol_of(slug):
        return service.slug_to_symbol(slug)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/meta")
    def meta():
        return {
            "disclaimer": service.DISCLAIMER,
            "ranges": list(service.CHART_RANGES),
            "default_range": service.DEFAULT_RANGE,
            "held_out_start": service.HELD_OUT_START,
            "max_question_chars": MAX_QUESTION_CHARS,
            "assistant_model_available": get_client() is not None,
        }

    @app.get("/api/assets")
    def assets():
        return service.list_assets(data_dir, current_time())

    @app.get("/api/assets/{slug}/today")
    def today(slug: str):
        return service.today(data_dir, symbol_of(slug), current_time())

    @app.get("/api/assets/{slug}/week")
    def week(slug: str):
        return service.week(data_dir, symbol_of(slug))

    @app.get("/api/assets/{slug}/prices")
    def prices(slug: str, range: RangeName = service.DEFAULT_RANGE):
        return service.prices(data_dir, symbol_of(slug), range)

    @app.get("/api/assets/{slug}/conditions")
    def conditions(slug: str):
        return service.conditions(data_dir, symbol_of(slug))

    @app.get("/api/assets/{slug}/patterns")
    def patterns(slug: str):
        return service.patterns(data_dir, symbol_of(slug))

    @app.get("/api/assets/{slug}/volatility")
    def volatility(slug: str, range: RangeName = service.DEFAULT_RANGE):
        return service.volatility(data_dir, symbol_of(slug), range)

    @app.get("/api/assets/{slug}/strategies")
    def strategies(slug: str, period: PeriodName = "full", range: RangeName = service.DEFAULT_RANGE):
        return service.strategies(data_dir, symbol_of(slug), period, range)

    @app.get("/api/risk")
    def risk(
        price: float = Query(gt=0),
        typical_move: float = Query(gt=0),
        account: float = Query(gt=0),
        risk_pct: float = Query(gt=0, le=100),
    ):
        try:
            return service.risk(price, typical_move, account, risk_pct)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/assets/{slug}/ask")
    def ask(slug: str, body: AskRequest, request: Request):
        symbol = symbol_of(slug)
        if not body.question.strip():
            raise HTTPException(status_code=422, detail="The question is empty.")
        who = request.client.host if request.client else "unknown"
        if not limiter.allow(who) or not daily_limiter.allow("all"):
            raise HTTPException(status_code=429, detail="Too many questions. Please wait a minute and try again.")
        history = [message.model_dump() for message in body.history]
        return service.answer(data_dir, symbol, body.question, history, get_client(), current_time())

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    def api_not_found(path: str):
        raise HTTPException(status_code=404, detail="Not found.")

    index = web_dir / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def front_end(path: str):
        """Serve the built front end; unknown paths get index.html so in-app links work."""
        if not index.exists():
            return JSONResponse(
                {"detail": "The front end has not been built yet. The API is under /api."}, status_code=404
            )
        target = (web_dir / path).resolve()
        if path and target.is_file() and web_dir.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(index)

    return app


app = create_app()
