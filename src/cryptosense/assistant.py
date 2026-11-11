"""AI assistant layer: answers questions using only facts computed by the tested modules.

The language model never sees raw data and is never asked to predict prices.
It receives a fact sheet (numbers and deductions already computed and tested)
and strict rules, so it can explain and answer in plain language without
adding claims the research does not support.
"""

import os
import re

from cryptosense.explain import TREND_WORDS, VOLATILITY_WORDS as VOL_WORDS

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
DEFAULT_MODEL = OLLAMA_MODEL
MAX_TOKENS = 600
MAX_HISTORY_MESSAGES = 12  # the last 6 question-and-answer pairs sent to the model

RULES = """You are the CryptoSense assistant. You help a non-technical user understand the current market conditions shown in the dashboard.

Rules:
1. Use ONLY the facts in the fact sheet below. If the answer is not in the fact sheet, say that the dashboard does not have that information.
2. Never predict whether the price will go up or down, and never tell the user to buy or sell. The project's research found no pattern that predicts direction reliably after costs.
3. When you mention a number, keep its meaning: a "typical swing" is a typical size of move, not a forecast of direction; larger moves do happen.
4. Always keep stated uncertainty and limitations (for example "measured on BTC only") when they are relevant.
5. Use plain language, short paragraphs, and no jargon. Explain any necessary term in a few words.
6. End answers about trading decisions with: "This is information, not financial advice."
"""


def build_fact_sheet(facts, insights):
    """Format computed facts and deductions as plain text for the model."""
    lines = ["FACT SHEET"]
    for key in sorted(facts):
        value = facts[key]
        if isinstance(value, float):
            value = f"{value:.6g}"
        lines.append(f"- {key}: {value}")
    lines.append("DEDUCTIONS")
    for insight in insights:
        lines.append(f"- {insight.title}: {insight.text} (evidence: {insight.evidence})")
    return "\n".join(lines)


US_OPEN_WORDS = {"high": "unusually active", "middle": "about normal", "low": "quiet"}


def _pct(value):
    return "unknown" if value is None else f"{value:+.1%}"


def _money(value):
    """Prices are in USDT; no "$" sign, because Streamlit treats text between two "$" as a math formula."""
    return "unknown" if value is None else f"{value:,.2f} USDT"


def build_plain_fact_sheet(facts, insights):
    """The fact sheet written as plain sentences, so a small model does not have to interpret raw values."""
    f = facts
    lines = ["FACTS (computed by tested code from the user's data)"]
    lines.append(f"- Asset: {f.get('asset')}. Data up to {f.get('data_date')} ({f.get('data_age_days')} day(s) old).")
    lines.append(
        f"- Price: {_money(f.get('price'))}. Change over 1 day: {_pct(f.get('change_1d'))}; "
        f"7 days: {_pct(f.get('change_7d'))}; 30 days: {_pct(f.get('change_30d'))}."
    )
    lines.append(f"- 30-day low and high: {_money(f.get('low_30d'))} to {_money(f.get('high_30d'))}.")
    lines.append(f"- Trend: {TREND_WORDS.get(f.get('trend'), 'not enough history')}.")
    lines.append(f"- Turbulence: {VOL_WORDS.get(f.get('vol_regime'), 'not enough history')}.")
    if f.get("swing_1d") is not None:
        lines.append(
            f"- Tomorrow's typical swing: about ±{f['swing_1d']:.1%}, roughly {_money(f.get('range_low'))} "
            f"to {_money(f.get('range_high'))}. This is the usual SIZE of a move in EITHER direction, "
            "not a forecast of direction; larger moves do happen."
        )
    if f.get("forecast_track_record"):
        lines.append(f"- About that forecast: {f['forecast_track_record']}")
    if f.get("us_open_activity"):
        lines.append(f"- Today's US market open (09:00-11:00 New York) was {US_OPEN_WORDS[f['us_open_activity']]}.")
    if f.get("next_weekday"):
        lines.append(f"- Tomorrow is {f['next_weekday']}.")
    patterns = f.get("active_patterns") or []
    if patterns:
        for p in patterns:
            lines.append(
                f"- Pattern seen today: {p.get('description')}. This describes what has ALREADY happened, "
                f"not what will happen next. Evidence that it matters: {p.get('evidence')}."
            )
    else:
        lines.append("- No researched pattern is present today.")
    lines.append("- No tested pattern predicts whether the price will go up or down after costs.")
    lines.append("DEDUCTIONS")
    for insight in insights:
        lines.append(f"- {insight.title}: {insight.text}")
    return "\n".join(lines)


DIRECTIONAL_PATTERNS = [
    r"\byou should (buy|sell)\b",
    r"\bi (would )?recommend (buying|selling)\b",
    r"\bgood (time|moment) to (buy|sell)\b",
    r"\bexpect (a |an )?(possible )?(dip|drop|rise|rally|increase|decrease|fall|gain|decline)\b",
    r"\bexpect (the )?price to (rise|fall|drop|increase|decrease|go up|go down|dip)\b",
    r"\b(price|it|sol|btc|eth) (will|is going to|is likely to|should) (rise|fall|drop|increase|decrease|go up|go down|dip|climb)\b",
    r"\blikely to (rise|fall|drop|increase|decrease|go up|go down|dip|climb)\b",
    r"\bpossible (dip|drop|rise|rally)\b",
]

# Sentence-level checks, used in addition to the exact phrases above.
_MOVE = (
    r"(rise|rises|rising|climb|climbs|climbing|go up|goes up|going up|increase|increases|rally|rallies|"
    r"rebound|rebounds|recover|recovers|surge|surges|fall|falls|falling|drop|drops|dropping|go down|"
    r"goes down|going down|decline|declines|dip|dips|decrease|decreases|crash|crashes|plunge|plunges|"
    r"slide|slides|sink|sinks|higher|lower|upward|upwards|downward|downwards)"
)
_FUTURE = (
    r"(will|would|could|may|might|should|is going to|are going to|likely|expected to|probably|"
    r"poised to|set to|about to|tends? to|looks? (set|ready) to)"
)
FUTURE_MOVE = re.compile(rf"\b{_FUTURE}\b[^.!?\n]*?\b{_MOVE}\b")
STANCE = re.compile(r"\b(bullish|bearish)\b")
TRADE_WORD = re.compile(
    r"\b(buy|buying|sell|selling|go long|going long|go short|going short|"
    r"enter|entering|exit|exiting)\b"
)
ADVICE_WORD = re.compile(
    r"\b(recommend|recommended|suggest|consider|advise|advisable|makes sense|"
    r"worth|wise|smart|good idea|best to|better to|you should|you could|you might)\b"
)
# A sentence about these is about the size of moves or risk, not the direction of the price.
NOT_ABOUT_PRICE = re.compile(r"(volatil|turbulen|swing|risk|cost|range|activity|volume|\bstops?\b|uncertain)")
# A sentence with one of these says that direction is unknown rather than predicting it.
HEDGE = re.compile(
    r"(\bwhether\b|either direction|up or down|rise or fall|\bcannot\b|\bcan't\b|\bcan not\b|"
    r"\bnobody\b|\bno one\b|not possible|impossible|\bno (tested |reliable )?pattern|"
    r"not a forecast|not a prediction|(does not|doesn't|do not|don't|did not|didn't) predict|"
    r"\bunknown\b|none of the)"
)


def _sentences(text):
    """Split text into sentences and list items."""
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def violates_rules(answer):
    """True if an answer predicts direction or recommends a trade.

    An answer is flagged if it contains one of ``DIRECTIONAL_PATTERNS``, or if
    one of its sentences (a) pairs a word about the future with a price-move
    word ("could climb", "may go up"), (b) calls the market bullish or
    bearish, or (c) pairs an advice word with a trade word ("buying makes
    sense"). For (a)-(c), a sentence is skipped when it says direction is
    unknown (``HEDGE``); for (a), also when it is about the size of moves or
    risk rather than the price (``NOT_ABOUT_PRICE``).
    """
    text = answer.lower()
    if any(re.search(pattern, text) for pattern in DIRECTIONAL_PATTERNS):
        return True
    for sentence in _sentences(text):
        if HEDGE.search(sentence):
            continue
        if STANCE.search(sentence):
            return True
        if TRADE_WORD.search(sentence) and ADVICE_WORD.search(sentence):
            return True
        if FUTURE_MOVE.search(sentence) and not NOT_ABOUT_PRICE.search(sentence):
            return True
    return False


TRADE_QUESTION = re.compile(
    r"\b(buy|buying|sell|selling|invest|investing)\b"
    r"|\b(go|going|went) (long|short)\b"
    r"|\b(should|shall|can|could|do|would|must|may)\s+(i|we)\b[^.!?]*\b(enter|exit|trade|get in|get out|take profit)\b"
    r"|\btime to (enter|exit|trade|get in|get out|take profit)\b",
    re.IGNORECASE,
)
DIRECTION_QUESTION = re.compile(
    r"\b(bullish|bearish|pump|pumping|dump|dumping|moon|mooning|crash|crashing|up or down|higher or lower|"
    r"(which|what|price) direction|direction of)\b",
    re.IGNORECASE,
)
FUTURE_MOVE_QUESTION = re.compile(
    r"\b(will|won't|going to|gonna|expect|expected|likely|about to|could|might|may|should|tomorrow|next)\b[^.!?]*?"
    rf"\b{_MOVE}\b|\b{_MOVE}\b[^.!?]*?\b(tomorrow|next (day|week|month)|soon)\b",
    re.IGNORECASE,
)


def is_trade_question(question):
    """True if a question asks whether to buy, sell, invest or take a trade."""
    return bool(question and TRADE_QUESTION.search(question))


def is_direction_question(question):
    """True if a question asks which way the price will move.

    Either it uses a direction word such as "bullish" or "up or down", or it
    pairs a word about the future with a price-move word ("will it rise") and
    is not about the size of moves or risk ("could volatility rise").
    Questions about the past ("why did the price drop yesterday") are not
    direction questions.
    """
    if not question:
        return False
    if DIRECTION_QUESTION.search(question):
        return True
    return bool(FUTURE_MOVE_QUESTION.search(question)) and not NOT_ABOUT_PRICE.search(question.lower())


def trade_decision_answer(facts, insights, opening=None):
    """A fixed, checked answer to buy/sell or direction questions, built only from the facts.

    Written as Markdown: a paragraph, a bullet list, then the disclaimer as its own paragraph.
    """
    f = facts
    opening = opening or (
        f"I can't tell you whether to buy or sell: none of the patterns tested in this "
        f"project predicted the direction of {f.get('asset')} reliably after costs."
    )
    parts = [opening + " Here is what the data says to help you decide for yourself:", ""]
    if f.get("swing_1d") is not None:
        parts.append(
            f"- A typical move tomorrow is about ±{f['swing_1d']:.1%} "
            f"(roughly {_money(f.get('range_low'))} to {_money(f.get('range_high'))}), in either direction."
        )
    parts.append(
        f"- Trend: {TREND_WORDS.get(f.get('trend'), 'not enough history')}; "
        f"turbulence: {VOL_WORDS.get(f.get('vol_regime'), 'not enough history')}."
    )
    for insight in insights:
        if insight.kind in ("risk", "warning") or insight.title.startswith(("Tomorrow is", "Active US market")):
            parts.append(f"- {insight.title}: {insight.text}")
    parts.append(
        "- If you do trade, decide in advance how much you can afford to lose, and use the risk helper "
        "in the dashboard to size the position and place a stop."
    )
    parts.append("")
    parts.append("This is information, not financial advice.")
    return "\n".join(parts)


def build_request(question, facts, insights, history=None, model=DEFAULT_MODEL):
    """Arguments for client.messages.create: system prompt, conversation and settings.

    Only the last ``MAX_HISTORY_MESSAGES`` messages of ``history`` are sent, so a
    long chat does not overflow a small local model.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    messages = list(history or [])[-MAX_HISTORY_MESSAGES:]
    messages.append({"role": "user", "content": question.strip()})
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": RULES + "\n" + build_plain_fact_sheet(facts, insights),
        "messages": messages,
    }


def ask(question, facts, insights, client, history=None, model=DEFAULT_MODEL):
    """Ask the assistant a question; returns the answer text, or a friendly error message."""
    if is_trade_question(question):
        return trade_decision_answer(facts, insights)
    if is_direction_question(question):
        return trade_decision_answer(
            facts,
            insights,
            opening=(
                f"Nobody can reliably say whether {facts.get('asset')} will go up or down tomorrow: none of the "
                "patterns tested in this project predicted the direction reliably after costs."
            ),
        )
    if client is None:
        return (
            "The assistant is not configured: no local model is running. "
            "Start Ollama and reload the page; the rest of the dashboard still works."
        )
    request = build_request(question, facts, insights, history, model)
    try:
        response = client.messages.create(**request)
    except Exception as error:  # the local server is unreachable, slow or returned an error
        return (
            f"The assistant is unavailable right now ({type(error).__name__}). The rest of the dashboard still works."
        )
    answer = "".join(block.text for block in response.content if getattr(block, "type", "text") == "text")
    if violates_rules(answer):
        return (
            "*The assistant's answer was withheld because it suggested a price direction, which this "
            "project's research does not support.*\n\n" + trade_decision_answer(facts, insights)
        )
    return answer


# ---------------------------------------------------------------------------
# The model: Ollama (https://ollama.com) runs open models on your own
# computer. No API key, no cost, and nothing leaves your machine.
# ---------------------------------------------------------------------------


class OllamaClient:
    """Client for a local Ollama server.

    It exposes messages.create(model, max_tokens, system, messages) and returns
    an object whose ``content`` is a list of text blocks, which is what ask() reads.
    """

    def __init__(self, url=OLLAMA_URL, timeout=120, session=None):
        import requests

        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.messages = self

    def create(self, model, max_tokens, system, messages):
        from types import SimpleNamespace

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + list(messages),
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        reply = self.session.post(self.url, json=payload, timeout=self.timeout)
        reply.raise_for_status()
        text = reply.json()["message"]["content"]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def get_local_client(url=OLLAMA_URL):
    """An OllamaClient if a local Ollama server answers, otherwise None."""
    import requests

    try:
        requests.get(url.rsplit("/api/", 1)[0] + "/api/tags", timeout=2).raise_for_status()
    except requests.RequestException:
        return None
    return OllamaClient(url)


# ---------------------------------------------------------------------------
# Hosted alternative: any chat service with an OpenAI-compatible
# "chat completions" endpoint (many free tiers offer one, and Ollama does
# too under /v1). The address, key and model come from environment
# variables, so the key is never written into the code.
# ---------------------------------------------------------------------------

CHAT_URL_ENV = "ASSISTANT_API_URL"  # e.g. https://<provider>/v1/chat/completions
CHAT_KEY_ENV = "ASSISTANT_API_KEY"
CHAT_MODEL_ENV = "ASSISTANT_MODEL"


class ChatAPIClient:
    """Client for an OpenAI-compatible chat-completions endpoint.

    It exposes messages.create(model, max_tokens, system, messages) like
    ``OllamaClient``, so ask() works unchanged. The model named at
    construction is always used; the ``model`` argument of ``create`` is
    ignored. The API key is sent only in the request header and is never
    included in an error message.
    """

    def __init__(self, url, api_key, model, timeout=60, session=None):
        import requests

        if not url or not api_key or not model:
            raise ValueError("url, api_key and model are all required")
        self.url = url
        self.model = model
        self.timeout = timeout
        self._api_key = api_key
        self.session = session or requests.Session()
        self.messages = self

    def create(self, model, max_tokens, system, messages):
        from types import SimpleNamespace

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + list(messages),
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        reply = self.session.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        reply.raise_for_status()
        text = reply.json()["choices"][0]["message"]["content"] or ""
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def get_hosted_client(env=None):
    """A ``ChatAPIClient`` configured from the environment, or None if it is not fully configured.

    Reads ``ASSISTANT_API_URL``, ``ASSISTANT_API_KEY`` and ``ASSISTANT_MODEL``
    from ``env`` (default: the process environment). All three are required.
    """
    env = os.environ if env is None else env
    url, key, model = (env.get(name, "").strip() for name in (CHAT_URL_ENV, CHAT_KEY_ENV, CHAT_MODEL_ENV))
    if not (url and key and model):
        return None
    return ChatAPIClient(url, key, model)
