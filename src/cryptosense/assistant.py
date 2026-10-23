"""AI assistant layer: answers questions using only facts computed by the tested modules.

The language model never sees raw data and is never asked to predict prices.
It receives a fact sheet (numbers and deductions already computed and tested)
and strict rules, so it can explain and answer in plain language without
adding claims the research does not support.
"""

import os

DEFAULT_MODEL = "claude-haiku-5-5"
MAX_TOKENS = 600

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


TREND_WORDS = {"up": "rising", "down": "falling", "range": "mixed (the signals disagree)"}
VOL_WORDS = {"high": "more turbulent than usual for this asset", "low": "calmer than usual for this asset"}
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
    lines.append(f"- Price: {_money(f.get('price'))}. Change over 1 day: {_pct(f.get('change_1d'))}; "
                 f"7 days: {_pct(f.get('change_7d'))}; 30 days: {_pct(f.get('change_30d'))}.")
    lines.append(f"- 30-day low and high: {_money(f.get('low_30d'))} to {_money(f.get('high_30d'))}.")
    lines.append(f"- Trend: {TREND_WORDS.get(f.get('trend'), 'not enough history')}.")
    lines.append(f"- Turbulence: {VOL_WORDS.get(f.get('vol_regime'), 'not enough history')}.")
    if f.get("swing_1d") is not None:
        lines.append(f"- Tomorrow's typical swing: about ±{f['swing_1d']:.1%}, roughly {_money(f.get('range_low'))} "
                     f"to {_money(f.get('range_high'))}. This is the usual SIZE of a move in EITHER direction, "
                     "not a forecast of direction; larger moves do happen.")
    if f.get("forecast_track_record"):
        lines.append(f"- About that forecast: {f['forecast_track_record']}")
    if f.get("us_open_activity"):
        lines.append(f"- Today's US market open (09:00-11:00 New York) was {US_OPEN_WORDS[f['us_open_activity']]}.")
    if f.get("next_weekday"):
        lines.append(f"- Tomorrow is {f['next_weekday']}.")
    patterns = f.get("active_patterns") or []
    if patterns:
        for p in patterns:
            lines.append(f"- Pattern seen today: {p.get('description')}. This describes what has ALREADY happened, "
                         f"not what will happen next. Evidence that it matters: {p.get('evidence')}.")
    else:
        lines.append("- No researched pattern is present today.")
    lines.append("- No tested pattern predicts whether the price will go up or down after costs.")
    lines.append("DEDUCTIONS")
    for insight in insights:
        lines.append(f"- {insight.title}: {insight.text}")
    return "\n".join(lines)


import re

DIRECTIONAL_PATTERNS = [
    r"\byou should (buy|sell)\b", r"\bi (would )?recommend (buying|selling)\b",
    r"\bgood (time|moment) to (buy|sell)\b",
    r"\bexpect (a |an )?(possible )?(dip|drop|rise|rally|increase|decrease|fall|gain|decline)\b",
    r"\bexpect (the )?price to (rise|fall|drop|increase|decrease|go up|go down|dip)\b",
    r"\b(price|it|sol|btc|eth) (will|is going to|is likely to|should) (rise|fall|drop|increase|decrease|go up|go down|dip|climb)\b",
    r"\blikely to (rise|fall|drop|increase|decrease|go up|go down|dip|climb)\b",
    r"\bpossible (dip|drop|rise|rally)\b",
]
TRADE_QUESTION = re.compile(r"\b(buy|sell|enter|exit|invest|long|short|trade)\b", re.IGNORECASE)
DIRECTION_QUESTION = re.compile(
    r"\b(go(ing)? (up|down)|rise|rising|fall|falling|drop|pump|dump|bullish|bearish|moon|crash|"
    r"higher|lower|up or down|direction)\b", re.IGNORECASE)


def violates_rules(answer):
    """True if an answer predicts direction or recommends a trade."""
    text = answer.lower()
    return any(re.search(pattern, text) for pattern in DIRECTIONAL_PATTERNS)


def trade_decision_answer(facts, insights, opening=None):
    """A fixed, checked answer to buy/sell or direction questions, built only from the facts.

    Written as Markdown: a paragraph, a bullet list, then the disclaimer as its own paragraph.
    """
    f = facts
    opening = opening or (f"I can't tell you whether to buy or sell: none of the patterns tested in this "
                          f"project predicted the direction of {f.get('asset')} reliably after costs.")
    parts = [opening + " Here is what the data says to help you decide for yourself:", ""]
    if f.get("swing_1d") is not None:
        parts.append(f"- A typical move tomorrow is about ±{f['swing_1d']:.1%} "
                     f"(roughly {_money(f.get('range_low'))} to {_money(f.get('range_high'))}), in either direction.")
    parts.append(f"- Trend: {TREND_WORDS.get(f.get('trend'), 'not enough history')}; "
                 f"turbulence: {VOL_WORDS.get(f.get('vol_regime'), 'not enough history')}.")
    for insight in insights:
        if insight.kind in ("risk", "warning") or insight.title.startswith(("Tomorrow is", "Active US market")):
            parts.append(f"- {insight.title}: {insight.text}")
    parts.append("- If you do trade, decide in advance how much you can afford to lose, and use the risk helper "
                 "in the dashboard to size the position and place a stop.")
    parts.append("")
    parts.append("This is information, not financial advice.")
    return "\n".join(parts)


def build_request(question, facts, insights, history=None, model=DEFAULT_MODEL):
    """Arguments for client.messages.create: system prompt, conversation and settings."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    messages = list(history or [])
    messages.append({"role": "user", "content": question.strip()})
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": RULES + "\n" + build_plain_fact_sheet(facts, insights),
        "messages": messages,
    }


def get_client(api_key=None):
    """An Anthropic client, or None if no API key is configured.

    The key comes from the argument or the ANTHROPIC_API_KEY environment
    variable. It must never be written into the code or committed.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=key)


def ask(question, facts, insights, client, history=None, model=DEFAULT_MODEL):
    """Ask the assistant a question; returns the answer text, or a friendly error message."""
    if question and TRADE_QUESTION.search(question):
        return trade_decision_answer(facts, insights)
    if question and DIRECTION_QUESTION.search(question):
        return trade_decision_answer(facts, insights, opening=(
            f"Nobody can reliably say whether {facts.get('asset')} will go up or down tomorrow: none of the "
            "patterns tested in this project predicted the direction reliably after costs."))
    if client is None:
        return "The assistant is not configured: no API key was found."
    request = build_request(question, facts, insights, history, model)
    try:
        response = client.messages.create(**request)
    except Exception as error:  # network, authentication or rate-limit errors
        return f"The assistant is unavailable right now ({type(error).__name__}). The rest of the dashboard still works."
    answer = "".join(block.text for block in response.content if getattr(block, "type", "text") == "text")
    if violates_rules(answer):
        return ("*The assistant's answer was withheld because it suggested a price direction, which this "
                "project's research does not support.*\n\n" + trade_decision_answer(facts, insights))
    return answer


# ---------------------------------------------------------------------------
# Free, local alternative: Ollama (https://ollama.com) runs open models on
# your own computer. No API key, no cost, and nothing leaves your machine.
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"


class OllamaClient:
    """Minimal stand-in for anthropic.Anthropic that talks to a local Ollama server.

    It exposes messages.create(model, max_tokens, system, messages) and returns
    an object shaped like the Anthropic response, so ask() works unchanged.
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
            "model": OLLAMA_MODEL if model == DEFAULT_MODEL else model,
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
