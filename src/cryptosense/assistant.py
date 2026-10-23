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
        lines.append(f"- {key}: {facts[key]}")
    lines.append("DEDUCTIONS")
    for insight in insights:
        lines.append(f"- {insight.title}: {insight.text} (evidence: {insight.evidence})")
    return "\n".join(lines)


def build_request(question, facts, insights, history=None, model=DEFAULT_MODEL):
    """Arguments for client.messages.create: system prompt, conversation and settings."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    messages = list(history or [])
    messages.append({"role": "user", "content": question.strip()})
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": RULES + "\n" + build_fact_sheet(facts, insights),
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
    if client is None:
        return "The assistant is not configured: no API key was found."
    request = build_request(question, facts, insights, history, model)
    try:
        response = client.messages.create(**request)
    except Exception as error:  # network, authentication or rate-limit errors
        return f"The assistant is unavailable right now ({type(error).__name__}). The rest of the dashboard still works."
    return "".join(block.text for block in response.content if getattr(block, "type", "text") == "text")


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
