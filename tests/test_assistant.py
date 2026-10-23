from types import SimpleNamespace

import pytest

from cryptosense.assistant import DEFAULT_MODEL, RULES, ask, build_fact_sheet, build_request, get_client
from cryptosense.context import derive_insights

FACTS = {"asset": "SOL/USDT", "price": 182.4, "swing_1d": 0.031, "vol_regime": "high",
         "next_weekday": "Saturday", "data_age_days": 0}


class FakeClient:
    """Stands in for anthropic.Anthropic and records each request."""

    def __init__(self, reply="Calm answer.", error=None):
        self.requests, self.reply, self.error = [], reply, error
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def test_fact_sheet_contains_facts_and_deductions():
    sheet = build_fact_sheet(FACTS, derive_insights(FACTS))
    assert "swing_1d: 0.031" in sheet
    assert "Tomorrow is Saturday" in sheet
    assert "evidence:" in sheet


def test_request_has_rules_facts_and_question():
    request = build_request("How much could SOL move tomorrow?", FACTS, derive_insights(FACTS))
    assert request["model"] == DEFAULT_MODEL
    assert request["system"].startswith(RULES)
    assert "price: 182.4" in request["system"]
    assert request["messages"][-1] == {"role": "user", "content": "How much could SOL move tomorrow?"}


def test_rules_forbid_direction_and_advice():
    assert "Never predict whether the price will go up or down" in RULES
    assert "never tell the user to buy or sell" in RULES


def test_history_is_kept_in_order():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    request = build_request("And tomorrow?", FACTS, [], history)
    assert request["messages"][:2] == history
    assert history == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        build_request("   ", FACTS, [])


def test_ask_returns_model_text():
    client = FakeClient(reply="SOL typically swings about 3% in a day.")
    assert ask("Range?", FACTS, [], client) == "SOL typically swings about 3% in a day."
    assert len(client.requests) == 1


def test_ask_handles_api_errors():
    answer = ask("Range?", FACTS, [], FakeClient(error=RuntimeError("down")))
    assert "unavailable" in answer and "RuntimeError" in answer


def test_ask_without_client():
    assert "not configured" in ask("Range?", FACTS, [], None)


def test_get_client_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_client() is None


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.data


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        return self.response


def test_ollama_client_builds_local_request_and_parses_reply():
    from cryptosense.assistant import OLLAMA_MODEL, OllamaClient

    session = FakeSession(FakeResponse({"message": {"role": "assistant", "content": "Local answer."}}))
    client = OllamaClient(session=session)
    answer = ask("Range?", FACTS, derive_insights(FACTS), client)

    url, payload = session.calls[0]
    assert answer == "Local answer."
    assert url.endswith("/api/chat")
    assert payload["model"] == OLLAMA_MODEL
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system" and payload["messages"][0]["content"].startswith(RULES)
    assert payload["messages"][-1] == {"role": "user", "content": "Range?"}


def test_ollama_errors_are_handled_by_ask():
    from cryptosense.assistant import OllamaClient

    client = OllamaClient(session=FakeSession(FakeResponse({}, status=500)))
    assert "unavailable" in ask("Range?", FACTS, [], client)


def test_get_local_client_returns_none_when_no_server():
    from cryptosense.assistant import get_local_client

    assert get_local_client("http://127.0.0.1:9/api/chat") is None
