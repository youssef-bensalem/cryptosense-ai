from types import SimpleNamespace

import pytest

from cryptosense.assistant import (
    DEFAULT_MODEL, RULES, ask, build_fact_sheet, build_plain_fact_sheet, build_request,
    trade_decision_answer, violates_rules,
)
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
    assert "182.40 USDT" in request["system"]
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


USER_SEEN_ANSWER = ("Since the pattern is a pullback and the current trend is a range, "
                    "we can expect a possible dip in price tomorrow.")


def test_plain_fact_sheet_is_readable():
    facts = {**FACTS, "change_30d": -0.0310367, "range_low": 176.8, "range_high": 188.1,
             "active_patterns": [{"description": "a short-term dip while the longer-term trend is rising",
                                  "evidence": "Weak evidence"}]}
    sheet = build_plain_fact_sheet(facts, derive_insights(facts))
    assert "30 days: -3.1%" in sheet
    assert "176.80 USDT to 188.10 USDT" in sheet
    assert "ALREADY happened" in sheet
    assert "EITHER direction" in sheet


@pytest.mark.parametrize("answer", [
    USER_SEEN_ANSWER,
    "You should buy SOL now.",
    "SOL will rise tomorrow.",
    "The price is likely to fall.",
    "Now looks like a good time to buy.",
])
def test_guard_flags_directional_answers(answer):
    assert violates_rules(answer)


@pytest.mark.parametrize("answer", [
    "I can't tell you whether to buy or sell.",
    "A typical move tomorrow is about 3%, in either direction.",
    "Turbulent periods have tended to persist.",
])
def test_guard_allows_safe_answers(answer):
    assert not violates_rules(answer)


def test_directional_model_answer_is_withheld():
    answer = ask("What about tomorrow?", FACTS, derive_insights(FACTS), FakeClient(reply=USER_SEEN_ANSWER))
    assert "withheld" in answer
    assert "possible dip" not in answer
    assert answer.endswith("This is information, not financial advice.")


@pytest.mark.parametrize("question", ["Should I buy SOL now?", "Is it time to sell?", "should i enter a trade"])
def test_trade_questions_get_the_checked_answer_without_the_model(question):
    client = FakeClient(reply="You should buy SOL now.")
    answer = ask(question, FACTS, derive_insights(FACTS), client)
    assert client.requests == []
    assert not violates_rules(answer)
    assert "can't tell you whether to buy or sell" in answer
    assert answer.endswith("This is information, not financial advice.")


def test_trade_answer_uses_facts():
    facts = {**FACTS, "range_low": 176.8, "range_high": 188.1}
    answer = trade_decision_answer(facts, derive_insights(facts))
    assert "±3.1%" in answer and "176.80 USDT" in answer


def test_no_dollar_signs_reach_the_screen():
    facts = {**FACTS, "range_low": 176.8, "range_high": 188.1}
    insights = derive_insights(facts)
    assert "$" not in build_plain_fact_sheet(facts, insights)
    assert "$" not in trade_decision_answer(facts, insights)


def test_disclaimer_is_its_own_paragraph():
    answer = trade_decision_answer(FACTS, derive_insights(FACTS))
    assert answer.endswith("\n\nThis is information, not financial advice.")


@pytest.mark.parametrize("question", ["Do you think SOL will go up tomorrow?", "Is SOL going to crash?",
                                      "will it be higher or lower tomorrow", "Is this bullish?"])
def test_direction_questions_get_the_checked_answer_without_the_model(question):
    client = FakeClient(reply="SOL will rise tomorrow.")
    answer = ask(question, FACTS, derive_insights(FACTS), client)
    assert client.requests == []
    assert answer.startswith("Nobody can reliably say whether SOL/USDT will go up or down tomorrow")
    assert not violates_rules(answer)


def test_ordinary_questions_still_reach_the_model():
    client = FakeClient(reply="Turbulence is lower than usual.")
    assert ask("Why is the expected range small?", FACTS, derive_insights(FACTS), client) == "Turbulence is lower than usual."
    assert len(client.requests) == 1


def test_trade_answer_mentions_weekend_context():
    facts = {**FACTS, "vol_regime": "low"}
    answer = trade_decision_answer(facts, derive_insights(facts))
    assert "Tomorrow is Saturday" in answer
