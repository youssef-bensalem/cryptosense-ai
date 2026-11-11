"""Question routing, the answer guard and the history cap of the assistant."""
from types import SimpleNamespace

import pytest

from cryptosense.assistant import (
    MAX_HISTORY_MESSAGES, VOL_WORDS, TREND_WORDS, ask, build_request, is_direction_question,
    is_trade_question, trade_decision_answer, violates_rules,
)
from cryptosense.context import derive_insights
from cryptosense import explain

FACTS = {"asset": "SOL/USDT", "price": 182.4, "swing_1d": 0.031, "vol_regime": "high",
         "next_weekday": "Saturday", "data_age_days": 0}


class FakeClient:
    def __init__(self, reply="Calm answer."):
        self.requests, self.reply = [], reply
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


ORDINARY_QUESTIONS = [
    "How long has this trend lasted?",
    "Is volatility higher than usual?",
    "What does typical swing mean in short?",
    "Why did the price drop yesterday?",
    "What is the exit of the US open window?",
    "Is the market calmer or lower risk today?",
    "Could volatility rise tomorrow?",
    "How much could SOL move tomorrow?",
    "What was the 30-day high?",
    "What is a stop?",
]


@pytest.mark.parametrize("question", ORDINARY_QUESTIONS)
def test_ordinary_questions_are_not_trade_or_direction_questions(question):
    assert not is_trade_question(question)
    assert not is_direction_question(question)


@pytest.mark.parametrize("question", ORDINARY_QUESTIONS)
def test_ordinary_questions_reach_the_model(question):
    client = FakeClient(reply="A plain explanation.")
    assert ask(question, FACTS, derive_insights(FACTS), client) == "A plain explanation."
    assert len(client.requests) == 1


@pytest.mark.parametrize("question", [
    "Should I buy SOL now?", "Is it time to sell?", "should i enter a trade", "Is selling a good idea?",
    "Should we get out now?", "Is it a good moment to go long?", "Can I invest 500 USDT today?",
    "Is it time to take profit?",
])
def test_trade_questions(question):
    assert is_trade_question(question)


@pytest.mark.parametrize("question", [
    "Do you think SOL will go up tomorrow?", "Is SOL going to crash?", "will it be higher or lower tomorrow",
    "Is this bullish?", "Will SOL rise?", "Is it going up tomorrow?", "Will the price be lower next week?",
    "Which direction is the price heading?", "Up or down?",
])
def test_direction_questions(question):
    assert is_direction_question(question)


def test_empty_questions_are_neither():
    for question in ["", None]:
        assert not is_trade_question(question)
        assert not is_direction_question(question)


@pytest.mark.parametrize("answer", [
    "I think SOL could climb tomorrow.",
    "The outlook is bullish, so buying now makes sense.",
    "Prices may go up from here.",
    "SOL will probably rise.",
    "Consider selling before the weekend.",
    "It is worth entering now.",
    "The trend is rising.\nIt might drop soon though.",
    "Momentum looks bearish.",
])
def test_guard_flags_reworded_directional_answers(answer):
    assert violates_rules(answer)


@pytest.mark.parametrize("answer", [
    "Turbulence is lower than usual.",
    "Volatility could rise after an active US open.",
    "The price has fallen 3.1% over 30 days.",
    "A wider stop with a smaller position keeps the risk the same.",
    "Larger moves do happen, and prices can jump past a stop.",
    "The 30-day high is 190.00 USDT, which is higher than today's price.",
    "The trend is rising, meaning the price is above its 100-day average.",
    "Costs take a larger share of the likely move on Saturdays.",
    "I can't say if this is bullish or bearish.",
    "The dashboard does not have that information.",
    "Calm periods can end abruptly.",
])
def test_guard_allows_descriptions_of_conditions_and_risk(answer):
    assert not violates_rules(answer)


@pytest.mark.parametrize("vol_regime", ["high", "low", None])
@pytest.mark.parametrize("weekday", ["Saturday", "Sunday", "Monday"])
def test_fixed_answers_always_pass_the_guard(vol_regime, weekday):
    facts = {**FACTS, "vol_regime": vol_regime, "next_weekday": weekday, "us_open_activity": "high",
             "range_low": 176.8, "range_high": 188.1, "data_age_days": 3}
    insights = derive_insights(facts)
    assert not violates_rules(trade_decision_answer(facts, insights))
    assert not violates_rules(ask("Will it go up?", facts, insights, None))


def test_reworded_directional_model_answer_is_withheld():
    client = FakeClient(reply="I think SOL could climb tomorrow.")
    answer = ask("What about tomorrow?", FACTS, derive_insights(FACTS), client)
    assert "withheld" in answer and "could climb" not in answer


def test_missing_client_message_names_the_local_model_not_an_api_key():
    answer = ask("What is a stop?", FACTS, [], None)
    assert "not configured" in answer and "Ollama" in answer
    assert "API key" not in answer


def test_history_is_capped_to_the_most_recent_messages():
    history = []
    for i in range(20):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    request = build_request("latest?", FACTS, [], history)
    assert len(request["messages"]) == MAX_HISTORY_MESSAGES + 1
    assert request["messages"][0] == {"role": "user", "content": "q14"}
    assert request["messages"][-1] == {"role": "user", "content": "latest?"}
    assert len(history) == 40


def test_short_history_is_sent_whole():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    assert build_request("And tomorrow?", FACTS, [], history)["messages"][:2] == history


def test_condition_words_have_one_source():
    assert TREND_WORDS is explain.TREND_WORDS
    assert VOL_WORDS is explain.VOLATILITY_WORDS
