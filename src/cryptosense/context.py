"""Plain-language deductions about today's market conditions.

Each insight combines measured conditions into a practical conclusion and
names the evidence it rests on. Insights describe conditions and risk; they
never predict price direction, because no tested setup predicted direction
reliably (see ROADMAP.md, Phases 5, 10 and 13).
"""

from dataclasses import dataclass

ROUND_TRIP_COST = 0.003  # 0.15% per side, as in Phases 6-13

# Thresholds for the share of a typical one-day swing taken by costs.
COST_SHARE_HIGH = 0.20
COST_SHARE_LOW = 0.10


@dataclass(frozen=True)
class Insight:
    """One deduction shown to the user."""

    title: str
    text: str
    kind: str      # "risk", "context" or "warning"
    evidence: str  # where the supporting result comes from


def describe_swing(price, daily_vol):
    """Typical one-day swing as a fraction and as a price range around `price`."""
    if price is None or daily_vol is None or price <= 0 or daily_vol <= 0:
        raise ValueError("price and daily_vol must be positive")
    import math

    return {
        "swing": daily_vol,
        "low": price * math.exp(-daily_vol),
        "high": price * math.exp(daily_vol),
    }


def derive_insights(facts):
    """Turn a dictionary of measured facts into a list of Insight objects.

    Expected keys (missing or None values simply skip the related insight):
    asset, swing_1d, vol_regime, us_open_activity, next_weekday, data_age_days.
    """
    insights = []
    asset = facts.get("asset", "")
    swing = facts.get("swing_1d")

    age = facts.get("data_age_days")
    if age is not None and age > 1:
        insights.append(Insight(
            "Data may be out of date",
            f"The latest complete day is {age} days old. Refresh the data before relying on this summary.",
            "warning",
            "data pipeline",
        ))

    if swing:
        share = ROUND_TRIP_COST / swing
        if share >= COST_SHARE_HIGH:
            insights.append(Insight(
                "Small expected range relative to costs",
                f"Trading costs (about {ROUND_TRIP_COST:.2%} for a buy and a sell) are {share:.0%} of a typical "
                "one-day swing, so a one-day trade has to move a long way just to break even.",
                "risk",
                "Phases 8 and 13: cost versus typical move",
            ))
        elif share <= COST_SHARE_LOW:
            insights.append(Insight(
                "Costs are small relative to the typical range",
                f"Trading costs are about {share:.0%} of a typical one-day swing.",
                "context",
                "Phases 8 and 13: cost versus typical move",
            ))

    if facts.get("vol_regime") == "high":
        insights.append(Insight(
            "Turbulent conditions tend to persist",
            "Volatility is higher than usual for this asset over the past year, and turbulent periods "
            "have tended to last for weeks. If you trade, a wider stop with a smaller position keeps the "
            "risk the same.",
            "risk",
            "Phases 2-4: volatility clustering and forecasting",
        ))
    elif facts.get("vol_regime") == "low":
        insights.append(Insight(
            "Calmer than usual",
            "Volatility is lower than usual for this asset over the past year. Calm periods have also "
            "tended to persist, but they can end abruptly.",
            "context",
            "Phases 2-4: volatility clustering",
        ))

    if facts.get("us_open_activity") == "high":
        scope = "" if asset == "BTC/USDT" else " This was measured on BTC; it has not been tested on this asset."
        insights.append(Insight(
            "Active US market open today",
            "Today's 09:00-11:00 New York window was unusually active. On BTC, days after highly active "
            "opens had about 30-40% larger moves (2024 onward)." + scope,
            "context",
            "Phase 13: one-day exploration (notebook 14)",
        ))

    weekday = facts.get("next_weekday")
    if weekday in ("Saturday", "Sunday"):
        scope = "" if asset == "BTC/USDT" else " This was measured on BTC; it has not been tested on this asset."
        detail = "about half the usual movement" if weekday == "Saturday" else "less movement than weekdays"
        insights.append(Insight(
            f"Tomorrow is {weekday}",
            f"On BTC, one-day holds starting on a {weekday} had {detail} from 2024 onward, so costs take a "
            "larger share of the likely move." + scope,
            "context",
            "Phase 13: one-day exploration (notebook 14)",
        ))

    insights.append(Insight(
        "No direction forecast",
        "None of the tested patterns predicted the direction of the next move reliably after costs, "
        "so this summary describes conditions and risk only.",
        "context",
        "Phases 5, 10 and 13: setup evaluations",
    ))
    return insights
