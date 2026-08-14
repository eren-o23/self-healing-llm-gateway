"""Hedging: pay for a second call to escape a slow tail, and measure what that costs.

interactive.classify is the hedged class - ladder groq -> anthropic -> ollama.
Nothing here fakes a clock: providers are made slow with real awaits and the
budget is waited out for real, which is why TEST_CONFIG shortens it to 100ms.
"""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import REGISTRY

from gateway.providers import Outcome
from tests.conftest import BODY, HEADERS, failed_result, ok_result

CLASSIFY = {**HEADERS, "X-Request-Class": "interactive.classify"}
BUDGET_S = 0.1  # TEST_CONFIG's hedge_after_ms


def _hedges(outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "gateway_hedge_total",
            {"class": "interactive.classify", "outcome": outcome},
        )
        or 0.0
    )


@pytest.fixture
def slow_ladder(monkeypatch):
    """Script each provider with a delay and a result, then watch who wins.

    Cancellation is recorded rather than inferred: a losing call that keeps
    running is still billing for tokens, so "was it cancelled" is the assertion
    that matters, not "did the right value come back".
    """
    cancelled: list[str] = []
    started: list[str] = []

    def _install(plan: dict[str, tuple[float, object]]):
        async def fake_call(name, provider, payload, chaos=None):
            started.append(name)
            delay, result = plan.get(name, (0.0, None))
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            return result or ok_result(name)

        monkeypatch.setattr("gateway.router.call_provider", fake_call)
        return started, cancelled

    return _install


def test_a_fast_primary_never_fires_the_hedge(client, slow_ladder):
    """The budget is what keeps hedging honest: a normal call costs nothing extra."""
    started, _ = slow_ladder({"groq": (0.0, ok_result("groq"))})
    before = _hedges("fired")

    response = client.post("/v1/chat/completions", json=BODY, headers=CLASSIFY)

    assert response.status_code == 200
    assert started == ["groq"]
    assert _hedges("fired") == before


def test_a_slow_primary_fires_the_hedge(client, slow_ladder):
    started, _ = slow_ladder(
        {"groq": (BUDGET_S * 3, ok_result("groq")), "anthropic": (0.0, ok_result("anthropic"))}
    )
    before = _hedges("fired")

    client.post("/v1/chat/completions", json=BODY, headers=CLASSIFY)

    assert started == ["groq", "anthropic"]
    assert _hedges("fired") == before + 1


def test_the_hedge_wins_and_the_loser_is_cancelled(client, slow_ladder):
    """The acceptance criterion, and the reason the loser must be awaited.

    A cancelled task that is never awaited keeps running, and a provider halfway
    through a generation goes on billing for the tokens it already produced.
    """
    _, cancelled = slow_ladder(
        {"groq": (BUDGET_S * 5, ok_result("groq")), "anthropic": (0.0, ok_result("anthropic"))}
    )
    before = _hedges("won_by_hedge")

    response = client.post("/v1/chat/completions", json=BODY, headers=CLASSIFY)

    assert response.json()["x_gateway"]["provider"] == "anthropic"
    assert _hedges("won_by_hedge") == before + 1
    assert cancelled == ["groq"]


def test_a_primary_that_wins_late_counts_as_wasted(client, slow_ladder):
    """Fired but overtaken is the case that makes hedging cost money for nothing."""
    _, cancelled = slow_ladder(
        {
            "groq": (BUDGET_S * 1.2, ok_result("groq")),
            "anthropic": (BUDGET_S * 4, ok_result("anthropic")),
        }
    )
    before = _hedges("wasted")

    response = client.post("/v1/chat/completions", json=BODY, headers=CLASSIFY)

    assert response.json()["x_gateway"]["provider"] == "groq"
    assert _hedges("wasted") == before + 1
    assert cancelled == ["anthropic"]


def test_a_hedge_that_returns_first_but_failed_does_not_win(client, slow_ladder):
    """Finishing first by failing is not winning; the slower real answer is better."""
    slow_ladder(
        {
            "groq": (BUDGET_S * 2, ok_result("groq")),
            "anthropic": (0.0, failed_result("anthropic", Outcome.SERVER_ERROR)),
        }
    )
    before_won, before_wasted = _hedges("won_by_hedge"), _hedges("wasted")

    response = client.post("/v1/chat/completions", json=BODY, headers=CLASSIFY)

    assert response.json()["x_gateway"]["provider"] == "groq"
    assert _hedges("won_by_hedge") == before_won
    assert _hedges("wasted") == before_wasted + 1


def test_both_rungs_failing_falls_through_to_the_rest_of_the_ladder(client, slow_ladder):
    started, _ = slow_ladder(
        {
            "groq": (BUDGET_S * 2, failed_result("groq", Outcome.TIMEOUT)),
            "anthropic": (0.0, failed_result("anthropic", Outcome.SERVER_ERROR)),
        }
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=CLASSIFY)

    assert response.status_code == 200
    assert started == ["groq", "anthropic", "ollama"]
    assert response.json()["x_gateway"]["provider"] == "ollama"


def test_a_cancelled_loser_leaves_no_attempt_behind(client, slow_ladder):
    """The trail records outcomes, and a call cancelled mid-flight never had one."""
    slow_ladder(
        {"groq": (BUDGET_S * 1.2, ok_result("groq")), "anthropic": (BUDGET_S * 4, None)}
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=CLASSIFY)

    trail = response.json()["x_gateway"]["attempts"]
    assert [a["provider"] for a in trail] == ["groq"], "a cancelled call has no outcome"


def test_an_unhedged_class_races_nothing(client, slow_ladder):
    started, _ = slow_ladder({"anthropic": (BUDGET_S * 3, ok_result("anthropic"))})
    before = _hedges("fired")

    client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    assert started == ["anthropic"]
    assert _hedges("fired") == before
