"""The ladder walk: what gets tried, what gets skipped, and what the caller is told.

These go through the HTTP boundary rather than calling route() directly. The
interesting behaviour is what a client actually receives when providers fail
under it, and the attempts trail is part of the contract, not an internal.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from gateway import breaker, health
from gateway.providers import Outcome
from tests.conftest import ADMIN_HEADERS, BODY, HEADERS, failed_result, ok_result

CHAT = {**HEADERS}  # interactive.chat: anthropic -> openai -> groq -> ollama


def _attempts(response) -> list[dict]:
    body = response.json()
    return body.get("x_gateway", {})["attempts"]


def _failovers(**labels) -> float:
    return REGISTRY.get_sample_value("gateway_failovers_total", labels) or 0.0


# --- failover -----------------------------------------------------------------


def test_a_healthy_first_rung_is_the_only_one_called(client, stub_ladder):
    calls = stub_ladder({})

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 200
    assert calls == ["anthropic"]


def test_a_trippable_failure_moves_to_the_next_rung(client, stub_ladder):
    """The whole point: the caller never sees anthropic's server error."""
    calls = stub_ladder({"anthropic": failed_result("anthropic", Outcome.SERVER_ERROR)})

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 200
    assert response.json()["x_gateway"]["provider"] == "openai"
    assert calls == ["anthropic", "openai"]


def test_it_keeps_walking_until_something_answers(client, stub_ladder):
    calls = stub_ladder(
        {
            "anthropic": failed_result("anthropic", Outcome.SERVER_ERROR),
            "openai": failed_result("openai", Outcome.RATE_LIMIT),
            "groq": failed_result("groq", Outcome.TIMEOUT),
        }
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 200
    assert calls == ["anthropic", "openai", "groq", "ollama"]
    assert response.json()["x_gateway"]["provider"] == "ollama"


def test_every_rung_shows_up_in_the_attempts_trail(client, stub_ladder):
    stub_ladder({"anthropic": failed_result("anthropic", Outcome.TIMEOUT)})

    attempts = _attempts(client.post("/v1/chat/completions", json=BODY, headers=CHAT))

    assert [a["provider"] for a in attempts] == ["anthropic", "openai"]
    assert attempts[0]["outcome"] == "timeout"
    assert attempts[1]["outcome"] == "ok"


def test_a_hop_is_counted_with_both_ends_and_a_reason(client, stub_ladder):
    labels = {
        "from": "anthropic",
        "to": "openai",
        "class": "interactive.chat",
        "reason": "server_error",
    }
    before = _failovers(**labels)
    stub_ladder({"anthropic": failed_result("anthropic", Outcome.SERVER_ERROR)})

    client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert _failovers(**labels) == before + 1


# --- caller faults do not fail over -------------------------------------------


@pytest.mark.parametrize("outcome", [Outcome.BAD_REQUEST, Outcome.CONTENT_FILTER])
def test_a_caller_fault_stops_the_walk_dead(client, stub_ladder, outcome):
    """Bad input is not a failover trigger.

    Every remaining rung will reject it identically, so walking on bills three
    more calls to reproduce the same 400. This is the same mistake as letting
    caller faults open a breaker, aimed at the ladder instead of the circuit.
    """
    calls = stub_ladder({"anthropic": failed_result("anthropic", outcome)})

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 400
    assert calls == ["anthropic"], "the caller's bad input must not cost four calls"


def test_a_stale_model_name_does_fail_over(client, stub_ladder):
    """The counterpart to the test above, and the line between them.

    Both are 4xx from the provider, so the naive reading puts them together - and
    that is exactly the bug this fixes. A model that has been decommissioned is
    the gateway's own configuration gone stale: the next rung has a different
    model and will serve the request perfectly well, so stopping the walk here
    turns one bad config line into a total outage for that class.
    """
    calls = stub_ladder(
        {"anthropic": failed_result("anthropic", Outcome.MODEL_NOT_FOUND)}
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 200
    assert calls == ["anthropic", "openai"]
    assert response.json()["x_gateway"]["provider"] == "openai"


def test_a_ladder_of_stale_models_is_a_502_not_a_400(client, stub_ladder):
    """Nobody sent anything wrong, so the caller must not be told they did."""
    stub_ladder(
        {p: failed_result(p, Outcome.MODEL_NOT_FOUND) for p in
         ("anthropic", "openai", "groq", "ollama")}
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "model_not_found"


def test_every_failure_outcome_has_a_status(client):
    """_serve and _exhausted_status index _STATUS_BY_OUTCOME directly.

    A new Outcome without an entry does not fail loudly at import - it raises a
    KeyError on the first request that hits it, turning the one response that
    would have named the misconfiguration into an unexplained 500.
    """
    from gateway.main import _STATUS_BY_OUTCOME

    missing = {o for o in Outcome if not o.ok} - set(_STATUS_BY_OUTCOME)
    assert not missing, f"no client status for {sorted(missing)}"


# --- open circuits are skipped ------------------------------------------------


async def _trip(fake_redis, provider):
    for _ in range(4):
        await health.record(fake_redis, provider, Outcome.SERVER_ERROR, 100.0)
        await breaker.record(fake_redis, provider, Outcome.SERVER_ERROR)


async def test_an_open_circuit_is_skipped_without_a_call(client, stub_ladder, fake_redis):
    """An open breaker has to save the call, not just record it afterwards."""
    await _trip(fake_redis, "anthropic")
    calls = stub_ladder({})

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 200
    assert calls == ["openai"], "anthropic was never dialled"
    assert response.json()["x_gateway"]["provider"] == "openai"


async def test_a_skip_says_why_and_for_how_long(client, stub_ladder, fake_redis):
    await _trip(fake_redis, "anthropic")
    stub_ladder({})

    attempts = _attempts(client.post("/v1/chat/completions", json=BODY, headers=CHAT))

    assert attempts[0] == {
        "provider": "anthropic",
        "outcome": "skipped",
        "latency_ms": 0.0,
        "error": attempts[0]["error"],
        "extra": {},
    }
    assert "circuit open" in attempts[0]["error"]


# --- exhaustion ---------------------------------------------------------------


def test_an_exhausted_ladder_names_everything_it_tried(client, stub_ladder):
    stub_ladder(
        {p: failed_result(p, Outcome.SERVER_ERROR) for p in
         ("anthropic", "openai", "groq", "ollama")}
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["ladder"] == ["anthropic", "openai", "groq", "ollama"]
    assert len(body["x_gateway"]["attempts"]) == 4


def test_exhaustion_reports_what_actually_failed(client, stub_ladder):
    """A ladder of rate limits is a 429. Flattening it to 503 loses the back-off signal."""
    stub_ladder(
        {p: failed_result(p, Outcome.RATE_LIMIT) for p in
         ("anthropic", "openai", "groq", "ollama")}
    )

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit"


async def test_every_circuit_open_is_a_503(client, stub_ladder, fake_redis):
    """Nothing was even attempted, which is precisely what 503 means."""
    for provider in ("anthropic", "openai", "groq", "ollama"):
        await _trip(fake_redis, provider)
    calls = stub_ladder({})

    response = client.post("/v1/chat/completions", json=BODY, headers=CHAT)

    assert response.status_code == 503
    assert calls == []
    assert response.json()["error"]["type"] == "no_provider_available"
    assert all(a["outcome"] == "skipped" for a in _attempts(response))


# --- chaos reaches the walk ---------------------------------------------------


def test_chaos_reaches_the_walk_through_the_real_dispatch(client):
    """Injected fault in at /admin/chaos, classified outcome out at the boundary.

    Pinned to one rung so nothing else is dialled for real. The point is that
    the router fetches the spec, hands it to call_provider, and what comes back
    has been through classify_exception like any genuine failure would.
    """
    client.post(
        "/admin/chaos",
        json={"provider": "anthropic", "error_rate": 1.0, "error_type": "server_error"},
        headers=ADMIN_HEADERS,
    )

    response = client.post(
        "/v1/chat/completions?provider=anthropic", json=BODY, headers=CHAT
    )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"
    assert "injected" in _attempts(response)[0]["error"]


# --- pinning ------------------------------------------------------------------


def test_pinning_a_provider_still_bypasses_the_ladder(client, stub_ladder):
    calls = stub_ladder({"groq": ok_result("groq")})

    response = client.post("/v1/chat/completions?provider=groq", json=BODY, headers=CHAT)

    assert response.status_code == 200
    assert calls == ["groq"]
