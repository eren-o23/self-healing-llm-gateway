"""The replay guard, which exists before anything is allowed to retry.

Every test here is ultimately about one number: how many times a provider was
called. The stub fixtures already record that, so the assertions are on the call
list rather than on anything the response happens to say.
"""

from __future__ import annotations

from gateway import idempotency
from gateway.providers import Outcome
from tests.conftest import BODY, HEADERS, failed_result, ok_result

KEY = {"Idempotency-Key": "order-4711"}


def test_replay_returns_the_stored_response_without_calling_a_provider(
    client, stub_provider
):
    calls = stub_provider(ok_result())

    first = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    assert first.status_code == 200
    assert calls == ["anthropic"]

    second = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    assert second.status_code == 200
    assert second.json() == first.json()
    assert second.headers["X-Idempotent-Replay"] == "true"
    # The whole point: the second request cost nothing.
    assert calls == ["anthropic"]


async def test_a_key_still_in_flight_is_a_409(client, stub_provider, fake_redis):
    stub_provider(ok_result())
    await fake_redis.set("idem:acme:order-4711", idempotency.IN_FLIGHT)

    response = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    assert response.status_code == 409


async def test_an_expired_claim_reads_as_in_flight(client, stub_provider, fake_redis):
    """SET NX can fail and the following GET still miss, if the TTL lands between.

    Answering that with a 409 the caller retries is the conservative side of the
    race; the other side is a duplicate call it cannot take back.
    """
    stub_provider(ok_result())

    async def vanished(_key):
        return None

    await fake_redis.set("idem:acme:order-4711", idempotency.IN_FLIGHT)
    fake_redis.get = vanished

    response = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    assert response.status_code == 409


def test_a_failed_request_releases_the_key(client, stub_ladder):
    """A transient failure must not pin the key for its whole TTL."""
    broken = {
        name: failed_result(name, Outcome.SERVER_ERROR)
        for name in ("anthropic", "openai", "groq", "ollama")
    }
    calls = stub_ladder(broken)

    first = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    assert first.status_code == 502
    assert len(calls) == 4

    # Retrying the same key reaches the providers again rather than replaying the
    # failure or being turned away with a 409.
    second = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    assert second.status_code == 502
    assert len(calls) == 8


def test_a_rejected_request_releases_the_key(client, stub_provider):
    """The claim is taken before the ladder resolves, so a 400 must free it too."""
    stub_provider(ok_result())

    rejected = client.post(
        "/v1/chat/completions?provider=nope", json=BODY, headers=HEADERS | KEY
    )
    assert rejected.status_code == 400

    ok = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    assert ok.status_code == 200


def test_the_same_key_from_two_tenants_does_not_collide(client, stub_provider):
    calls = stub_provider(ok_result())

    acme = client.post("/v1/chat/completions", json=BODY, headers=HEADERS | KEY)
    other = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers=HEADERS | KEY | {"X-Tenant-Id": "globex"},
    )

    assert acme.status_code == other.status_code == 200
    # Two real calls: globex got its own answer, not acme's.
    assert calls == ["anthropic", "anthropic"]
    assert "X-Idempotent-Replay" not in other.headers


def test_no_key_means_no_guard(client, stub_provider):
    calls = stub_provider(ok_result())

    for _ in range(2):
        assert (
            client.post("/v1/chat/completions", json=BODY, headers=HEADERS).status_code
            == 200
        )
    assert calls == ["anthropic", "anthropic"]


async def test_replaying_a_deferrable_request_returns_the_same_job(
    client, stub_provider, fake_redis
):
    """The stored envelope carries its status, so a 202 replays as a 202."""
    stub_provider(ok_result())
    deferrable = HEADERS | KEY | {"X-Request-Class": "batch.generate"}

    first = client.post("/v1/chat/completions", json=BODY, headers=deferrable)
    second = client.post("/v1/chat/completions", json=BODY, headers=deferrable)

    assert first.status_code == second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    # One job, not two. A retried submission must not double the work.
    assert await fake_redis.zcard("queue:ready") == 1
