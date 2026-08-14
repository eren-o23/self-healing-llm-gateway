"""The error taxonomy is the spine of the circuit breaker, so it gets a real test."""

from __future__ import annotations

import httpx
import litellm
import pytest
from litellm import exceptions as llm_exc

from gateway import chaos
from gateway.config import ProviderConfig
from gateway.providers import Outcome, _cost_of, call_provider, classify_exception

_RESPONSE = httpx.Response(403, request=httpx.Request("POST", "http://provider.test"))
_ARGS = {"message": "boom", "llm_provider": "openai", "model": "gpt-4o-mini"}


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (llm_exc.RateLimitError(**_ARGS), Outcome.RATE_LIMIT),
        (llm_exc.Timeout(**_ARGS), Outcome.TIMEOUT),
        (llm_exc.APIConnectionError(**_ARGS), Outcome.TIMEOUT),
        (TimeoutError(), Outcome.TIMEOUT),
        (llm_exc.AuthenticationError(**_ARGS), Outcome.AUTH),
        (llm_exc.PermissionDeniedError(**_ARGS, response=_RESPONSE), Outcome.AUTH),
        (llm_exc.ContentPolicyViolationError(**_ARGS), Outcome.CONTENT_FILTER),
        (llm_exc.ContextWindowExceededError(**_ARGS), Outcome.BAD_REQUEST),
        (llm_exc.NotFoundError(**_ARGS), Outcome.BAD_REQUEST),
        (llm_exc.BadRequestError(**_ARGS), Outcome.BAD_REQUEST),
        (llm_exc.ServiceUnavailableError(**_ARGS), Outcome.SERVER_ERROR),
        (llm_exc.InternalServerError(**_ARGS), Outcome.SERVER_ERROR),
        (llm_exc.APIError(status_code=500, **_ARGS), Outcome.SERVER_ERROR),
        (RuntimeError("something nobody anticipated"), Outcome.SERVER_ERROR),
    ],
    ids=lambda v: getattr(v, "value", type(v).__name__),
)
def test_classify_exception(exc, expected):
    assert classify_exception(exc) is expected


def test_content_filter_beats_bad_request_base_class():
    """ContentPolicyViolationError subclasses BadRequestError.

    If taxonomy order ever regresses so the base class matches first, content
    filters would silently become BAD_REQUEST. Both are non-trippable so no
    breaker misfires, but the error taxonomy on the dashboard would go wrong.
    """
    assert issubclass(llm_exc.ContentPolicyViolationError, llm_exc.BadRequestError)
    assert classify_exception(llm_exc.ContentPolicyViolationError(**_ARGS)) is (
        Outcome.CONTENT_FILTER
    )


def test_caller_faults_never_trip_a_breaker():
    """The point of the taxonomy: bad input must not take a healthy provider down."""
    assert not Outcome.CONTENT_FILTER.trippable
    assert not Outcome.BAD_REQUEST.trippable
    assert not Outcome.OK.trippable

    assert Outcome.RATE_LIMIT.trippable
    assert Outcome.TIMEOUT.trippable
    assert Outcome.SERVER_ERROR.trippable
    assert Outcome.AUTH.trippable


def _usage_response(model: str, prompt: int = 1000, completion: int = 1000):
    """Minimal stand-in for a provider response, enough for the cost table."""
    return litellm.ModelResponse(
        model=model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        usage=litellm.Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def test_cost_of_unpriced_model_is_zero_not_an_exception():
    """ollama is genuinely free; an unpriced model must not fail a served request."""
    assert _cost_of(_usage_response("ollama/llama3.2"), "ollama/llama3.2") == 0.0


def test_cost_uses_configured_model_not_the_echoed_one():
    """Regression: groq echoes a bare model id that is absent from the price table.

    Inferring the model from the response yielded $0 for a genuinely billed call.
    Silent zero-cost removes the main reason to run a gateway at all, so the
    configured name must win over whatever the provider echoes back.
    """
    configured = "groq/llama-3.3-70b-versatile"
    echoed = "llama-3.3-70b-versatile"

    assert echoed not in litellm.model_cost, "premise: the bare id is unpriced"

    response = _usage_response(echoed)

    assert _cost_of(response, configured) > 0.0
    assert _cost_of(response, echoed) == 0.0, "inferring from the response is the bug"


@pytest.mark.parametrize("error_type", sorted(chaos.INJECTABLE))
async def test_injected_faults_travel_the_real_classification_path(error_type):
    """Chaos must be indistinguishable from a genuine failure downstream.

    The injected exception is a real LiteLLM type raised at the real dispatch
    point, so classify_exception maps it exactly as it would the provider's own -
    which is what makes a chaos-driven demo evidence that failover works, rather
    than evidence that a mock works.
    """
    spec = chaos.ChaosSpec(provider="groq", error_rate=1.0, error_type=error_type)

    result = await call_provider(
        "groq", ProviderConfig(model="groq/llama-3.3-70b-versatile"), {}, chaos=spec
    )

    assert result.outcome == error_type
    assert "injected" in result.error


async def test_injected_latency_lands_inside_the_measured_span():
    """Slow and broken are different symptoms; the p95 trip path needs the first.

    Latency is applied unconditionally and before the error roll, so a spec can
    make a provider slow without making it fail - and the delay is inside
    call_provider's own timer, which is what the health window records.
    """
    spec = chaos.ChaosSpec(
        provider="groq", error_rate=1.0, latency_ms=50, error_type="timeout"
    )

    result = await call_provider(
        "groq", ProviderConfig(model="groq/llama-3.3-70b-versatile"), {}, chaos=spec
    )

    assert result.outcome == Outcome.TIMEOUT
    assert result.latency_ms >= 50


async def test_latency_only_chaos_does_not_raise():
    await chaos.ChaosSpec(provider="groq", error_rate=0.0, latency_ms=1).apply()
