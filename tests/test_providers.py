"""The error taxonomy is the spine of the circuit breaker, so it gets a real test."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from litellm import exceptions as llm_exc

from gateway.providers import Outcome, _cost_of, classify_exception

_RESPONSE = httpx.Response(403, request=httpx.Request("POST", "http://provider.test"))
_ARGS = {"message": "boom", "llm_provider": "openai", "model": "gpt-4o-mini"}


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (llm_exc.RateLimitError(**_ARGS), Outcome.RATE_LIMIT),
        (llm_exc.Timeout(**_ARGS), Outcome.TIMEOUT),
        (llm_exc.APIConnectionError(**_ARGS), Outcome.TIMEOUT),
        (asyncio.TimeoutError(), Outcome.TIMEOUT),
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


def test_cost_of_unpriced_model_is_zero_not_an_exception():
    """ollama is genuinely free; an unpriced model must not fail a served request."""

    class Unpriced:
        model = "ollama/llama3.2"

    assert _cost_of(Unpriced()) == 0.0
