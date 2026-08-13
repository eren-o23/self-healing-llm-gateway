"""FastAPI app: one service every model call routes through."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from gateway.config import GatewayConfig, get_config
from gateway.providers import Outcome, ProviderCallLog, ProviderResult, call_provider
from gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    GatewayMeta,
    RequestMetadata,
    Usage,
)

log = logging.getLogger(__name__)

# How a provider outcome surfaces to the caller. AUTH is a 502 on purpose: a
# missing or rejected provider key is the gateway's problem, not the caller's.
_STATUS_BY_OUTCOME: dict[Outcome, int] = {
    Outcome.RATE_LIMIT: 429,
    Outcome.TIMEOUT: 504,
    Outcome.SERVER_ERROR: 502,
    Outcome.AUTH: 502,
    Outcome.CONTENT_FILTER: 400,
    Outcome.BAD_REQUEST: 400,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_config().log_startup_state()
    yield


app = FastAPI(
    title="Self-Healing LLM Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


def require_metadata(
    config: Annotated[GatewayConfig, Depends(get_config)],
    x_tenant_id: Annotated[str, Header(min_length=1)],
    x_feature: Annotated[str, Header(min_length=1)],
    x_request_id: Annotated[str, Header(min_length=1)],
    x_request_class: Annotated[str | None, Header()] = None,
) -> RequestMetadata:
    """Reject unattributable traffic at the boundary.

    Without tenant and feature, cost can never be attributed - and retrofitting
    required fields onto a running gateway is the migration nobody wants. So the
    headers are mandatory from the first request the service ever serves.
    """
    request_class = x_request_class or config.default_class
    if request_class not in config.classes:
        raise HTTPException(
            status_code=400,
            detail=f"unknown request class {request_class!r}; "
            f"known: {sorted(config.classes)}",
        )
    return RequestMetadata(
        tenant=x_tenant_id,
        feature=x_feature,
        request_id=x_request_id,
        request_class=request_class,
    )


@app.get("/healthz")
async def healthz(config: Annotated[GatewayConfig, Depends(get_config)]) -> dict[str, Any]:
    return {
        "status": "ok",
        "providers": {
            name: ("available" if p.available else "disabled")
            for name, p in config.providers.items()
        },
        "classes": {name: config.ladder_for(name) for name in config.classes},
    }


@app.get("/v1/models")
async def list_models(config: Annotated[GatewayConfig, Depends(get_config)]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": name, "model": p.model}
            for name, p in config.providers.items()
            if p.available
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    meta: Annotated[RequestMetadata, Depends(require_metadata)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    provider: Annotated[
        str | None, Query(description="Pin a provider, bypassing the ladder")
    ] = None,
):
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported")

    ladder = _resolve_ladder(config, meta.request_class, provider)
    payload = request.to_provider_kwargs()

    # ponytail: phase 1 tries the first rung only. Failover across the ladder,
    # circuit breaking and hedging land in phase 3 as gateway/router.py.
    chosen = ladder[0]
    result = await call_provider(chosen, config.providers[chosen], payload)

    attempt = ProviderCallLog(
        provider=result.provider,
        outcome=str(result.outcome),
        latency_ms=round(result.latency_ms, 1),
        error=result.error,
    )
    log.info(
        "request_id=%s tenant=%s feature=%s class=%s provider=%s outcome=%s "
        "latency_ms=%.1f cost_usd=%.6f",
        meta.request_id,
        meta.tenant,
        meta.feature,
        meta.request_class,
        result.provider,
        result.outcome,
        result.latency_ms,
        result.cost_usd,
    )

    if not result.outcome.ok:
        return JSONResponse(
            status_code=_STATUS_BY_OUTCOME[result.outcome],
            content={
                "error": {
                    "message": result.error or str(result.outcome),
                    "type": str(result.outcome),
                    "provider": result.provider,
                },
                "x_gateway": {
                    "request_id": meta.request_id,
                    "request_class": meta.request_class,
                    "attempts": [vars(attempt)],
                },
            },
        )

    return _to_openai_response(result, meta, [attempt])


def _resolve_ladder(
    config: GatewayConfig, request_class: str, override: str | None
) -> list[str]:
    """The ordered providers to try, or a 4xx/503 explaining why there are none."""
    if override:
        if override not in config.providers:
            raise HTTPException(
                status_code=400,
                detail=f"unknown provider {override!r}; known: {sorted(config.providers)}",
            )
        if not config.providers[override].available:
            raise HTTPException(
                status_code=503,
                detail=f"provider {override!r} is disabled: "
                f"{config.providers[override].api_key_env} is unset",
            )
        return [override]

    ladder = config.ladder_for(request_class)
    if not ladder:
        raise HTTPException(
            status_code=503,
            detail=f"no available provider for class {request_class!r}; "
            "every provider in its ladder is missing an API key",
        )
    return ladder


def _to_openai_response(
    result: ProviderResult, meta: RequestMetadata, attempts: list[ProviderCallLog]
) -> ChatCompletionResponse:
    response = result.response
    choices = [
        Choice(
            index=getattr(c, "index", i),
            message=ChatMessage(
                role=getattr(c.message, "role", "assistant") or "assistant",
                content=getattr(c.message, "content", None),
            ),
            finish_reason=getattr(c, "finish_reason", None),
        )
        for i, c in enumerate(getattr(response, "choices", []))
    ]
    return ChatCompletionResponse(
        id=getattr(response, "id", None) or f"chatcmpl-{uuid.uuid4().hex}",
        created=getattr(response, "created", None) or int(time.time()),
        model=getattr(response, "model", None) or result.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
        x_gateway=GatewayMeta(
            provider=result.provider,
            model=result.model,
            request_class=meta.request_class,
            request_id=meta.request_id,
            tenant=meta.tenant,
            feature=meta.feature,
            latency_ms=round(result.latency_ms, 1),
            cost_usd=result.cost_usd,
            attempts=[vars(a) for a in attempts],
        ),
    )
