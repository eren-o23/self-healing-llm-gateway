"""OpenAI-compatible request/response shapes, plus the metadata the gateway requires."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """The OpenAI request shape.

    extra="allow" so params we do not enumerate (tools, response_format, seed, ...)
    still reach the provider. The point of the gateway is that existing clients
    change a base URL and nothing else.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None  # advisory only: the ladder picks the real model
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False

    def to_provider_kwargs(self) -> dict[str, Any]:
        """Everything except the fields the gateway owns."""
        payload = self.model_dump(exclude_none=True, exclude={"model", "stream"})
        payload["messages"] = [m.model_dump(exclude_none=True) for m in self.messages]
        return payload


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class GatewayMeta(BaseModel):
    """Gateway extension block. Namespaced so OpenAI clients ignore it."""

    provider: str
    model: str
    request_class: str
    request_id: str
    tenant: str
    feature: str
    latency_ms: float
    cost_usd: float
    attempts: list[dict[str, Any]] = Field(default_factory=list)


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    x_gateway: GatewayMeta


class RequestMetadata(BaseModel):
    """Required on every request. Without it, cost can never be attributed."""

    tenant: str
    feature: str
    request_id: str
    request_class: str


class ChaosRequest(BaseModel):
    """Break a provider on purpose.

    ttl_s is required to be finite and is capped: injected faults must expire on
    their own, because the failure mode of a fault-injection API is somebody
    walking away from a broken provider and no longer remembering they broke it.
    """

    provider: str
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: int = Field(default=0, ge=0, le=120_000)
    error_type: str = "server_error"
    ttl_s: int = Field(default=120, gt=0, le=3600)
