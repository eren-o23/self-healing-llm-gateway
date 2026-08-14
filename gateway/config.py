"""Gateway configuration: providers, request classes, and key-aware ladders."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import redis.asyncio as aioredis
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "gateway.yaml"

# Load .env once, at import, so every entrypoint sees provider keys: uvicorn,
# pytest, and the phase 4 worker alike. Real environment variables win, which
# keeps docker compose (which injects them directly) behaving identically.
load_dotenv(override=False)


class ProviderConfig(BaseModel):
    """One backend the gateway can dispatch to."""

    model: str
    api_key_env: str | None = None
    api_base_env: str | None = None
    api_base_default: str | None = None
    timeout_s: float = 30.0

    # Per provider, never global. Ollama's ~14s on local CPU is healthy; judging it
    # against anthropic's budget would hold its breaker open forever and remove the
    # floor from every ladder.
    latency_budget_ms: float = 15_000.0

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None

    @property
    def api_base(self) -> str | None:
        if not self.api_base_env:
            return None
        return os.getenv(self.api_base_env) or self.api_base_default

    @property
    def available(self) -> bool:
        """A provider needing a key it does not have cannot serve traffic."""
        return self.api_key_env is None or bool(self.api_key)


class ClassConfig(BaseModel):
    """A request class: which providers to try, in what order, and how to fail."""

    ladder: list[str] = Field(min_length=1)
    deferrable: bool = False
    hedge_after_ms: int | None = None


class HealthConfig(BaseModel):
    """The sliding window every provider's health is judged over.

    Shortening this makes the breaker react faster and trip on thinner evidence.
    It is the first knob to turn if failover reads as sluggish on camera.
    """

    window_s: int = 60


class BreakerConfig(BaseModel):
    """When a provider is pulled out of the ladder, and how it earns its way back.

    `min_samples` is the guard against acting on thin evidence, and it is counted
    over HealthSnapshot.health_samples - successes plus trippable failures - not
    over the raw window. Counting raw samples would let a hundred bad requests and
    three real failures clear the threshold at a 100% error rate.
    """

    error_threshold: float = 0.5
    min_samples: int = 8
    cooldown_s: float = 20.0
    cooldown_max_s: float = 120.0
    probe_ratio: float = 0.1
    probe_successes_required: int = 2


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    classes: dict[str, ClassConfig]
    default_class: str
    health: HealthConfig = HealthConfig()
    breaker: BreakerConfig = BreakerConfig()

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        if self.default_class not in self.classes:
            raise ValueError(f"default_class {self.default_class!r} is not a defined class")
        for name, cls in self.classes.items():
            unknown = set(cls.ladder) - set(self.providers)
            if unknown:
                raise ValueError(f"class {name!r} references unknown providers: {sorted(unknown)}")
        return self

    def available_providers(self) -> list[str]:
        return [n for n, p in self.providers.items() if p.available]

    def ladder_for(self, request_class: str) -> list[str]:
        """The class's preference list, minus providers that cannot serve traffic.

        Returns [] for an unknown class; the caller decides whether that is a 400
        or a fallback to default_class.
        """
        cls = self.classes.get(request_class)
        if cls is None:
            return []
        return [p for p in cls.ladder if self.providers[p].available]

    def log_startup_state(self) -> None:
        for name, provider in self.providers.items():
            if provider.available:
                log.info("provider %s available (model=%s)", name, provider.model)
            else:
                log.warning(
                    "provider %s DISABLED: %s is unset; dropping it from all ladders",
                    name,
                    provider.api_key_env,
                )
        for name in self.classes:
            log.info("class %s ladder=%s", name, self.ladder_for(name))


def load_config(path: str | Path | None = None) -> GatewayConfig:
    path = Path(path or os.getenv("GATEWAY_CONFIG") or DEFAULT_CONFIG_PATH)
    return GatewayConfig.model_validate(yaml.safe_load(path.read_text()))


@lru_cache(maxsize=1)
def get_config() -> GatewayConfig:
    return load_config()


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    """The shared Redis handle: health windows now, breaker and queue later.

    Sync and cached rather than async: lru_cache on an async function would cache
    the coroutine object, which can only be awaited once. Handlers take this as a
    FastAPI dependency so tests can override it; the phase 4 worker calls it
    directly, with no FastAPI in sight.

    decode_responses=True so sorted-set members come back as str rather than
    needing a .decode() at every parse site.
    """
    return aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
    )
