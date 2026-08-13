"""Gateway configuration: providers, request classes, and key-aware ladders."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "gateway.yaml"


class ProviderConfig(BaseModel):
    """One backend the gateway can dispatch to."""

    model: str
    api_key_env: str | None = None
    api_base_env: str | None = None
    api_base_default: str | None = None
    timeout_s: float = 30.0

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


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    classes: dict[str, ClassConfig]
    default_class: str

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
