from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from autofinetune.config import OrchestratorConfig
from autofinetune.errors import FatalError

DEEPSEEK_API_BASE = "https://api.deepseek.com"


@dataclass(frozen=True)
class ResolvedLiteLLMCall:
    model: str
    api_base: str | None
    api_key: str | None


def _getenv_default(name: str) -> str | None:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _require_key(env_name: str, provider: str, getenv: Callable[[str], str | None]) -> str:
    key = getenv(env_name)
    if not key:
        raise FatalError(
            f"{env_name} is required when orchestrator.provider={provider}"
        )
    return key


def _deepseek_model(model: str) -> str:
    bare = model[len("openai/") :] if model.startswith("openai/") else model
    return f"openai/{bare}"


def resolve_litellm_call(
    cfg: OrchestratorConfig,
    *,
    getenv: Callable[[str], str | None] | None = None,
) -> ResolvedLiteLLMCall:
    getenv = getenv or _getenv_default
    provider = (cfg.provider or "").strip().lower()
    user_base = cfg.api_base

    if provider == "deepseek":
        return ResolvedLiteLLMCall(
            model=_deepseek_model(cfg.model),
            api_base=user_base or DEEPSEEK_API_BASE,
            api_key=_require_key("DEEPSEEK_API_KEY", provider, getenv),
        )
    if provider == "openai":
        return ResolvedLiteLLMCall(
            model=cfg.model,
            api_base=user_base,
            api_key=_require_key("OPENAI_API_KEY", provider, getenv),
        )
    if provider == "litellm":
        return ResolvedLiteLLMCall(
            model=cfg.model,
            api_base=user_base,
            api_key=None,
        )
    raise FatalError(
        f"Unknown orchestrator.provider={cfg.provider!r}; "
        "expected deepseek, openai, or litellm"
    )
