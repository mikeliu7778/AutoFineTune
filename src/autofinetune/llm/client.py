from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from autofinetune.config import OrchestratorConfig
from autofinetune.errors import FatalError, RoundError

# Best-effort v1 estimate; not provider-metered billing.
EST_COST_USD_PER_CALL = 0.001


class LLMClient(Protocol):
    cost_usd_est: float

    def complete_json(self, system: str, user: str, schema_name: str) -> dict[str, Any]:
        ...


class FakeLLMClient:
    def __init__(self, handlers: dict[str, Callable[[str, str], dict[str, Any]]]) -> None:
        self.handlers = handlers
        self.calls: list[tuple[str, str, str]] = []
        self.cost_usd_est: float = 0.0

    def complete_json(self, system: str, user: str, schema_name: str) -> dict[str, Any]:
        self.calls.append((system, user, schema_name))
        self.cost_usd_est += EST_COST_USD_PER_CALL
        if schema_name not in self.handlers:
            raise FatalError(f"FakeLLMClient missing handler for {schema_name}")
        return self.handlers[schema_name](system, user)


class LiteLLMClient:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.cost_usd_est: float = 0.0

    def complete_json(self, system: str, user: str, schema_name: str) -> dict[str, Any]:
        try:
            from litellm import completion
        except ImportError as e:
            raise FatalError("litellm is required for cloud orchestrator") from e

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = completion(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": system
                            + f"\nRespond with a JSON object for schema '{schema_name}'.",
                        },
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or "{}"
                self.cost_usd_est += EST_COST_USD_PER_CALL
                return json.loads(content)
            except Exception as e:  # noqa: BLE001 — retried network/provider errors
                last_err = e
                time.sleep(min(2**attempt, 8))
        raise RoundError(f"LLM call failed after retries: {last_err}")
