# DeepSeek Provider Preset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DeepSeek the default orchestrator provider via LiteLLM presets (`provider` + `api_base` + `DEEPSEEK_API_KEY`), with OpenAI/generic LiteLLM still available.

**Architecture:** Extend `OrchestratorConfig` with `provider` and optional `api_base`. Add `resolve_litellm_call` helper that maps provider → `(model, api_base, api_key)`. `LiteLLMClient.complete_json` passes those into `litellm.completion`. Defaults and README switch to DeepSeek; fake/CI path unchanged.

**Tech Stack:** Python 3.11+, Pydantic, LiteLLM, pytest, monkeypatch

## Global Constraints

- No dedicated DeepSeekClient; stay on LiteLLM.
- No DeepSeek thinking / `reasoning_effort` in v1.
- Do not rewrite `cfg.model` at runtime based on provider.
- DeepSeek routing: LiteLLM model `openai/<bare_id>` + `api_base=https://api.deepseek.com`.
- Missing required key → `FatalError` naming the env var.
- Author commits as `mike <mliu36292@gmail.com>`; never leave `Co-authored-by: Cursor` trailers.
- Spec: `docs/superpowers/specs/2026-08-13-deepseek-provider-design.md`

## File Structure

| File | Responsibility |
|------|----------------|
| `src/autofinetune/config.py` | `OrchestratorConfig.provider`, `api_base`; defaults |
| `src/autofinetune/defaults/config.yaml` | Default yaml: deepseek + deepseek-v4-flash |
| `src/autofinetune/llm/providers.py` | Pure resolver: provider → litellm kwargs pieces |
| `src/autofinetune/llm/client.py` | `LiteLLMClient` uses resolver; key check |
| `src/autofinetune/llm/__init__.py` | Export resolver if useful for tests |
| `tests/test_llm_providers.py` | Resolver + LiteLLMClient mock completion tests |
| `tests/test_config.py` | Assert new defaults |
| `README.md` | DeepSeek quick start + override examples |

---

### Task 1: Config defaults (`provider`, `api_base`, DeepSeek model)

**Files:**
- Modify: `src/autofinetune/config.py`
- Modify: `src/autofinetune/defaults/config.yaml`
- Modify: `tests/test_config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `OrchestratorConfig(provider: str = "deepseek", model: str = "deepseek-v4-flash", temperature: float = 0.2, max_retries: int = 3, api_base: str | None = None)`

- [ ] **Step 1: Extend failing assertions in `tests/test_config.py`**

```python
from autofinetune.config import load_config


def test_load_config_defaults_max_rounds_and_auto_base():
    cfg = load_config(None)
    assert cfg.budgets.max_rounds >= 1
    assert cfg.base_model == "auto"
    assert cfg.trainer.backend == "trl"
    assert cfg.data.min_qa_for_full >= 1
    assert len(cfg.allowlist) >= 1
    assert cfg.orchestrator.provider == "deepseek"
    assert cfg.orchestrator.model == "deepseek-v4-flash"
    assert cfg.orchestrator.api_base is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_load_config_defaults_max_rounds_and_auto_base -v`  
Expected: FAIL (provider/model attributes missing or old openai default)

- [ ] **Step 3: Update `OrchestratorConfig` and defaults yaml**

In `src/autofinetune/config.py`:

```python
class OrchestratorConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.2
    max_retries: int = 3
    api_base: str | None = None
```

In `src/autofinetune/defaults/config.yaml`:

```yaml
orchestrator:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.2
  max_retries: 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/config.py src/autofinetune/defaults/config.yaml tests/test_config.py
git commit -m "feat: default orchestrator to DeepSeek provider and model"
```

---

### Task 2: Provider resolver helper

**Files:**
- Create: `src/autofinetune/llm/providers.py`
- Create: `tests/test_llm_providers.py`
- Modify: `src/autofinetune/llm/__init__.py` (export `resolve_litellm_call`)

**Interfaces:**
- Consumes: `OrchestratorConfig`
- Produces:
  - `ResolvedLiteLLMCall` dataclass or NamedTuple with fields `model: str`, `api_base: str | None`, `api_key: str | None`, `require_api_key_env: str | None`
  - `resolve_litellm_call(cfg: OrchestratorConfig, *, getenv: Callable[[str], str | None] | None = None) -> ResolvedLiteLLMCall`
  - Raises `FatalError` when `provider` in `{deepseek, openai}` and required env is missing/empty
  - Raises `FatalError` for unknown `provider`

- [ ] **Step 1: Write failing tests in `tests/test_llm_providers.py`**

```python
import pytest

from autofinetune.config import OrchestratorConfig
from autofinetune.errors import FatalError
from autofinetune.llm.providers import resolve_litellm_call


def test_deepseek_resolves_openai_compat_route(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    cfg = OrchestratorConfig(provider="deepseek", model="deepseek-v4-flash")
    resolved = resolve_litellm_call(cfg)
    assert resolved.model == "openai/deepseek-v4-flash"
    assert resolved.api_base == "https://api.deepseek.com"
    assert resolved.api_key == "sk-ds-test"


def test_deepseek_pro_model_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    cfg = OrchestratorConfig(provider="deepseek", model="deepseek-v4-pro")
    resolved = resolve_litellm_call(cfg)
    assert resolved.model == "openai/deepseek-v4-pro"


def test_deepseek_strips_existing_openai_prefix(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    cfg = OrchestratorConfig(provider="deepseek", model="openai/deepseek-v4-flash")
    resolved = resolve_litellm_call(cfg)
    assert resolved.model == "openai/deepseek-v4-flash"


def test_deepseek_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = OrchestratorConfig(provider="deepseek", model="deepseek-v4-flash")
    with pytest.raises(FatalError, match="DEEPSEEK_API_KEY"):
        resolve_litellm_call(cfg)


def test_openai_provider_uses_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    cfg = OrchestratorConfig(provider="openai", model="openai/gpt-4o-mini")
    resolved = resolve_litellm_call(cfg)
    assert resolved.model == "openai/gpt-4o-mini"
    assert resolved.api_base is None
    assert resolved.api_key == "sk-oai"


def test_user_api_base_overrides_preset(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    cfg = OrchestratorConfig(
        provider="deepseek",
        model="deepseek-v4-flash",
        api_base="https://example.com/v1",
    )
    resolved = resolve_litellm_call(cfg)
    assert resolved.api_base == "https://example.com/v1"


def test_litellm_provider_no_key_enforcement(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = OrchestratorConfig(provider="litellm", model="anthropic/claude-3-haiku")
    resolved = resolve_litellm_call(cfg)
    assert resolved.model == "anthropic/claude-3-haiku"
    assert resolved.api_key is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_providers.py -v`  
Expected: FAIL (import / missing module)

- [ ] **Step 3: Implement `src/autofinetune/llm/providers.py`**

```python
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
```

Export from `llm/__init__.py`:

```python
from autofinetune.llm.client import FakeLLMClient, LLMClient, LiteLLMClient
from autofinetune.llm.providers import ResolvedLiteLLMCall, resolve_litellm_call

__all__ = [
    "LLMClient",
    "LiteLLMClient",
    "FakeLLMClient",
    "ResolvedLiteLLMCall",
    "resolve_litellm_call",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_providers.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/llm/providers.py src/autofinetune/llm/__init__.py tests/test_llm_providers.py
git commit -m "feat: resolve LiteLLM call kwargs from orchestrator provider"
```

---

### Task 3: Wire `LiteLLMClient` to resolver (mock completion)

**Files:**
- Modify: `src/autofinetune/llm/client.py`
- Modify: `tests/test_llm_providers.py` (add client integration tests)

**Interfaces:**
- Consumes: `resolve_litellm_call(cfg)`
- Produces: `LiteLLMClient.complete_json` calls `completion(model=..., api_base=..., api_key=..., ...)` when non-None

- [ ] **Step 1: Add failing client tests**

Append to `tests/test_llm_providers.py`:

```python
from autofinetune.llm.client import LiteLLMClient


def test_litellm_client_passes_deepseek_kwargs(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)

        class Msg:
            content = '{"ok": true}'

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()

    import sys
    import types

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    client = LiteLLMClient(OrchestratorConfig(provider="deepseek", model="deepseek-v4-flash"))
    out = client.complete_json("sys", "user", "round_plan")
    assert out == {"ok": True}
    assert calls[0]["model"] == "openai/deepseek-v4-flash"
    assert calls[0]["api_base"] == "https://api.deepseek.com"
    assert calls[0]["api_key"] == "sk-ds-test"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_litellm_client_openai_omits_forced_deepseek_base(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)

        class Msg:
            content = "{}"

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()

    import sys
    import types

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    client = LiteLLMClient(
        OrchestratorConfig(provider="openai", model="openai/gpt-4o-mini")
    )
    client.complete_json("sys", "user", "x")
    assert calls[0]["model"] == "openai/gpt-4o-mini"
    assert calls[0].get("api_base") in (None, ...)
    # Prefer asserting key is present and api_base not set to DeepSeek:
    assert calls[0].get("api_base") != "https://api.deepseek.com"
    assert calls[0]["api_key"] == "sk-oai"
```

Note for implementer: when `api_base` is `None`, either omit the kwarg or pass `api_base=None`—tests should accept both by checking DeepSeek base is not forced for openai. Prefer **omitting** None kwargs in implementation for cleanliness; then assert `"api_base" not in calls[0]` for openai.

- [ ] **Step 2: Run new tests to verify fail**

Run: `pytest tests/test_llm_providers.py::test_litellm_client_passes_deepseek_kwargs -v`  
Expected: FAIL (completion not receiving api_base/api_key)

- [ ] **Step 3: Update `LiteLLMClient.complete_json`**

Replace the `completion(...)` call construction with:

```python
from autofinetune.llm.providers import resolve_litellm_call

# inside complete_json, after importing completion:
resolved = resolve_litellm_call(self.cfg)
kwargs = {
    "model": resolved.model,
    "temperature": self.cfg.temperature,
    "response_format": {"type": "json_object"},
    "messages": [
        {
            "role": "system",
            "content": system
            + f"\nRespond with a JSON object for schema '{schema_name}'.",
        },
        {"role": "user", "content": user},
    ],
}
if resolved.api_base is not None:
    kwargs["api_base"] = resolved.api_base
if resolved.api_key is not None:
    kwargs["api_key"] = resolved.api_key
resp = completion(**kwargs)
```

Keep existing retry loop and cost estimate. `FatalError` from resolver should **not** be retried—catch/re-raise before the retry loop or let it propagate outside `except Exception`.

Structure:

```python
def complete_json(...):
    try:
        from litellm import completion
    except ImportError as e:
        raise FatalError("litellm is required for cloud orchestrator") from e

    resolved = resolve_litellm_call(self.cfg)  # FatalError propagates

    last_err: Exception | None = None
    for attempt in range(self.cfg.max_retries):
        try:
            kwargs = {...}
            ...
            resp = completion(**kwargs)
            ...
            return json.loads(content)
        except FatalError:
            raise
        except Exception as e:
            last_err = e
            time.sleep(...)
    raise RoundError(...)
```

- [ ] **Step 4: Run provider + fake llm tests**

Run: `pytest tests/test_llm_providers.py tests/test_llm_fake.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/llm/client.py tests/test_llm_providers.py
git commit -m "feat: pass DeepSeek preset kwargs through LiteLLMClient"
```

---

### Task 4: README + full regression

**Files:**
- Modify: `README.md`
- Test: full `pytest`

- [ ] **Step 1: Update README real-training section**

Replace OpenAI-first snippet with:

```markdown
## Real local training

```bash
pip install -e ".[train]"
export DEEPSEEK_API_KEY=...   # default orchestrator: DeepSeek deepseek-v4-flash
# Optional: override model in config.yaml → orchestrator.model: deepseek-v4-pro
# Optional: provider: openai + export OPENAI_API_KEY=...
autofinetune run ./my_input --base-model Qwen/Qwen2.5-7B-Instruct --trainer trl
# or let the orchestrator recommend:
autofinetune run ./my_input --base-model auto
```

DeepSeek API docs: https://api-docs.deepseek.com/zh-cn/
```

Keep fake/CI section unchanged (`AUTOFINETUNE_LLM=fake`).

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`  
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document DeepSeek as default orchestrator provider"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Default provider deepseek + model flash | Task 1 |
| Configurable deepseek-v4-pro | Task 2 tests + config model field |
| api_base + DEEPSEEK_API_KEY preset | Task 2–3 |
| openai / litellm providers | Task 2 |
| Missing key FatalError | Task 2 |
| openai/ prefix routing for DeepSeek | Task 2–3 |
| No thinking mode | Task 3 (no extra_body) |
| README | Task 4 |
| Fake/CI unchanged | Task 4 regression |

## Placeholder / consistency self-review

- No TBD steps; resolver and client kwargs named consistently (`ResolvedLiteLLMCall`, `resolve_litellm_call`).
- Tests assert `api_base` omission for openai after preferring omit-None behavior.
