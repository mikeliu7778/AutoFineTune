# DeepSeek Provider Preset Design

**Date:** 2026-08-13  
**Status:** Approved for implementation  
**Scope:** First-class DeepSeek orchestrator preset via LiteLLM (config + docs + tests)

## Problem

AutoFineTune’s orchestrator already uses LiteLLM, but defaults and docs assume OpenAI (`openai/gpt-4o-mini` + `OPENAI_API_KEY`). DeepSeek exposes an [OpenAI-compatible API](https://api-docs.deepseek.com/zh-cn/) at `https://api.deepseek.com`, yet users must manually wire model strings and env vars. We want a one-click DeepSeek preset as the project default.

## Goals

- Default orchestrator provider is **DeepSeek**, model **`deepseek-v4-flash`**.
- Users can override model to **`deepseek-v4-pro`** (or another DeepSeek model id) via config.
- Setting `provider: deepseek` automatically applies `api_base` and reads `DEEPSEEK_API_KEY`.
- OpenAI and generic LiteLLM remain available via explicit `provider`.
- Missing API key fails fast with a clear `FatalError`.
- CI / `AUTOFINETUNE_LLM=fake` path unchanged.

## Non-goals

- Dedicated `DeepSeekClient` separate from LiteLLM.
- Enabling DeepSeek **thinking** / `reasoning_effort` in v1 (JSON orchestration stability first).
- Provider-metered cost tracking (keep fixed `EST_COST_USD_PER_CALL`).
- Changing train base-model allowlist (train models remain HF ids; this is orchestrator-only).

## Approach

**LiteLLM + `provider` preset** on existing `LiteLLMClient`.

Resolve preset before each `completion` call (or once in `__init__` and reuse):

1. Map `provider` → preset `api_base` and required env key name. Do **not** rewrite `cfg.model` at runtime; package defaults live in `OrchestratorConfig` / `defaults/config.yaml`.
2. Pass resolved `api_base` / `api_key` into LiteLLM `completion` (user `api_base` overrides preset).
3. For DeepSeek, call LiteLLM with OpenAI-compatible routing: model as `openai/<id>` plus `api_base=https://api.deepseek.com`, so newer ids (`deepseek-v4-flash`, `deepseek-v4-pro`) do not depend on LiteLLM’s built-in DeepSeek alias list.
4. Keep `response_format={"type": "json_object"}`, retries, and cost estimate behavior.

## Config

Extend `OrchestratorConfig`:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `provider` | `str` | `"deepseek"` | `deepseek` \| `openai` \| `litellm` |
| `model` | `str` | `"deepseek-v4-flash"` | Overridable (e.g. `deepseek-v4-pro`) |
| `temperature` | `float` | `0.2` | Unchanged |
| `max_retries` | `int` | `3` | Unchanged |
| `api_base` | `str \| None` | `None` | Optional override of provider preset |

### Provider presets

| `provider` | Default `model` (if using package defaults) | `api_base` | Env var |
|------------|-----------------------------------------------|------------|---------|
| `deepseek` | `deepseek-v4-flash` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `openai` | `openai/gpt-4o-mini` | LiteLLM default (omit) | `OPENAI_API_KEY` |
| `litellm` | whatever user sets in `model` | omit unless `api_base` set | user-managed |

Update `src/autofinetune/defaults/config.yaml` to:

```yaml
orchestrator:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.2
  max_retries: 3
```

**Key missing:** when building/using the real LiteLLM client (not fake), if the resolved env var is empty → raise `FatalError` naming the expected variable (e.g. `DEEPSEEK_API_KEY is required when orchestrator.provider=deepseek`).

**Model override rule:** user-supplied `orchestrator.model` always wins. Defaults only apply via config/defaults files; switching `provider` in a user yaml without changing `model` keeps the user’s model string as written (no silent rewrite at runtime).

## Client behavior

- File: `src/autofinetune/llm/client.py` (`LiteLLMClient`).
- Add a small resolver helper (same module or `llm/providers.py`) that returns `(litellm_model, api_base | None, api_key | None)`.
- DeepSeek → `litellm_model = "openai/" + bare_id` where `bare_id` strips an accidental `openai/` prefix if present; never double-prefix.
- OpenAI → pass `cfg.model` through (expect `openai/...` style); require `OPENAI_API_KEY`.
- `litellm` → pass `cfg.model` and optional `api_base`; no AutoFineTune-level key enforcement (LiteLLM’s own errors apply).
- Thinking / `extra_body` for DeepSeek: **not** sent in v1.

CLI (`cli.py`): still `AUTOFINETUNE_LLM=fake` → `FakeLLMClient`; else `LiteLLMClient(cfg.orchestrator)`.

## Docs

Update `README.md`:

- Real training quick start uses `export DEEPSEEK_API_KEY=...`.
- Note default model `deepseek-v4-flash`; example override to `deepseek-v4-pro`.
- Short example for `provider: openai` with `OPENAI_API_KEY`.

Optional: one-line pointer to DeepSeek API docs.

## Testing

- Unit tests (mock `litellm.completion`):
  - DeepSeek preset asserts `model` starts with `openai/`, `api_base == https://api.deepseek.com`, `api_key` from env.
  - `model: deepseek-v4-pro` flows through as `openai/deepseek-v4-pro`.
  - Missing `DEEPSEEK_API_KEY` → `FatalError`.
  - `provider: openai` uses `OPENAI_API_KEY` and does not force DeepSeek `api_base`.
- Existing fake/orchestrator/CLI integration tests remain green without network.

## Acceptance

1. Fresh defaults + `DEEPSEEK_API_KEY` set → LiteLLM calls target DeepSeek `api_base` with `deepseek-v4-flash`.
2. Config `model: deepseek-v4-pro` overrides flash.
3. `provider: openai` restores prior OpenAI-oriented behavior.
4. Missing key → clear fatal error.
5. `AUTOFINETUNE_LLM=fake` and CI suite pass offline.

## Out of scope / follow-ups

- DeepSeek thinking mode for judge/plan quality.
- Mapping LiteLLM native `deepseek/...` model ids if/when they add v4 aliases.
- Persisting provider choice in `run.json` beyond existing config snapshot behavior (if any).
