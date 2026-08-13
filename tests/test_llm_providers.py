import pytest

from autofinetune.config import OrchestratorConfig
from autofinetune.errors import FatalError
from autofinetune.llm.client import LiteLLMClient
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
    assert "api_base" not in calls[0]
    assert calls[0]["api_key"] == "sk-oai"
