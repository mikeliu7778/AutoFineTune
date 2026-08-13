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
