import platform

from autofinetune.config import load_config


def test_load_config_defaults_max_rounds_and_auto_base():
    cfg = load_config(None)
    assert cfg.budgets.max_rounds >= 1
    assert cfg.base_model == "auto"
    assert cfg.trainer.backend == "auto"
    assert cfg.data.min_qa_for_full >= 1
    assert len(cfg.allowlist) >= 1
    assert cfg.orchestrator.provider == "deepseek"
    assert cfg.orchestrator.model == "deepseek-v4-flash"
    assert cfg.orchestrator.api_base is None


def test_trainer_backend_defaults_to_auto():
    cfg = load_config(None)
    assert cfg.trainer.backend == "auto"


def test_darwin_default_gpu_profile(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    cfg = load_config(None)
    assert cfg.gpu_profile.name == "apple-unified-16gb"
    assert cfg.gpu_profile.vram_gb == 16


def test_user_gpu_profile_not_overridden_on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    p = tmp_path / "c.yaml"
    p.write_text("gpu_profile:\n  name: custom\n  vram_gb: 24\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.gpu_profile.name == "custom"
    assert cfg.gpu_profile.vram_gb == 24
