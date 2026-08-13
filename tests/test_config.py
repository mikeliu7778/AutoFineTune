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
