import pytest

from autofinetune.config import load_config
from autofinetune.errors import FatalError
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import FakeLLMClient
from autofinetune.model_select.selector import filter_allowlist, select_base_model
from autofinetune.schemas import DataRoute


def _ingest():
    return IngestResult(route=DataRoute.none, brief="billing domain", docs_text="", qa=[])


def test_user_pin_wins():
    cfg = load_config(None)
    llm = FakeLLMClient(handlers={})
    choice = select_base_model(
        cfg, _ingest(), llm, base_model_arg="Qwen/Qwen2.5-7B-Instruct"
    )
    assert choice.mode == "user"
    assert choice.model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_empty_pin_is_fatal():
    cfg = load_config(None)
    llm = FakeLLMClient(handlers={})
    with pytest.raises(FatalError, match="non-empty"):
        select_base_model(cfg, _ingest(), llm, base_model_arg="   ")


def test_pin_outside_allowlist_allowed_with_note():
    cfg = load_config(None)
    llm = FakeLLMClient(handlers={})
    choice = select_base_model(
        cfg, _ingest(), llm, base_model_arg="custom/org-model-not-listed"
    )
    assert choice.mode == "user"
    assert "outside allowlist" in choice.rationale


def test_pin_min_vram_exceeds_gpu_is_fatal():
    cfg = load_config(None)
    cfg.gpu_profile.vram_gb = 4
    llm = FakeLLMClient(handlers={})
    with pytest.raises(FatalError, match="min_vram"):
        select_base_model(
            cfg, _ingest(), llm, base_model_arg="Qwen/Qwen2.5-7B-Instruct"
        )


def test_auto_uses_llm_within_allowlist():
    cfg = load_config(None)
    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "rationale": "smaller safer fit",
            }
        }
    )
    choice = select_base_model(cfg, _ingest(), llm, base_model_arg="auto")
    assert choice.mode == "auto"
    assert choice.model_id == "Qwen/Qwen2.5-3B-Instruct"


def test_auto_rejects_model_outside_filtered_allowlist():
    cfg = load_config(None)
    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "something/not-on-list",
                "rationale": "bad",
            }
        }
    )
    with pytest.raises(FatalError):
        select_base_model(cfg, _ingest(), llm, base_model_arg="auto")


def test_filter_allowlist_respects_vram():
    cfg = load_config(None)
    cfg.gpu_profile.vram_gb = 8
    filtered = filter_allowlist(cfg.allowlist, cfg.gpu_profile)
    assert all(e.min_vram_gb <= 8 for e in filtered)
