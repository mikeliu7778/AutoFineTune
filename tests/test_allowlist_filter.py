from autofinetune.config import load_config
from autofinetune.model_select.selector import filter_allowlist
from autofinetune.schemas import GpuProfile


def test_filter_16gb_includes_small_excludes_8b():
    cfg = load_config(None)
    kept = filter_allowlist(cfg.allowlist, GpuProfile(name="t", vram_gb=16))
    ids = {e.id for e in kept}
    assert "Qwen/Qwen2.5-1.5B-Instruct" in ids
    assert "Qwen/Qwen2.5-3B-Instruct" in ids
    assert "meta-llama/Llama-3.1-8B-Instruct" not in ids
