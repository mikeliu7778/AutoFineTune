from __future__ import annotations

import json

from autofinetune.config import AppConfig
from autofinetune.errors import FatalError
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import LLMClient
from autofinetune.schemas import AllowlistEntry, BaseModelChoice, GpuProfile


def filter_allowlist(
    allowlist: list[AllowlistEntry], gpu: GpuProfile
) -> list[AllowlistEntry]:
    return [e for e in allowlist if e.min_vram_gb <= gpu.vram_gb]


def select_base_model(
    cfg: AppConfig,
    ingest: IngestResult,
    llm: LLMClient,
    base_model_arg: str | None,
) -> BaseModelChoice:
    requested = (base_model_arg if base_model_arg is not None else cfg.base_model).strip()
    if requested != "auto":
        return BaseModelChoice(model_id=requested, mode="user", rationale="user-specified")

    candidates = filter_allowlist(cfg.allowlist, cfg.gpu_profile)
    if not candidates:
        raise FatalError(
            "No allowlist models fit the GPU profile; pin --base-model or widen allowlist/VRAM"
        )

    payload = {
        "brief": ingest.brief[:4000],
        "route": ingest.route.value,
        "qa_count": len(ingest.qa),
        "gpu": cfg.gpu_profile.model_dump(),
        "candidates": [e.model_dump() for e in candidates],
    }
    out = llm.complete_json(
        system=(
            "You recommend a base HF model for domain LoRA fine-tuning. "
            "Choose ONLY from candidates. Return JSON keys: model_id, rationale."
        ),
        user=json.dumps(payload, ensure_ascii=False),
        schema_name="recommend_model",
    )
    model_id = str(out.get("model_id", "")).strip()
    allowed = {e.id for e in candidates}
    if model_id not in allowed:
        raise FatalError(
            f"LLM recommended '{model_id}' which is not in the GPU-filtered allowlist"
        )
    return BaseModelChoice(
        model_id=model_id,
        mode="auto",
        rationale=str(out.get("rationale", "")),
    )
