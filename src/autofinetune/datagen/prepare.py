from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from autofinetune.config import AppConfig
from autofinetune.errors import RoundError
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import LLMClient
from autofinetune.schemas import DataRoute, QAItem, RoundPlan


class PrepareResult(BaseModel):
    train: list[QAItem] = Field(default_factory=list)
    holdout: list[QAItem] = Field(default_factory=list)


def write_jsonl(path: Path, items: list[QAItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[QAItem]:
    items: list[QAItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(QAItem.model_validate_json(line))
    return items


def _split(qa: list[QAItem], holdout_ratio: float, seed: int = 7) -> tuple[list[QAItem], list[QAItem]]:
    items = list(qa)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_hold = max(1, int(len(items) * holdout_ratio)) if len(items) > 1 else 1
    n_hold = min(n_hold, len(items) - 1) if len(items) > 1 else len(items)
    holdout = items[:n_hold]
    train = items[n_hold:] or items[:1]
    if len(items) == 1:
        # single example: duplicate risk — keep as train and holdout copy marked for eval only
        holdout = items
        train = items
    return train, holdout


def _synthesize(
    cfg: AppConfig,
    ingest: IngestResult,
    plan: RoundPlan,
    llm: LLMClient,
    n: int,
) -> list[QAItem]:
    user = json.dumps(
        {
            "brief": ingest.brief,
            "docs_text": ingest.docs_text[:12000],
            "n": n,
            "strategy": plan.data_strategy,
        },
        ensure_ascii=False,
    )
    out = llm.complete_json(
        system=(
            "Generate domain QA pairs for fine-tuning. "
            'Return JSON: {"items":[{"question":str,"answer":str,"source":"synthetic"}]}'
        ),
        user=user,
        schema_name="synthesize_qa",
    )
    items = [QAItem.model_validate(x) for x in out.get("items", [])]
    if not items:
        raise RoundError("DataGen returned zero synthetic QA items")
    return items


def prepare_datasets(
    cfg: AppConfig,
    ingest: IngestResult,
    plan: RoundPlan,
    llm: LLMClient,
    existing_holdout: list[QAItem] | None,
) -> PrepareResult:
    if ingest.route == DataRoute.full:
        train, holdout = _split(ingest.qa, cfg.data.holdout_ratio)
        if existing_holdout is not None:
            hold_q = {h.question for h in existing_holdout}
            train = [t for t in ingest.qa if t.question not in hold_q]
            holdout = existing_holdout
        return PrepareResult(train=train, holdout=holdout)

    if ingest.route == DataRoute.partial:
        need = max(plan.target_train_size - len(ingest.qa), 1)
        synth = _synthesize(cfg, ingest, plan, llm, need)
        combined = list(ingest.qa) + synth
        if existing_holdout is not None:
            hold_q = {h.question for h in existing_holdout}
            train = [t for t in combined if t.question not in hold_q]
            return PrepareResult(train=train, holdout=existing_holdout)
        train, holdout = _split(combined, cfg.data.holdout_ratio)
        return PrepareResult(train=train, holdout=holdout)

    # none
    total = max(plan.target_train_size, 5)
    synth = _synthesize(cfg, ingest, plan, llm, total)
    if existing_holdout is not None:
        hold_q = {h.question for h in existing_holdout}
        train = [t for t in synth if t.question not in hold_q] or synth
        return PrepareResult(train=train, holdout=existing_holdout)
    train, holdout = _split(synth, cfg.data.holdout_ratio)
    return PrepareResult(train=train, holdout=holdout)
