from __future__ import annotations

import json
from collections.abc import Callable

from autofinetune.llm.client import LLMClient
from autofinetune.schemas import QAItem, RoundMetrics


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def evaluate_holdout(
    llm: LLMClient,
    holdout: list[QAItem],
    predict_fn: Callable[[str], str],
) -> RoundMetrics:
    preds: list[dict[str, str]] = []
    exact = 0
    for item in holdout:
        pred = predict_fn(item.question)
        preds.append(
            {
                "question": item.question,
                "gold": item.answer,
                "prediction": pred,
            }
        )
        if _norm(pred) == _norm(item.answer):
            exact += 1
    n = len(holdout)
    aux = (exact / n) if n else 0.0

    try:
        out = llm.complete_json(
            system=(
                "You are grading domain QA predictions. "
                'Return JSON {"scores":[{"question":str,"score":0..1,"rationale":str}]}. '
                "score 1 = fully correct, 0 = wrong."
            ),
            user=json.dumps({"items": preds}, ensure_ascii=False),
            schema_name="judge_qa",
        )
        scores = [float(x["score"]) for x in out.get("scores", [])]
        judge = (sum(scores) / len(scores)) if scores else None
        return RoundMetrics(
            judge_score=judge,
            aux_exact_match=aux,
            n_eval=n,
            extras={"predictions": preds},
        )
    except Exception as e:  # noqa: BLE001 — judge soft-fail per spec
        return RoundMetrics(
            judge_score=None,
            aux_exact_match=aux,
            n_eval=n,
            judge_error=str(e),
            extras={"predictions": preds},
        )
