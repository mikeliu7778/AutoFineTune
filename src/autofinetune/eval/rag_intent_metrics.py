"""Field-level metrics for RAG query-intent JSON predictions."""

from __future__ import annotations

import json
import re
from typing import Any


CORE_FIELDS = ("intent", "grade", "volume", "unit", "subject")


def parse_json_answer(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output (raw or fenced)."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fallback: first {...} span
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _norm_unit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value  # type: ignore[return-value]


def score_prediction(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, bool]:
    """Compare core fields; ignore confidence / raw_spans for match."""
    g_unit = _norm_unit(gold.get("unit"))
    p_unit = _norm_unit(pred.get("unit"))
    field_ok = {
        "intent": gold.get("intent") == pred.get("intent"),
        "grade": gold.get("grade") == pred.get("grade"),
        "volume": gold.get("volume") == pred.get("volume"),
        "unit": g_unit == p_unit,
        "subject": gold.get("subject") == pred.get("subject"),
    }
    field_ok["full"] = all(field_ok[k] for k in CORE_FIELDS)
    return field_ok


def aggregate_scores(rows: list[dict[str, bool]]) -> dict[str, float]:
    if not rows:
        return {**{k: 0.0 for k in CORE_FIELDS}, "full": 0.0, "n": 0.0, "parse_ok": 0.0}
    n = len(rows)
    out: dict[str, float] = {"n": float(n)}
    for key in (*CORE_FIELDS, "full"):
        out[key] = sum(1 for r in rows if r.get(key)) / n
    return out
