"""Post-process / validate RAG intent JSON against closed enums."""

from __future__ import annotations

from typing import Any

from autofinetune.eval.rag_intent_metrics import _norm_unit, parse_json_answer

VALID_INTENTS = frozenset({"summary", "exercises", "knowledge", "unknown"})
VALID_GRADES = frozenset({"七年级", "八年级", "九年级"})
VALID_VOLUMES = frozenset({"上", "下"})
VALID_SUBJECTS = frozenset({"语文", "数学", "英语"})


def validate_and_fix(
    obj: dict[str, Any] | None,
    *,
    default_confidence_on_fix: float = 0.35,
) -> dict[str, Any] | None:
    """Clamp fields to enums; illegal values become null / unknown.

    - Invalid intent → ``unknown``
    - Invalid grade/volume/subject/unit → ``null``
    - Lowers confidence when any fix was applied
    - Does not invent missing slots (no hallucinated fill)
    """
    if obj is None or not isinstance(obj, dict):
        return None

    out: dict[str, Any] = dict(obj)
    fixed = False

    intent = out.get("intent")
    if intent not in VALID_INTENTS:
        out["intent"] = "unknown"
        fixed = True

    grade = out.get("grade")
    if grade is not None and grade not in VALID_GRADES:
        out["grade"] = None
        fixed = True

    volume = out.get("volume")
    if volume is not None and volume not in VALID_VOLUMES:
        out["volume"] = None
        fixed = True

    subject = out.get("subject")
    if subject is not None and subject not in VALID_SUBJECTS:
        out["subject"] = None
        fixed = True

    unit = _norm_unit(out.get("unit"))
    if unit is None:
        if out.get("unit") is not None:
            fixed = True
        out["unit"] = None
    elif isinstance(unit, int) and unit >= 1:
        if out.get("unit") != unit:
            fixed = True
        out["unit"] = unit
    else:
        out["unit"] = None
        fixed = True

    # Strip nested garbage in raw_spans if present but keep if dict
    raw = out.get("raw_spans")
    if raw is not None and not isinstance(raw, dict):
        out.pop("raw_spans", None)
        fixed = True

    conf = out.get("confidence")
    try:
        conf_f = float(conf) if conf is not None else 1.0
    except (TypeError, ValueError):
        conf_f = default_confidence_on_fix
        fixed = True
    if fixed:
        conf_f = min(conf_f, default_confidence_on_fix)
    out["confidence"] = max(0.0, min(1.0, conf_f))

    return out


def parse_and_validate(text: str) -> dict[str, Any] | None:
    return validate_and_fix(parse_json_answer(text))
