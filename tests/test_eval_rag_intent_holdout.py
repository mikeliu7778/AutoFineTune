import json

from autofinetune.eval.rag_intent_metrics import (
    aggregate_scores,
    parse_json_answer,
    score_prediction,
)


def test_score_prediction_exact_match():
    gold = {
        "intent": "summary",
        "grade": "八年级",
        "volume": "上",
        "unit": 1,
        "subject": None,
    }
    pred = {
        "intent": "summary",
        "grade": "八年级",
        "volume": "上",
        "unit": 1,
        "subject": None,
        "confidence": 0.9,
        "raw_spans": {"grade_mention": "初二"},
    }
    s = score_prediction(gold, pred)
    assert s["full"] is True
    assert s["grade"] is True


def test_score_prediction_grade_mismatch():
    gold = {"intent": "summary", "grade": "八年级", "volume": "上", "unit": 1, "subject": None}
    pred = {"intent": "summary", "grade": "初二", "volume": "上", "unit": 1, "subject": None}
    s = score_prediction(gold, pred)
    assert s["grade"] is False
    assert s["full"] is False
    assert s["intent"] is True


def test_score_prediction_unit_string_int():
    gold = {"intent": "summary", "grade": "七年级", "volume": "下", "unit": 2, "subject": "英语"}
    pred = {"intent": "summary", "grade": "七年级", "volume": "下", "unit": "2", "subject": "英语"}
    s = score_prediction(gold, pred)
    assert s["unit"] is True
    assert s["full"] is True


def test_parse_json_answer_fenced():
    raw = '```json\n{"intent":"unknown","grade":null,"volume":null,"unit":null,"subject":null}\n```'
    obj = parse_json_answer(raw)
    assert obj is not None
    assert obj["intent"] == "unknown"
    assert obj["grade"] is None


def test_parse_json_answer_embedded():
    raw = '好的：{"intent":"exercises","grade":"九年级","volume":"上","unit":3,"subject":null}完'
    obj = parse_json_answer(raw)
    assert obj is not None
    assert obj["unit"] == 3


def test_aggregate_scores():
    rows = [
        score_prediction(
            {"intent": "summary", "grade": "八年级", "volume": "上", "unit": 1, "subject": None},
            {"intent": "summary", "grade": "八年级", "volume": "上", "unit": 1, "subject": None},
        ),
        score_prediction(
            {"intent": "summary", "grade": "八年级", "volume": "上", "unit": 1, "subject": None},
            {"intent": "summary", "grade": "七年级", "volume": "上", "unit": 1, "subject": None},
        ),
    ]
    m = aggregate_scores(rows)
    assert m["n"] == 2
    assert m["grade"] == 0.5
    assert m["intent"] == 1.0
    assert m["full"] == 0.5


def test_roundtrip_holdout_line():
    line = {
        "question": "初中2年级上册第一单元的总结",
        "answer": json.dumps(
            {
                "intent": "summary",
                "grade": "八年级",
                "volume": "上",
                "unit": 1,
                "subject": None,
                "confidence": 1.0,
            },
            ensure_ascii=False,
        ),
    }
    gold = parse_json_answer(line["answer"])
    assert gold is not None
    s = score_prediction(gold, gold)
    assert s["full"] is True
