from autofinetune.eval.rag_intent_validate import parse_and_validate, validate_and_fix


def test_validate_clamps_illegal_intent_and_grade():
    out = validate_and_fix(
        {
            "intent": "weather",
            "grade": "高一",
            "volume": "上",
            "unit": 1,
            "subject": "今天",
            "confidence": 1.0,
        }
    )
    assert out is not None
    assert out["intent"] == "unknown"
    assert out["grade"] is None
    assert out["volume"] == "上"
    assert out["unit"] == 1
    assert out["subject"] is None
    assert out["confidence"] <= 0.35


def test_validate_keeps_valid_alias_normalized_fields():
    out = validate_and_fix(
        {
            "intent": "summary",
            "grade": "八年级",
            "volume": "上",
            "unit": 1,
            "subject": None,
            "confidence": 0.9,
        }
    )
    assert out is not None
    assert out["intent"] == "summary"
    assert out["grade"] == "八年级"
    assert out["confidence"] == 0.9


def test_validate_unit_string():
    out = validate_and_fix(
        {
            "intent": "exercises",
            "grade": "七年级",
            "volume": "下",
            "unit": "3",
            "subject": "数学",
            "confidence": 1.0,
        }
    )
    assert out is not None
    assert out["unit"] == 3


def test_parse_and_validate_bad_json():
    assert parse_and_validate("not json") is None


def test_validate_volume_garbage():
    out = validate_and_fix(
        {
            "intent": "knowledge",
            "grade": "九年级",
            "volume": "四",
            "unit": 4,
            "subject": None,
            "confidence": 1.0,
        }
    )
    assert out is not None
    assert out["volume"] is None
    assert out["confidence"] <= 0.35
