from pathlib import Path

from autofinetune.eval.predict import get_predict_factory, lookup_predict_factory, trl_predict_factory


def test_get_predict_factory_fake_is_lookup():
    assert get_predict_factory("fake") is lookup_predict_factory


def test_get_predict_factory_trl_is_not_lookup():
    factory = get_predict_factory("trl")
    assert factory is trl_predict_factory
    assert factory is not lookup_predict_factory


def test_lookup_predict_factory_reads_train_jsonl(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"question":"Q0","answer":"A0","source":"user"}\n',
        encoding="utf-8",
    )
    predict = lookup_predict_factory(train_jsonl=train)
    assert predict("Q0") == "A0"
    assert predict("missing") == ""
