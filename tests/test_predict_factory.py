import sys
import types
from pathlib import Path

import pytest

from autofinetune.errors import FatalError, RoundError
from autofinetune.eval.predict import (
    get_predict_factory,
    lookup_predict_factory,
    mlx_predict_factory,
    trl_predict_factory,
)


def test_get_predict_factory_fake_is_lookup():
    assert get_predict_factory("fake") is lookup_predict_factory


def test_get_predict_factory_trl_is_not_lookup():
    factory = get_predict_factory("trl")
    assert factory is trl_predict_factory
    assert factory is not lookup_predict_factory


def test_get_predict_factory_mlx_is_registered():
    factory = get_predict_factory("mlx")
    assert factory is mlx_predict_factory
    assert factory is not lookup_predict_factory
    assert factory is not trl_predict_factory


def test_lookup_predict_factory_reads_train_jsonl(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"question":"Q0","answer":"A0","source":"user"}\n',
        encoding="utf-8",
    )
    predict = lookup_predict_factory(train_jsonl=train)
    assert predict("Q0") == "A0"
    assert predict("missing") == ""


def test_mlx_predict_factory_missing_mlx_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    factory = get_predict_factory("mlx")
    with pytest.raises(FatalError, match=r"autofinetune\[mlx\]"):
        factory(base_model_id="Qwen/Qwen2.5-1.5B-Instruct", adapter_dir=tmp_path)


def test_mlx_predict_uses_chat_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict = {}

    class FakeTok:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            captured["messages"] = messages
            captured["tokenize"] = tokenize
            captured["add_generation_prompt"] = add_generation_prompt
            return "<chat>PROMPT"

    def fake_load(model_id, adapter_path=None):
        captured["model_id"] = model_id
        captured["adapter_path"] = adapter_path
        return ("model", FakeTok())

    def fake_generate(model, tokenizer, prompt=None, max_tokens=None, **kwargs):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return "  42  "

    fake = types.ModuleType("mlx_lm")
    fake.load = fake_load
    fake.generate = fake_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    predict = mlx_predict_factory(
        base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_dir=tmp_path / "adapter",
    )
    assert predict("What?") == "42"
    assert captured["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert captured["adapter_path"] == str(tmp_path / "adapter")
    assert captured["messages"] == [
        {"role": "user", "content": "### Question:\nWhat?\n\n### Answer:\n"}
    ]
    assert captured["tokenize"] is False
    assert captured["add_generation_prompt"] is True
    assert captured["prompt"] == "<chat>PROMPT"
    assert captured["max_tokens"] == 64


def test_mlx_predict_falls_back_without_chat_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict = {}

    class FakeTok:
        pass

    def fake_load(model_id, adapter_path=None):
        return ("model", FakeTok())

    def fake_generate(model, tokenizer, prompt=None, max_tokens=None, **kwargs):
        captured["prompt"] = prompt
        return "ok"

    fake = types.ModuleType("mlx_lm")
    fake.load = fake_load
    fake.generate = fake_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    predict = mlx_predict_factory(
        base_model_id="m", adapter_dir=tmp_path / "adapter"
    )
    assert predict("Q") == "ok"
    assert captured["prompt"] == "### Question:\nQ\n\n### Answer:\n"


def test_mlx_predict_load_failure_is_round_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("bad adapter")

    fake = types.ModuleType("mlx_lm")
    fake.load = boom
    fake.generate = lambda *a, **k: ""
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    with pytest.raises(RoundError, match="MLX predict load failed"):
        mlx_predict_factory(base_model_id="m", adapter_dir=tmp_path)
