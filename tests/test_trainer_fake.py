from pathlib import Path

from autofinetune.schemas import RoundPlan
from autofinetune.trainer.fake import FakeTrainer
from autofinetune.trainer.base import get_trainer


def test_fake_trainer_writes_adapter_marker(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"question":"q","answer":"a","source":"user"}\n', encoding="utf-8"
    )
    out = tmp_path / "adapter"
    result = FakeTrainer().train(
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        train_jsonl=train,
        output_dir=out,
        plan=RoundPlan(data_strategy="x", target_train_size=1),
    )
    assert result.output_dir == out
    assert (out / "fake_adapter.json").is_file()


def test_get_trainer_fake():
    t = get_trainer("fake")
    assert isinstance(t, FakeTrainer)
