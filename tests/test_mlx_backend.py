import json
from pathlib import Path

import pytest
import yaml

from autofinetune.errors import FatalError, RoundError
from autofinetune.schemas import LoraHyperparams, RoundPlan
from autofinetune.trainer.base import get_trainer
from autofinetune.trainer.mlx_backend import MLXTrainerBackend, _build_mlx_lora_cmd


def _plan(**kwargs) -> RoundPlan:
    lora_kwargs = kwargs.pop("lora", {})
    return RoundPlan(
        data_strategy="x",
        target_train_size=1,
        lora=LoraHyperparams(**lora_kwargs) if lora_kwargs else LoraHyperparams(),
        **kwargs,
    )


def test_get_trainer_mlx():
    t = get_trainer("mlx")
    assert isinstance(t, MLXTrainerBackend)


def test_build_mlx_lora_cmd_includes_adapter_path(tmp_path: Path):
    data_dir = tmp_path / "data"
    adapter = tmp_path / "adapter"
    config = tmp_path / "lora_config.yaml"
    cmd = _build_mlx_lora_cmd(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        data_dir=data_dir,
        adapter_path=adapter,
        batch_size=1,
        learning_rate=2e-4,
        iters=4,
        grad_accumulation_steps=8,
        config_path=config,
    )
    assert cmd[1:4] == ["-m", "mlx_lm", "lora"]
    assert "--adapter-path" in cmd
    assert cmd[cmd.index("--adapter-path") + 1] == str(adapter)
    assert "--data" in cmd
    assert cmd[cmd.index("--data") + 1] == str(data_dir)
    assert "-c" in cmd
    assert "--grad-accumulation-steps" in cmd
    assert "--iters" in cmd
    assert cmd[cmd.index("--iters") + 1] == "4"


def test_missing_mlx_import_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom() -> None:
        raise ModuleNotFoundError("No module named 'mlx_lm'")

    monkeypatch.setattr(
        "autofinetune.trainer.mlx_backend._ensure_mlx_lm", boom
    )
    train = tmp_path / "train.jsonl"
    train.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")
    with pytest.raises(FatalError, match=r"autofinetune\[mlx\]"):
        MLXTrainerBackend().train(
            "Qwen/Qwen2.5-1.5B-Instruct",
            train,
            tmp_path / "adapter",
            _plan(),
        )


def test_mlx_backend_writes_completions_and_calls_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "autofinetune.trainer.mlx_backend._ensure_mlx_lm", lambda: None
    )
    captured: dict = {}

    def fake_run(**kwargs):
        data_file = Path(kwargs["data_dir"]) / "train.jsonl"
        captured["line"] = json.loads(
            data_file.read_text(encoding="utf-8").splitlines()[0]
        )
        captured["adapter_path"] = Path(kwargs["adapter_path"])
        captured["iters"] = kwargs["iters"]
        captured["batch_size"] = kwargs["batch_size"]
        captured["grad_accumulation_steps"] = kwargs["grad_accumulation_steps"]
        captured["learning_rate"] = kwargs["learning_rate"]
        captured["config"] = yaml.safe_load(
            Path(kwargs["config_path"]).read_text(encoding="utf-8")
        )

    monkeypatch.setattr(
        "autofinetune.trainer.mlx_backend._run_mlx_lora", fake_run
    )

    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"question":"What?","answer":"42","source":"user"}\n'
        '{"question":"Who?","answer":"me"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "adapter"
    result = MLXTrainerBackend().train(
        "Qwen/Qwen2.5-1.5B-Instruct",
        train,
        out,
        _plan(
            lora={
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "epochs": 3,
                "learning_rate": 1e-4,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
            }
        ),
    )

    assert result.backend == "mlx"
    assert result.output_dir == out
    assert captured["adapter_path"] == out
    assert captured["line"]["prompt"] == "### Question:\nWhat?\n\n### Answer:\n"
    assert captured["line"]["completion"] == "42"
    assert captured["iters"] == 6  # epochs * n_rows
    assert captured["batch_size"] == 2
    assert captured["grad_accumulation_steps"] == 4
    assert captured["learning_rate"] == 1e-4
    assert captured["config"]["lora_parameters"]["rank"] == 8
    assert captured["config"]["lora_parameters"]["scale"] == 2.0  # alpha / r
    assert captured["config"]["lora_parameters"]["dropout"] == 0.1


def test_mlx_train_failure_is_round_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "autofinetune.trainer.mlx_backend._ensure_mlx_lm", lambda: None
    )

    def boom(**kwargs):
        raise RuntimeError("cuda-ish mlx crash")

    monkeypatch.setattr(
        "autofinetune.trainer.mlx_backend._run_mlx_lora", boom
    )
    train = tmp_path / "train.jsonl"
    train.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")
    with pytest.raises(RoundError, match="MLX training failed"):
        MLXTrainerBackend().train(
            "m", train, tmp_path / "out", _plan()
        )
