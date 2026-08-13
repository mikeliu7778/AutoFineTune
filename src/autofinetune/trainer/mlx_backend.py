from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from autofinetune.errors import FatalError, RoundError
from autofinetune.schemas import LoraHyperparams, RoundPlan
from autofinetune.trainer.base import TrainResult

_MLX_INSTALL = "pip install 'autofinetune[mlx]'"


def _ensure_mlx_lm() -> None:
    import mlx_lm  # noqa: F401


def _load_qa_rows(train_jsonl: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in train_jsonl.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append({"question": obj["question"], "answer": obj["answer"]})
    return rows


def _write_completions_jsonl(data_dir: Path, rows: list[dict[str, str]]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "train.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            rec = {
                "prompt": (
                    f"### Question:\n{row['question']}\n\n### Answer:\n"
                ),
                "completion": row["answer"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def _write_lora_config(path: Path, lora: LoraHyperparams) -> Path:
    # mlx-lm only accepts LoRA rank/scale/dropout via YAML (not CLI flags).
    payload = {
        "lora_parameters": {
            "rank": lora.r,
            "scale": lora.alpha,
            "dropout": lora.dropout,
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _iters_for_plan(epochs: int, n_rows: int) -> int:
    return max(epochs * max(n_rows, 1), 1)


def _build_mlx_lora_cmd(
    *,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    batch_size: int,
    learning_rate: float,
    iters: int,
    grad_accumulation_steps: int,
    config_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--iters",
        str(iters),
        "--grad-accumulation-steps",
        str(grad_accumulation_steps),
        "-c",
        str(config_path),
    ]


def _run_mlx_lora(
    *,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    batch_size: int,
    learning_rate: float,
    iters: int,
    grad_accumulation_steps: int,
    config_path: Path,
) -> None:
    cmd = _build_mlx_lora_cmd(
        model=model,
        data_dir=data_dir,
        adapter_path=adapter_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        iters=iters,
        grad_accumulation_steps=grad_accumulation_steps,
        config_path=config_path,
    )
    subprocess.run(cmd, check=True)


class MLXTrainerBackend:
    def train(
        self,
        base_model_id: str,
        train_jsonl: Path,
        output_dir: Path,
        plan: RoundPlan,
    ) -> TrainResult:
        try:
            _ensure_mlx_lm()
        except ImportError as e:
            raise FatalError(
                f"MLX backend requires extras: {_MLX_INSTALL}"
            ) from e

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = _load_qa_rows(train_jsonl)
            iters = _iters_for_plan(plan.lora.epochs, len(rows))
            with tempfile.TemporaryDirectory(prefix="aft-mlx-") as tmp:
                data_dir = Path(tmp) / "data"
                _write_completions_jsonl(data_dir, rows)
                config_path = _write_lora_config(
                    Path(tmp) / "lora_config.yaml", plan.lora
                )
                _run_mlx_lora(
                    model=base_model_id,
                    data_dir=data_dir,
                    adapter_path=output_dir,
                    batch_size=plan.lora.per_device_train_batch_size,
                    learning_rate=plan.lora.learning_rate,
                    iters=iters,
                    grad_accumulation_steps=plan.lora.gradient_accumulation_steps,
                    config_path=config_path,
                )
            return TrainResult(output_dir=output_dir, backend="mlx")
        except FatalError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RoundError(f"MLX training failed: {e}") from e
