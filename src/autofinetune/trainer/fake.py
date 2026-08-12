from __future__ import annotations

import json
from pathlib import Path

from autofinetune.schemas import RoundPlan
from autofinetune.trainer.base import TrainResult


class FakeTrainer:
    def train(
        self,
        base_model_id: str,
        train_jsonl: Path,
        output_dir: Path,
        plan: RoundPlan,
    ) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = {
            "base_model_id": base_model_id,
            "train_jsonl": str(train_jsonl),
            "lora": plan.lora.model_dump(),
        }
        (output_dir / "fake_adapter.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8"
        )
        return TrainResult(output_dir=output_dir, backend="fake")
