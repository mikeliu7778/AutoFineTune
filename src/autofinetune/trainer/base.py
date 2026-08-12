from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from autofinetune.errors import FatalError
from autofinetune.schemas import RoundPlan


class TrainResult(BaseModel):
    output_dir: Path
    backend: str


class TrainerBackend(Protocol):
    def train(
        self,
        base_model_id: str,
        train_jsonl: Path,
        output_dir: Path,
        plan: RoundPlan,
    ) -> TrainResult: ...


def get_trainer(name: str) -> TrainerBackend:
    key = name.strip().lower()
    if key == "fake":
        from autofinetune.trainer.fake import FakeTrainer

        return FakeTrainer()
    if key == "trl":
        from autofinetune.trainer.trl_backend import TRLTrainerBackend

        return TRLTrainerBackend()
    raise FatalError(f"Unknown trainer backend: {name}")
