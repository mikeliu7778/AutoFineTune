from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DataRoute(str, Enum):
    none = "none"
    partial = "partial"
    full = "full"


class RunStatus(str, Enum):
    created = "created"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"


class QAItem(BaseModel):
    question: str
    answer: str
    source: Literal["user", "synthetic"] = "user"


class AllowlistEntry(BaseModel):
    id: str
    approx_params_b: float
    min_vram_gb: float
    chat_template: str = "chatml"
    notes: str = ""


class GpuProfile(BaseModel):
    name: str = "single-24gb"
    vram_gb: float = 24.0


class LoraHyperparams(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    epochs: int = 1
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8


class RoundPlan(BaseModel):
    data_strategy: str
    target_train_size: int
    lora: LoraHyperparams = Field(default_factory=LoraHyperparams)
    eval_focus: str = "factual domain accuracy"
    notes: str = ""


class DecideResult(BaseModel):
    action: Literal["continue", "stop"]
    hypothesis: str = ""
    reason: str = ""


class RoundMetrics(BaseModel):
    judge_score: float | None = None
    aux_exact_match: float | None = None
    n_eval: int = 0
    judge_error: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class BaseModelChoice(BaseModel):
    model_id: str
    mode: Literal["user", "auto"]
    rationale: str = ""
