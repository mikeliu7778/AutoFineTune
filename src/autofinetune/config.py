from __future__ import annotations

import platform
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from autofinetune.schemas import AllowlistEntry, GpuProfile


class OrchestratorConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.2
    max_retries: int = 3
    api_base: str | None = None


class TrainerConfig(BaseModel):
    backend: str = "auto"
    max_seq_length: int = 2048
    default_lora_r: int = 16
    default_lora_alpha: int = 32
    default_epochs: int = 1
    default_lr: float = 2e-4


class DataConfig(BaseModel):
    min_qa_for_full: int = 50
    holdout_ratio: float = 0.2
    target_train_size: int = 200


class BudgetsConfig(BaseModel):
    max_rounds: int = 3
    max_wall_time_sec: int = 86400
    max_llm_cost_usd: float | None = None


class AppConfig(BaseModel):
    base_model: str = "auto"
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    gpu_profile: GpuProfile = Field(default_factory=GpuProfile)
    data: DataConfig = Field(default_factory=DataConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    allowlist: list[AllowlistEntry] = Field(default_factory=list)
    runs_dir: Path = Path("runs")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _default_yaml(name: str) -> dict[str, Any]:
    root = resources.files("autofinetune.defaults")
    with resources.as_file(root.joinpath(name)) as path:
        return _read_yaml(path)


def load_config(path: Path | None = None) -> AppConfig:
    raw = _default_yaml("config.yaml")
    allow_raw = _default_yaml("allowlist.yaml")
    user: dict[str, Any] = {}
    user_has_gpu = False
    if path is not None:
        user = _read_yaml(path)
        user_has_gpu = "gpu_profile" in user
        raw.update(user)
        if "allowlist" in user or "models" in user:
            allow_raw = user.get("allowlist") or user
    models = allow_raw.get("models") or raw.get("allowlist") or []
    raw = {k: v for k, v in raw.items() if k != "allowlist"}
    cfg = AppConfig.model_validate(raw)
    cfg.allowlist = [AllowlistEntry.model_validate(m) for m in models]
    if platform.system() == "Darwin" and not user_has_gpu:
        cfg.gpu_profile = GpuProfile(name="apple-unified-16gb", vram_gb=16)
    return cfg
