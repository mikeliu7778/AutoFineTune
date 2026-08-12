from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autofinetune.errors import FatalError
from autofinetune.schemas import (
    BaseModelChoice,
    DecideResult,
    RoundMetrics,
    RoundPlan,
    RunStatus,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecord(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.created
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    input_route: str | None = None
    base_model: BaseModelChoice | None = None
    current_round: int = 0
    best_round: int | None = None
    pause_requested: bool = False
    user_note: str | None = None
    last_error: str | None = None
    llm_cost_usd_est: float = 0.0
    started_at: str | None = None


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def _run_json(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def create(self, input_dir: Path) -> RunRecord:
        if not input_dir.is_dir():
            raise FatalError(f"input_dir not found: {input_dir}")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        dest = self.run_dir(run_id)
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copytree(input_dir, dest / "input")
        (dest / "rounds").mkdir()
        (dest / "adapters").mkdir()
        rec = RunRecord(run_id=run_id)
        self._write(rec)
        return rec

    def load(self, run_id: str) -> RunRecord:
        path = self._run_json(run_id)
        if not path.is_file():
            raise FatalError(f"run not found: {run_id}")
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _write(self, rec: RunRecord) -> None:
        rec.updated_at = _now()
        path = self._run_json(rec.run_id)
        path.write_text(rec.model_dump_json(indent=2), encoding="utf-8")

    def save(self, rec: RunRecord) -> None:
        self._write(rec)

    def request_pause(self, run_id: str) -> None:
        rec = self.load(run_id)
        rec.pause_requested = True
        self._write(rec)

    def clear_pause(self, run_id: str) -> None:
        rec = self.load(run_id)
        rec.pause_requested = False
        self._write(rec)

    def set_status(self, run_id: str, status: RunStatus) -> None:
        rec = self.load(run_id)
        rec.status = status
        self._write(rec)

    def set_base_model(self, run_id: str, choice: BaseModelChoice) -> None:
        rec = self.load(run_id)
        rec.base_model = choice
        self._write(rec)

    def set_route(self, run_id: str, route: str) -> None:
        rec = self.load(run_id)
        rec.input_route = route
        self._write(rec)

    def round_dir(self, run_id: str, round_idx: int) -> Path:
        d = self.run_dir(run_id) / "rounds" / f"r{round_idx}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def adapter_dir(self, run_id: str, round_idx: int) -> Path:
        d = self.run_dir(run_id) / "adapters" / f"r{round_idx}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, path: Path, data: BaseModel | dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, BaseModel):
            path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_plan(self, run_id: str, round_idx: int, plan: RoundPlan) -> Path:
        path = self.round_dir(run_id, round_idx) / "plan.json"
        self.write_json(path, plan)
        return path

    def save_metrics(self, run_id: str, round_idx: int, metrics: RoundMetrics) -> Path:
        path = self.round_dir(run_id, round_idx) / "metrics.json"
        self.write_json(path, metrics)
        return path

    def save_report(self, run_id: str, round_idx: int, text: str) -> Path:
        path = self.round_dir(run_id, round_idx) / "report.md"
        path.write_text(text, encoding="utf-8")
        return path

    def save_decide(self, run_id: str, round_idx: int, decide: DecideResult) -> Path:
        path = self.round_dir(run_id, round_idx) / "decide.json"
        self.write_json(path, decide)
        return path

    def holdout_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "holdout.jsonl"

    def best_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "best.json"

    def update_best(self, run_id: str, round_idx: int, metrics: RoundMetrics) -> None:
        rec = self.load(run_id)
        rec.best_round = round_idx
        self._write(rec)
        self.write_json(
            self.best_path(run_id),
            {"round": round_idx, "metrics": metrics.model_dump()},
        )
