# AutoFineTune v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a CLI-driven closed-loop domain-knowledge fine-tuning agent: ingest flexible inputs, select base model (pin or LLM recommend), iterate plan→data→train→eval→decide locally with LoRA, pause/resume via disk state.

**Architecture:** Fixed outer-round state machine with LiteLLM orchestrator; pluggable `TrainerBackend` (fake for CI, TRL+PEFT for local GPU); all run state under `runs/<id>/`.

**Tech Stack:** Python 3.11+, Typer, Rich, Pydantic v2, pydantic-settings, PyYAML, LiteLLM, Hugging Face transformers + PEFT + TRL, pytest.

**Spec:** `docs/superpowers/specs/2026-08-12-autofinetune-design.md`

## Global Constraints

- Domain-knowledge fine-tuning only in v1 (no DPO / general SFT product mode).
- Trainer abstraction required; v1 backends: `fake` (tests) + `trl` (local LoRA); Unsloth optional later.
- Base model: user pin or `auto` recommend from GPU-filtered allowlist; selected once per run.
- Holdout never used for training.
- Pause only at round boundaries; resume must not change recorded base model.
- Prefer OSS libraries listed in the spec; no LangGraph/AutoGen as core runtime.
- CI must pass with mocked LLM + fake trainer (no GPU).
- Git commits: author/committer `mike <mliu36292@gmail.com>`; never add Cursor co-author trailers.

---

## File structure (create as tasks proceed)

```text
pyproject.toml
README.md
src/autofinetune/
  __init__.py
  __main__.py
  cli.py
  config.py
  schemas.py
  errors.py
  llm/
    __init__.py
    client.py
  ingest/
    __init__.py
    bundle.py
  model_select/
    __init__.py
    selector.py
  datagen/
    __init__.py
    prepare.py
  trainer/
    __init__.py
    base.py
    fake.py
    trl_backend.py
  eval/
    __init__.py
    runner.py
  store/
    __init__.py
    run_store.py
  orchestrator/
    __init__.py
    loop.py
  defaults/
    config.yaml
    allowlist.yaml
tests/
  conftest.py
  test_config.py
  test_ingest.py
  test_store.py
  test_model_select.py
  test_datagen.py
  test_trainer_fake.py
  test_eval.py
  test_orchestrator.py
  test_cli_integration.py
  fixtures/
    minimal_input/
      brief.md
```

---

### Task 1: Project scaffold, schemas, config

**Files:**
- Create: `pyproject.toml`
- Create: `src/autofinetune/__init__.py`
- Create: `src/autofinetune/__main__.py`
- Create: `src/autofinetune/errors.py`
- Create: `src/autofinetune/schemas.py`
- Create: `src/autofinetune/config.py`
- Create: `src/autofinetune/defaults/config.yaml`
- Create: `src/autofinetune/defaults/allowlist.yaml`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `AppConfig`, `load_config(path: Path | None) -> AppConfig`, `FatalError`, schemas `QAItem`, `DataRoute`, `GpuProfile`, `AllowlistEntry`, `RoundPlan`, `DecideResult`, `RunStatus`

- [ ] **Step 1: Write failing test for config defaults**

```python
# tests/test_config.py
from autofinetune.config import load_config


def test_load_config_defaults_max_rounds_and_auto_base():
    cfg = load_config(None)
    assert cfg.budgets.max_rounds >= 1
    assert cfg.base_model == "auto"
    assert cfg.trainer.backend == "trl"
    assert cfg.data.min_qa_for_full >= 1
    assert len(cfg.allowlist) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/liubing/work/autofinetune && pip install -e ".[dev]" 2>/dev/null; pytest tests/test_config.py::test_load_config_defaults_max_rounds_and_auto_base -v`

Expected: FAIL (package/module missing) until scaffold exists — if install fails first, create `pyproject.toml` then re-run; still FAIL on missing `load_config`.

- [ ] **Step 3: Create `pyproject.toml` and package skeleton**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "autofinetune"
version = "0.1.0"
description = "Closed-loop domain-knowledge fine-tuning agent"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "typer>=0.12",
  "rich>=13.0",
  "pydantic>=2.6",
  "pydantic-settings>=2.2",
  "pyyaml>=6.0",
  "litellm>=1.40",
  "httpx>=0.27",
]

[project.optional-dependencies]
train = [
  "torch",
  "transformers>=4.44",
  "peft>=0.12",
  "trl>=0.9",
  "datasets>=2.20",
  "accelerate>=0.33",
  "pypdf>=4.0",
]
dev = [
  "pytest>=8.0",
  "autofinetune[train]",
]

[project.scripts]
autofinetune = "autofinetune.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/autofinetune"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/autofinetune/__init__.py
__version__ = "0.1.0"
```

```python
# src/autofinetune/__main__.py
from autofinetune.cli import app

if __name__ == "__main__":
    app()
```

```python
# src/autofinetune/errors.py
class AutoFineTuneError(Exception):
    """Base error."""


class FatalError(AutoFineTuneError):
    """Unrecoverable; CLI should exit non-zero immediately."""


class RoundError(AutoFineTuneError):
    """Recoverable at round level; orchestrator may replan."""
```

- [ ] **Step 4: Implement schemas + defaults + `load_config`**

```yaml
# src/autofinetune/defaults/allowlist.yaml
models:
  - id: Qwen/Qwen2.5-7B-Instruct
    approx_params_b: 7.6
    min_vram_gb: 16
    chat_template: qwen
    notes: "Default 24GB-friendly instruct model"
  - id: Qwen/Qwen2.5-3B-Instruct
    approx_params_b: 3.1
    min_vram_gb: 8
    chat_template: qwen
    notes: "Smaller / 8-16GB profile"
  - id: meta-llama/Llama-3.1-8B-Instruct
    approx_params_b: 8.0
    min_vram_gb: 18
    chat_template: llama3
    notes: "Requires HF access token for gated weights"
```

```yaml
# src/autofinetune/defaults/config.yaml
base_model: auto
orchestrator:
  model: openai/gpt-4o-mini
  temperature: 0.2
  max_retries: 3
trainer:
  backend: trl  # trl | fake
  max_seq_length: 2048
  default_lora_r: 16
  default_lora_alpha: 32
  default_epochs: 1
  default_lr: 2.0e-4
gpu_profile:
  name: single-24gb
  vram_gb: 24
data:
  min_qa_for_full: 50
  holdout_ratio: 0.2
  target_train_size: 200
budgets:
  max_rounds: 3
  max_wall_time_sec: 86400
  max_llm_cost_usd: null
```

```python
# src/autofinetune/schemas.py
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
```

```python
# src/autofinetune/config.py
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from autofinetune.schemas import AllowlistEntry, GpuProfile


class OrchestratorConfig(BaseModel):
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.2
    max_retries: int = 3


class TrainerConfig(BaseModel):
    backend: str = "trl"
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
    if path is not None:
        user = _read_yaml(path)
        raw.update(user)
        if "allowlist" in user or "models" in user:
            allow_raw = user.get("allowlist") or user
    models = allow_raw.get("models") or raw.get("allowlist") or []
    raw = {k: v for k, v in raw.items() if k != "allowlist"}
    cfg = AppConfig.model_validate(raw)
    cfg.allowlist = [AllowlistEntry.model_validate(m) for m in models]
    return cfg
```

- [ ] **Step 5: Install and run test**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/autofinetune tests/test_config.py
git commit -m "feat: scaffold package, schemas, and config loading"
```

(Use mike author env; strip Cursor co-author if injected.)

---

### Task 2: Experiment store

**Files:**
- Create: `src/autofinetune/store/__init__.py`
- Create: `src/autofinetune/store/run_store.py`
- Create: `tests/test_store.py`

**Interfaces:**
- Consumes: `RunStatus`, `BaseModelChoice`, `RoundPlan`, `RoundMetrics` from schemas; `AppConfig.runs_dir`
- Produces: `RunStore.create(...) -> RunRecord`, `RunStore.load(run_id)`, `save_round_*`, `request_pause()`, `set_status()`, `update_best()`, path helpers

- [ ] **Step 1: Write failing tests**

```python
# tests/test_store.py
from pathlib import Path

from autofinetune.schemas import BaseModelChoice, RunStatus
from autofinetune.store.run_store import RunStore


def test_create_and_reload_run(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    rec = store.create(input_dir=tmp_path / "in")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "brief.md").write_text("domain", encoding="utf-8")
    rec = store.create(input_dir=tmp_path / "in")
    assert rec.run_id
    assert rec.status == RunStatus.created
    loaded = store.load(rec.run_id)
    assert loaded.run_id == rec.run_id
    assert (store.root / rec.run_id / "run.json").is_file()
    assert (store.root / rec.run_id / "input" / "brief.md").is_file()


def test_pause_flag_round_trip(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "brief.md").write_text("x", encoding="utf-8")
    rec = store.create(input_dir=tmp_path / "in")
    store.request_pause(rec.run_id)
    loaded = store.load(rec.run_id)
    assert loaded.pause_requested is True


def test_set_base_model_persisted(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "brief.md").write_text("x", encoding="utf-8")
    rec = store.create(input_dir=tmp_path / "in")
    choice = BaseModelChoice(model_id="Qwen/Qwen2.5-7B-Instruct", mode="user", rationale="pin")
    store.set_base_model(rec.run_id, choice)
    loaded = store.load(rec.run_id)
    assert loaded.base_model is not None
    assert loaded.base_model.model_id.endswith("7B-Instruct")
```

Fix the duplicate `create` in the first test when implementing — the test above accidentally creates twice; use this corrected first test:

```python
def test_create_and_reload_run(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("domain", encoding="utf-8")
    rec = store.create(input_dir=inp)
    assert rec.run_id
    assert rec.status == RunStatus.created
    loaded = store.load(rec.run_id)
    assert loaded.run_id == rec.run_id
    assert (store.root / rec.run_id / "run.json").is_file()
    assert (store.root / rec.run_id / "input" / "brief.md").is_file()
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_store.py -v`

Expected: FAIL import error

- [ ] **Step 3: Implement `RunStore`**

```python
# src/autofinetune/store/__init__.py
from autofinetune.store.run_store import RunRecord, RunStore

__all__ = ["RunRecord", "RunStore"]
```

```python
# src/autofinetune/store/run_store.py
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
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_store.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/store tests/test_store.py
git commit -m "feat: add run experiment store with pause and artifacts"
```

---

### Task 3: Ingest + routing

**Files:**
- Create: `src/autofinetune/ingest/__init__.py`
- Create: `src/autofinetune/ingest/bundle.py`
- Create: `tests/test_ingest.py`
- Create: `tests/fixtures/minimal_input/brief.md`

**Interfaces:**
- Consumes: `AppConfig.data.min_qa_for_full`, `QAItem`, `DataRoute`
- Produces: `IngestResult(route, brief, docs_text, qa: list[QAItem])`, `ingest_bundle(path, cfg) -> IngestResult`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ingest.py
from pathlib import Path

import pytest

from autofinetune.config import load_config
from autofinetune.errors import FatalError
from autofinetune.ingest.bundle import ingest_bundle
from autofinetune.schemas import DataRoute


def test_brief_only_routes_none(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("Internal wiki about ACME billing", encoding="utf-8")
    cfg = load_config(None)
    result = ingest_bundle(inp, cfg)
    assert result.route == DataRoute.none
    assert "ACME" in result.brief


def test_empty_input_fatal(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    cfg = load_config(None)
    with pytest.raises(FatalError):
        ingest_bundle(inp, cfg)


def test_full_qa_route(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    cfg = load_config(None)
    lines = [
        '{"question":"Q%d","answer":"A%d"}' % (i, i)
        for i in range(cfg.data.min_qa_for_full)
    ]
    (inp / "qa.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = ingest_bundle(inp, cfg)
    assert result.route == DataRoute.full
    assert len(result.qa) == cfg.data.min_qa_for_full


def test_partial_qa_route(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    cfg = load_config(None)
    n = max(1, cfg.data.min_qa_for_full // 5)
    lines = ['{"question":"Q%d","answer":"A%d"}' % (i, i) for i in range(n)]
    (inp / "qa.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (inp / "brief.md").write_text("domain", encoding="utf-8")
    result = ingest_bundle(inp, cfg)
    assert result.route == DataRoute.partial
```

```markdown
# tests/fixtures/minimal_input/brief.md
ACME Corp internal billing policies and refund rules.
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_ingest.py -v`

- [ ] **Step 3: Implement ingest**

```python
# src/autofinetune/ingest/__init__.py
from autofinetune.ingest.bundle import IngestResult, ingest_bundle

__all__ = ["IngestResult", "ingest_bundle"]
```

```python
# src/autofinetune/ingest/bundle.py
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from autofinetune.config import AppConfig
from autofinetune.errors import FatalError
from autofinetune.schemas import DataRoute, QAItem

_DOC_SUFFIXES = {".md", ".txt", ".markdown"}


class IngestResult(BaseModel):
    route: DataRoute
    brief: str = ""
    docs_text: str = ""
    qa: list[QAItem] = Field(default_factory=list)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _load_docs(docs_dir: Path) -> str:
    if not docs_dir.is_dir():
        return ""
    chunks: list[str] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in _DOC_SUFFIXES:
            chunks.append(f"# {path.name}\n{_read_text(path)}")
        elif path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise FatalError("pypdf required to read PDF docs; pip install pypdf") from e
            reader = PdfReader(str(path))
            text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
            if text:
                chunks.append(f"# {path.name}\n{text}")
    return "\n\n".join(chunks).strip()


def _load_qa(path: Path) -> list[QAItem]:
    if not path.is_file():
        return []
    items: list[QAItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            items.append(QAItem.model_validate(raw))
        except Exception as e:
            raise FatalError(f"invalid qa.jsonl line {line_no}: {e}") from e
    return items


def ingest_bundle(input_dir: Path, cfg: AppConfig) -> IngestResult:
    brief_path = input_dir / "brief.md"
    brief = _read_text(brief_path) if brief_path.is_file() else ""
    docs_text = _load_docs(input_dir / "docs")
    qa = _load_qa(input_dir / "qa.jsonl")

    if not brief and not docs_text and not qa:
        raise FatalError(
            "Minimum input required: non-empty brief.md, docs/, or qa.jsonl"
        )

    if not qa:
        route = DataRoute.none
    elif len(qa) >= cfg.data.min_qa_for_full:
        route = DataRoute.full
    else:
        route = DataRoute.partial

    return IngestResult(route=route, brief=brief, docs_text=docs_text, qa=qa)
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_ingest.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/ingest tests/test_ingest.py tests/fixtures
git commit -m "feat: ingest input bundles and route none/partial/full"
```

---

### Task 4: LLM client (LiteLLM + fake)

**Files:**
- Create: `src/autofinetune/llm/__init__.py`
- Create: `src/autofinetune/llm/client.py`
- Create: `tests/conftest.py`
- Modify: add coverage via model_select / later tests — still add `tests/test_llm_fake.py`

**Interfaces:**
- Produces: `LLMClient` protocol with `complete_json(system: str, user: str, schema_name: str) -> dict`
- Produces: `LiteLLMClient`, `FakeLLMClient(handlers: dict[str, Callable])`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm_fake.py
from autofinetune.llm.client import FakeLLMClient


def test_fake_routes_by_schema_name():
    client = FakeLLMClient(
        handlers={
            "recommend_model": lambda system, user: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "fits 24GB",
            }
        }
    )
    out = client.complete_json("sys", "user", "recommend_model")
    assert out["model_id"].startswith("Qwen/")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_llm_fake.py -v`

- [ ] **Step 3: Implement client**

```python
# src/autofinetune/llm/__init__.py
from autofinetune.llm.client import FakeLLMClient, LLMClient, LiteLLMClient

__all__ = ["LLMClient", "LiteLLMClient", "FakeLLMClient"]
```

```python
# src/autofinetune/llm/client.py
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from autofinetune.config import OrchestratorConfig
from autofinetune.errors import FatalError, RoundError


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str, schema_name: str) -> dict[str, Any]:
        ...


class FakeLLMClient:
    def __init__(self, handlers: dict[str, Callable[[str, str], dict[str, Any]]]) -> None:
        self.handlers = handlers
        self.calls: list[tuple[str, str, str]] = []

    def complete_json(self, system: str, user: str, schema_name: str) -> dict[str, Any]:
        self.calls.append((system, user, schema_name))
        if schema_name not in self.handlers:
            raise FatalError(f"FakeLLMClient missing handler for {schema_name}")
        return self.handlers[schema_name](system, user)


class LiteLLMClient:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg

    def complete_json(self, system: str, user: str, schema_name: str) -> dict[str, Any]:
        try:
            from litellm import completion
        except ImportError as e:
            raise FatalError("litellm is required for cloud orchestrator") from e

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = completion(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": system
                            + f"\nRespond with a JSON object for schema '{schema_name}'.",
                        },
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:  # noqa: BLE001 — retried network/provider errors
                last_err = e
                time.sleep(min(2**attempt, 8))
        raise RoundError(f"LLM call failed after retries: {last_err}")
```

```python
# tests/conftest.py
import pytest


@pytest.fixture
def fake_llm_factory():
    from autofinetune.llm.client import FakeLLMClient

    def _make(**handlers):
        return FakeLLMClient(handlers=handlers)

    return _make
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_llm_fake.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/llm tests/test_llm_fake.py tests/conftest.py
git commit -m "feat: add LiteLLM and FakeLLM JSON clients"
```

---

### Task 5: Model selector (pin + auto)

**Files:**
- Create: `src/autofinetune/model_select/__init__.py`
- Create: `src/autofinetune/model_select/selector.py`
- Create: `tests/test_model_select.py`

**Interfaces:**
- Consumes: `AppConfig.allowlist`, `gpu_profile`, `LLMClient`, `IngestResult`
- Produces: `select_base_model(cfg, ingest, llm, base_model_arg: str | None) -> BaseModelChoice`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_select.py
import pytest

from autofinetune.config import load_config
from autofinetune.errors import FatalError
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import FakeLLMClient
from autofinetune.model_select.selector import filter_allowlist, select_base_model
from autofinetune.schemas import DataRoute


def _ingest():
    return IngestResult(route=DataRoute.none, brief="billing domain", docs_text="", qa=[])


def test_user_pin_wins():
    cfg = load_config(None)
    llm = FakeLLMClient(handlers={})
    choice = select_base_model(
        cfg, _ingest(), llm, base_model_arg="Qwen/Qwen2.5-7B-Instruct"
    )
    assert choice.mode == "user"
    assert choice.model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_auto_uses_llm_within_allowlist():
    cfg = load_config(None)
    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "rationale": "smaller safer fit",
            }
        }
    )
    choice = select_base_model(cfg, _ingest(), llm, base_model_arg="auto")
    assert choice.mode == "auto"
    assert choice.model_id == "Qwen/Qwen2.5-3B-Instruct"


def test_auto_rejects_model_outside_filtered_allowlist():
    cfg = load_config(None)
    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "something/not-on-list",
                "rationale": "bad",
            }
        }
    )
    with pytest.raises(FatalError):
        select_base_model(cfg, _ingest(), llm, base_model_arg="auto")


def test_filter_allowlist_respects_vram():
    cfg = load_config(None)
    cfg.gpu_profile.vram_gb = 8
    filtered = filter_allowlist(cfg.allowlist, cfg.gpu_profile)
    assert all(e.min_vram_gb <= 8 for e in filtered)
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_model_select.py -v`

- [ ] **Step 3: Implement selector**

```python
# src/autofinetune/model_select/__init__.py
from autofinetune.model_select.selector import filter_allowlist, select_base_model

__all__ = ["filter_allowlist", "select_base_model"]
```

```python
# src/autofinetune/model_select/selector.py
from __future__ import annotations

import json

from autofinetune.config import AppConfig
from autofinetune.errors import FatalError
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import LLMClient
from autofinetune.schemas import AllowlistEntry, BaseModelChoice, GpuProfile


def filter_allowlist(
    allowlist: list[AllowlistEntry], gpu: GpuProfile
) -> list[AllowlistEntry]:
    return [e for e in allowlist if e.min_vram_gb <= gpu.vram_gb]


def select_base_model(
    cfg: AppConfig,
    ingest: IngestResult,
    llm: LLMClient,
    base_model_arg: str | None,
) -> BaseModelChoice:
    requested = (base_model_arg if base_model_arg is not None else cfg.base_model).strip()
    if requested != "auto":
        return BaseModelChoice(model_id=requested, mode="user", rationale="user-specified")

    candidates = filter_allowlist(cfg.allowlist, cfg.gpu_profile)
    if not candidates:
        raise FatalError(
            "No allowlist models fit the GPU profile; pin --base-model or widen allowlist/VRAM"
        )

    payload = {
        "brief": ingest.brief[:4000],
        "route": ingest.route.value,
        "qa_count": len(ingest.qa),
        "gpu": cfg.gpu_profile.model_dump(),
        "candidates": [e.model_dump() for e in candidates],
    }
    out = llm.complete_json(
        system=(
            "You recommend a base HF model for domain LoRA fine-tuning. "
            "Choose ONLY from candidates. Return JSON keys: model_id, rationale."
        ),
        user=json.dumps(payload, ensure_ascii=False),
        schema_name="recommend_model",
    )
    model_id = str(out.get("model_id", "")).strip()
    allowed = {e.id for e in candidates}
    if model_id not in allowed:
        raise FatalError(
            f"LLM recommended '{model_id}' which is not in the GPU-filtered allowlist"
        )
    return BaseModelChoice(
        model_id=model_id,
        mode="auto",
        rationale=str(out.get("rationale", "")),
    )
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_model_select.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/model_select tests/test_model_select.py
git commit -m "feat: select base model via user pin or LLM allowlist recommend"
```

---

### Task 6: Data preparation (split / synthesize)

**Files:**
- Create: `src/autofinetune/datagen/__init__.py`
- Create: `src/autofinetune/datagen/prepare.py`
- Create: `tests/test_datagen.py`

**Interfaces:**
- Consumes: `IngestResult`, `RoundPlan`, `LLMClient`, `AppConfig.data`
- Produces: `PrepareResult(train: list[QAItem], holdout: list[QAItem])`, `prepare_datasets(...)`  
  - If holdout already exists on disk for the run, reuse it (do not regenerate).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_datagen.py
from autofinetune.config import load_config
from autofinetune.datagen.prepare import prepare_datasets
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import FakeLLMClient
from autofinetune.schemas import DataRoute, LoraHyperparams, QAItem, RoundPlan


def test_full_route_splits_without_llm():
    cfg = load_config(None)
    qa = [QAItem(question=f"Q{i}", answer=f"A{i}") for i in range(100)]
    ingest = IngestResult(route=DataRoute.full, brief="b", qa=qa)
    plan = RoundPlan(data_strategy="use_user_qa", target_train_size=80)
    llm = FakeLLMClient(handlers={})
    result = prepare_datasets(cfg, ingest, plan, llm, existing_holdout=None)
    assert len(result.holdout) >= 1
    assert len(result.train) >= 1
    hold_q = {x.question for x in result.holdout}
    assert all(t.question not in hold_q for t in result.train)


def test_none_route_synthesizes_train_and_holdout():
    cfg = load_config(None)
    ingest = IngestResult(route=DataRoute.none, brief="ACME refunds", docs_text="Refunds in 14 days")
    plan = RoundPlan(data_strategy="synthesize", target_train_size=5, lora=LoraHyperparams())
    llm = FakeLLMClient(
        handlers={
            "synthesize_qa": lambda s, u: {
                "items": [
                    {"question": f"Q{i}", "answer": f"A{i}", "source": "synthetic"}
                    for i in range(8)
                ]
            }
        }
    )
    result = prepare_datasets(cfg, ingest, plan, llm, existing_holdout=None)
    assert len(result.train) >= 1
    assert len(result.holdout) >= 1
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_datagen.py -v`

- [ ] **Step 3: Implement prepare**

```python
# src/autofinetune/datagen/__init__.py
from autofinetune.datagen.prepare import PrepareResult, prepare_datasets, write_jsonl

__all__ = ["PrepareResult", "prepare_datasets", "write_jsonl"]
```

```python
# src/autofinetune/datagen/prepare.py
from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from autofinetune.config import AppConfig
from autofinetune.errors import RoundError
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import LLMClient
from autofinetune.schemas import DataRoute, QAItem, RoundPlan


class PrepareResult(BaseModel):
    train: list[QAItem] = Field(default_factory=list)
    holdout: list[QAItem] = Field(default_factory=list)


def write_jsonl(path: Path, items: list[QAItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[QAItem]:
    items: list[QAItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(QAItem.model_validate_json(line))
    return items


def _split(qa: list[QAItem], holdout_ratio: float, seed: int = 7) -> tuple[list[QAItem], list[QAItem]]:
    items = list(qa)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_hold = max(1, int(len(items) * holdout_ratio)) if len(items) > 1 else 1
    n_hold = min(n_hold, len(items) - 1) if len(items) > 1 else len(items)
    holdout = items[:n_hold]
    train = items[n_hold:] or items[:1]
    if len(items) == 1:
        # single example: duplicate risk — keep as train and holdout copy marked for eval only
        holdout = items
        train = items
    return train, holdout


def _synthesize(
    cfg: AppConfig,
    ingest: IngestResult,
    plan: RoundPlan,
    llm: LLMClient,
    n: int,
) -> list[QAItem]:
    user = json.dumps(
        {
            "brief": ingest.brief,
            "docs_text": ingest.docs_text[:12000],
            "n": n,
            "strategy": plan.data_strategy,
        },
        ensure_ascii=False,
    )
    out = llm.complete_json(
        system=(
            "Generate domain QA pairs for fine-tuning. "
            "Return JSON: {\"items\":[{\"question\":str,\"answer\":str,\"source\":\"synthetic\"}]}"
        ),
        user=user,
        schema_name="synthesize_qa",
    )
    items = [QAItem.model_validate(x) for x in out.get("items", [])]
    if not items:
        raise RoundError("DataGen returned zero synthetic QA items")
    return items


def prepare_datasets(
    cfg: AppConfig,
    ingest: IngestResult,
    plan: RoundPlan,
    llm: LLMClient,
    existing_holdout: list[QAItem] | None,
) -> PrepareResult:
    if ingest.route == DataRoute.full:
        train, holdout = _split(ingest.qa, cfg.data.holdout_ratio)
        if existing_holdout is not None:
            hold_q = {h.question for h in existing_holdout}
            train = [t for t in ingest.qa if t.question not in hold_q]
            holdout = existing_holdout
        return PrepareResult(train=train, holdout=holdout)

    if ingest.route == DataRoute.partial:
        need = max(plan.target_train_size - len(ingest.qa), 1)
        synth = _synthesize(cfg, ingest, plan, llm, need)
        combined = list(ingest.qa) + synth
        if existing_holdout is not None:
            hold_q = {h.question for h in existing_holdout}
            train = [t for t in combined if t.question not in hold_q]
            return PrepareResult(train=train, holdout=existing_holdout)
        train, holdout = _split(combined, cfg.data.holdout_ratio)
        return PrepareResult(train=train, holdout=holdout)

    # none
    total = max(plan.target_train_size, 5)
    synth = _synthesize(cfg, ingest, plan, llm, total)
    if existing_holdout is not None:
        hold_q = {h.question for h in existing_holdout}
        train = [t for t in synth if t.question not in hold_q] or synth
        return PrepareResult(train=train, holdout=existing_holdout)
    train, holdout = _split(synth, cfg.data.holdout_ratio)
    return PrepareResult(train=train, holdout=holdout)
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_datagen.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/datagen tests/test_datagen.py
git commit -m "feat: prepare train/holdout via split or LLM synthesis"
```

---

### Task 7: Trainer backends (protocol + fake + TRL)

**Files:**
- Create: `src/autofinetune/trainer/__init__.py`
- Create: `src/autofinetune/trainer/base.py`
- Create: `src/autofinetune/trainer/fake.py`
- Create: `src/autofinetune/trainer/trl_backend.py`
- Create: `tests/test_trainer_fake.py`

**Interfaces:**
- Produces: `TrainerBackend` protocol `train(base_model_id, train_jsonl, output_dir, plan) -> TrainResult`
- Produces: `get_trainer(name: str) -> TrainerBackend` (`fake` | `trl`)
- `FakeTrainer` writes a marker file under `output_dir`
- `TRLTrainerBackend` uses TRL SFTTrainer + PEFT LoRA when `autofinetune[train]` installed; raises `FatalError` with install hint if imports missing

- [ ] **Step 1: Write failing test**

```python
# tests/test_trainer_fake.py
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
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_trainer_fake.py -v`

- [ ] **Step 3: Implement trainer package**

```python
# src/autofinetune/trainer/base.py
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
```

```python
# src/autofinetune/trainer/fake.py
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
```

```python
# src/autofinetune/trainer/trl_backend.py
from __future__ import annotations

from pathlib import Path

from autofinetune.errors import FatalError, RoundError
from autofinetune.schemas import RoundPlan
from autofinetune.trainer.base import TrainResult


class TRLTrainerBackend:
    def train(
        self,
        base_model_id: str,
        train_jsonl: Path,
        output_dir: Path,
        plan: RoundPlan,
    ) -> TrainResult:
        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer
        except ImportError as e:
            raise FatalError(
                "TRL backend requires extras: pip install 'autofinetune[train]'"
            ) from e

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            ds = load_dataset("json", data_files=str(train_jsonl), split="train")

            def to_text(example):
                return {
                    "text": (
                        f"### Question:\n{example['question']}\n\n"
                        f"### Answer:\n{example['answer']}"
                    )
                }

            ds = ds.map(to_text)
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            lora = LoraConfig(
                r=plan.lora.r,
                lora_alpha=plan.lora.alpha,
                lora_dropout=plan.lora.dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            args = SFTConfig(
                output_dir=str(output_dir),
                num_train_epochs=plan.lora.epochs,
                learning_rate=plan.lora.learning_rate,
                per_device_train_batch_size=plan.lora.per_device_train_batch_size,
                gradient_accumulation_steps=plan.lora.gradient_accumulation_steps,
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                max_seq_length=2048,
            )
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=ds,
                peft_config=lora,
                processing_class=tokenizer,
            )
            trainer.train()
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            return TrainResult(output_dir=output_dir, backend="trl")
        except FatalError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RoundError(f"TRL training failed: {e}") from e
```

```python
# src/autofinetune/trainer/__init__.py
from autofinetune.trainer.base import TrainResult, TrainerBackend, get_trainer

__all__ = ["TrainResult", "TrainerBackend", "get_trainer"]
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_trainer_fake.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/trainer tests/test_trainer_fake.py
git commit -m "feat: add trainer backend protocol with fake and TRL"
```

---

### Task 8: Evaluator (aux metrics + LLM judge)

**Files:**
- Create: `src/autofinetune/eval/__init__.py`
- Create: `src/autofinetune/eval/runner.py`
- Create: `tests/test_eval.py`

**Interfaces:**
- Produces: `evaluate_holdout(llm, holdout, predict_fn) -> RoundMetrics`  
  - `predict_fn(question: str) -> str` supplied by orchestrator (fake: echo/lookup; real: generate with base+adapter)
- Judge primary: average score 0–1 from LLM JSON `{scores:[{question, score, rationale}]}`
- On judge failure: set `judge_score=None`, `judge_error=...`, still compute `aux_exact_match`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eval.py
from autofinetune.eval.runner import evaluate_holdout
from autofinetune.llm.client import FakeLLMClient
from autofinetune.schemas import QAItem


def test_judge_primary_and_aux():
    holdout = [
        QAItem(question="Q1", answer="yes"),
        QAItem(question="Q2", answer="no"),
    ]
    llm = FakeLLMClient(
        handlers={
            "judge_qa": lambda s, u: {
                "scores": [
                    {"question": "Q1", "score": 1.0, "rationale": "ok"},
                    {"question": "Q2", "score": 0.0, "rationale": "bad"},
                ]
            }
        }
    )

    def predict(q: str) -> str:
        return "yes" if q == "Q1" else "maybe"

    metrics = evaluate_holdout(llm, holdout, predict)
    assert metrics.n_eval == 2
    assert metrics.judge_score == 0.5
    assert metrics.aux_exact_match == 0.5


def test_judge_failure_keeps_aux():
    holdout = [QAItem(question="Q1", answer="yes")]
    llm = FakeLLMClient(
        handlers={"judge_qa": lambda s, u: (_ for _ in ()).throw(RuntimeError("down"))}
    )

    def predict(q: str) -> str:
        return "yes"

    # FakeLLMClient raises FatalError on handler exception only if handler raises —
    # implement evaluate_holdout to catch RoundError/Exception from complete_json.
    from autofinetune.errors import RoundError
    from autofinetune.llm.client import FakeLLMClient as F

    class Boom(F):
        def complete_json(self, system, user, schema_name):
            raise RoundError("judge down")

    metrics = evaluate_holdout(Boom(handlers={}), holdout, predict)
    assert metrics.judge_score is None
    assert metrics.judge_error
    assert metrics.aux_exact_match == 1.0
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_eval.py -v`

- [ ] **Step 3: Implement evaluator**

```python
# src/autofinetune/eval/__init__.py
from autofinetune.eval.runner import evaluate_holdout

__all__ = ["evaluate_holdout"]
```

```python
# src/autofinetune/eval/runner.py
from __future__ import annotations

import json
from collections.abc import Callable

from autofinetune.llm.client import LLMClient
from autofinetune.schemas import QAItem, RoundMetrics


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def evaluate_holdout(
    llm: LLMClient,
    holdout: list[QAItem],
    predict_fn: Callable[[str], str],
) -> RoundMetrics:
    preds: list[dict[str, str]] = []
    exact = 0
    for item in holdout:
        pred = predict_fn(item.question)
        preds.append(
            {
                "question": item.question,
                "gold": item.answer,
                "prediction": pred,
            }
        )
        if _norm(pred) == _norm(item.answer):
            exact += 1
    n = len(holdout)
    aux = (exact / n) if n else 0.0

    try:
        out = llm.complete_json(
            system=(
                "You are grading domain QA predictions. "
                "Return JSON {\"scores\":[{\"question\":str,\"score\":0..1,\"rationale\":str}]}. "
                "score 1 = fully correct, 0 = wrong."
            ),
            user=json.dumps({"items": preds}, ensure_ascii=False),
            schema_name="judge_qa",
        )
        scores = [float(x["score"]) for x in out.get("scores", [])]
        judge = (sum(scores) / len(scores)) if scores else None
        return RoundMetrics(
            judge_score=judge,
            aux_exact_match=aux,
            n_eval=n,
            extras={"predictions": preds},
        )
    except Exception as e:  # noqa: BLE001 — judge soft-fail per spec
        return RoundMetrics(
            judge_score=None,
            aux_exact_match=aux,
            n_eval=n,
            judge_error=str(e),
            extras={"predictions": preds},
        )
```

- [ ] **Step 4: Fix second test if needed and PASS**

Run: `pytest tests/test_eval.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/eval tests/test_eval.py
git commit -m "feat: evaluate holdout with LLM judge primary and aux match"
```

---

### Task 9: Orchestrator loop (plan → train → eval → decide → budgets → pause)

**Files:**
- Create: `src/autofinetune/orchestrator/__init__.py`
- Create: `src/autofinetune/orchestrator/loop.py`
- Create: `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `run_experiment(cfg, store, run_id, llm, trainer, base_model_arg=None, resume_note=None) -> RunRecord`
- Startup: ingest from `runs/<id>/input` → select base if missing → loop rounds
- Honor `pause_requested` after finishing a round (set status `paused`)
- Stop on decide.stop, max_rounds, wall clock, optional cost
- Update `best.json` when judge_score improves (if judge None, fall back to aux)

- [ ] **Step 1: Write failing integration-style unit test**

```python
# tests/test_orchestrator.py
from pathlib import Path

from autofinetune.config import load_config
from autofinetune.llm.client import FakeLLMClient
from autofinetune.orchestrator.loop import run_experiment
from autofinetune.schemas import RunStatus
from autofinetune.store.run_store import RunStore
from autofinetune.trainer.fake import FakeTrainer


def test_one_round_stop_with_fakes(tmp_path: Path):
    cfg = load_config(None)
    cfg.trainer.backend = "fake"
    cfg.budgets.max_rounds = 2
    cfg.runs_dir = tmp_path / "runs"
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("ACME billing domain knowledge", encoding="utf-8")

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=inp)

    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "default",
            },
            "round_plan": lambda s, u: {
                "data_strategy": "synthesize",
                "target_train_size": 6,
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "epochs": 1,
                    "learning_rate": 0.0002,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                },
                "eval_focus": "facts",
                "notes": "",
            },
            "synthesize_qa": lambda s, u: {
                "items": [
                    {"question": f"Q{i}", "answer": f"A{i}", "source": "synthetic"}
                    for i in range(8)
                ]
            },
            "judge_qa": lambda s, u: {
                "scores": [{"question": f"Q{i}", "score": 0.8, "rationale": "ok"} for i in range(8)]
            },
            "decide": lambda s, u: {
                "action": "stop",
                "hypothesis": "",
                "reason": "good enough",
            },
        }
    )

    def predict(q: str) -> str:
        return q.replace("Q", "A") if q.startswith("Q") else "A0"

    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        FakeTrainer(),
        base_model_arg="auto",
        predict_fn_factory=lambda **kwargs: predict,
    )
    assert final.status == RunStatus.completed
    assert final.base_model is not None
    assert final.current_round >= 1
    assert (store.best_path(rec.run_id)).is_file()
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_orchestrator.py -v`

- [ ] **Step 3: Implement orchestrator**

Implement `src/autofinetune/orchestrator/loop.py` with approximately this structure (keep functions small):

```python
# src/autofinetune/orchestrator/loop.py
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autofinetune.config import AppConfig
from autofinetune.datagen.prepare import prepare_datasets, read_jsonl, write_jsonl
from autofinetune.errors import FatalError, RoundError
from autofinetune.eval.runner import evaluate_holdout
from autofinetune.ingest.bundle import ingest_bundle
from autofinetune.llm.client import LLMClient
from autofinetune.model_select.selector import select_base_model
from autofinetune.schemas import DecideResult, RoundPlan, RunStatus
from autofinetune.store.run_store import RunRecord, RunStore
from autofinetune.trainer.base import TrainerBackend


PredictFactory = Callable[..., Callable[[str], str]]


def _default_predict_factory(**kwargs: Any) -> Callable[[str], str]:
    # Fake/default: map from train file if present; else empty
    train_path: Path | None = kwargs.get("train_jsonl")
    lookup: dict[str, str] = {}
    if train_path and train_path.is_file():
        for item in read_jsonl(train_path):
            lookup[item.question] = item.answer

    def predict(q: str) -> str:
        return lookup.get(q, "")

    return predict


def run_experiment(
    cfg: AppConfig,
    store: RunStore,
    run_id: str,
    llm: LLMClient,
    trainer: TrainerBackend,
    base_model_arg: str | None = None,
    resume_note: str | None = None,
    predict_fn_factory: PredictFactory | None = None,
) -> RunRecord:
    predict_fn_factory = predict_fn_factory or _default_predict_factory
    rec = store.load(run_id)
    if resume_note:
        rec.user_note = resume_note
        store.save(rec)

    input_dir = store.run_dir(run_id) / "input"
    ingest = ingest_bundle(input_dir, cfg)
    store.set_route(run_id, ingest.route.value)

    if rec.base_model is None:
        choice = select_base_model(cfg, ingest, llm, base_model_arg)
        store.set_base_model(run_id, choice)
        rec = store.load(run_id)
    # resume: never re-select

    if rec.started_at is None:
        rec.started_at = rec.created_at
        store.save(rec)

    store.set_status(run_id, RunStatus.running)
    started = time.time()
    start_round = rec.current_round + 1

    for round_idx in range(start_round, cfg.budgets.max_rounds + 1):
        rec = store.load(run_id)
        if cfg.budgets.max_wall_time_sec and (time.time() - started) > cfg.budgets.max_wall_time_sec:
            store.set_status(run_id, RunStatus.completed)
            break
        if (
            cfg.budgets.max_llm_cost_usd is not None
            and rec.llm_cost_usd_est >= cfg.budgets.max_llm_cost_usd
        ):
            store.set_status(run_id, RunStatus.completed)
            break

        try:
            plan = _plan_round(llm, cfg, ingest, rec, round_idx)
            store.save_plan(run_id, round_idx, plan)

            existing_holdout = None
            hp = store.holdout_path(run_id)
            if hp.is_file():
                existing_holdout = read_jsonl(hp)

            prepared = prepare_datasets(cfg, ingest, plan, llm, existing_holdout)
            if not hp.is_file():
                write_jsonl(hp, prepared.holdout)
            train_path = store.round_dir(run_id, round_idx) / "train.jsonl"
            write_jsonl(train_path, prepared.train)

            adapter_dir = store.adapter_dir(run_id, round_idx)
            trainer.train(rec.base_model.model_id, train_path, adapter_dir, plan)

            predict = predict_fn_factory(
                base_model_id=rec.base_model.model_id,
                adapter_dir=adapter_dir,
                train_jsonl=train_path,
            )
            holdout_items = read_jsonl(store.holdout_path(run_id))
            metrics = evaluate_holdout(llm, holdout_items, predict)
            store.save_metrics(run_id, round_idx, metrics)

            report = (
                f"# Round {round_idx}\n\n"
                f"Base: `{rec.base_model.model_id}`\n\n"
                f"Judge: {metrics.judge_score}\n\n"
                f"Aux EM: {metrics.aux_exact_match}\n\n"
                f"Plan: {plan.data_strategy}\n"
            )
            store.save_report(run_id, round_idx, report)

            _maybe_update_best(store, run_id, round_idx, metrics)

            decide = _decide(llm, rec, round_idx, metrics, cfg)
            store.save_decide(run_id, round_idx, decide)

            rec = store.load(run_id)
            rec.current_round = round_idx
            rec.last_error = None
            store.save(rec)

            if store.load(run_id).pause_requested:
                store.set_status(run_id, RunStatus.paused)
                store.clear_pause(run_id)
                return store.load(run_id)

            if decide.action == "stop":
                store.set_status(run_id, RunStatus.completed)
                return store.load(run_id)

        except RoundError as e:
            rec = store.load(run_id)
            rec.current_round = round_idx
            rec.last_error = str(e)
            store.save(rec)
            store.save_report(run_id, round_idx, f"# Round {round_idx} FAILED\n\n{e}\n")
            continue
        except FatalError:
            store.set_status(run_id, RunStatus.failed)
            raise

    store.set_status(run_id, RunStatus.completed)
    return store.load(run_id)


def _plan_round(llm, cfg, ingest, rec, round_idx: int) -> RoundPlan:
    user = {
        "round": round_idx,
        "base_model": rec.base_model.model_dump() if rec.base_model else None,
        "route": ingest.route.value,
        "brief": ingest.brief[:3000],
        "user_note": rec.user_note,
        "last_error": rec.last_error,
        "defaults": {
            "target_train_size": cfg.data.target_train_size,
            "lora_r": cfg.trainer.default_lora_r,
        },
    }
    out = llm.complete_json(
        system=(
            "Plan one fine-tuning round for domain knowledge. "
            "Do not change base model. Return JSON matching RoundPlan fields: "
            "data_strategy, target_train_size, lora{...}, eval_focus, notes."
        ),
        user=json.dumps(user, ensure_ascii=False),
        schema_name="round_plan",
    )
    return RoundPlan.model_validate(out)


def _decide(llm, rec, round_idx, metrics, cfg) -> DecideResult:
    out = llm.complete_json(
        system="Decide whether to continue fine-tuning rounds. Return JSON: action(continue|stop), hypothesis, reason.",
        user=json.dumps(
            {
                "round": round_idx,
                "metrics": metrics.model_dump(),
                "max_rounds": cfg.budgets.max_rounds,
                "best_round": rec.best_round,
            },
            ensure_ascii=False,
        ),
        schema_name="decide",
    )
    return DecideResult.model_validate(out)


def _maybe_update_best(store, run_id, round_idx, metrics) -> None:
    rec = store.load(run_id)
    score = metrics.judge_score
    if score is None:
        score = metrics.aux_exact_match or 0.0
    if rec.best_round is None:
        store.update_best(run_id, round_idx, metrics)
        return
    best = json.loads(store.best_path(run_id).read_text(encoding="utf-8"))
    prev = best.get("metrics", {})
    prev_score = prev.get("judge_score")
    if prev_score is None:
        prev_score = prev.get("aux_exact_match") or 0.0
    if score >= prev_score:
        store.update_best(run_id, round_idx, metrics)
```

```python
# src/autofinetune/orchestrator/__init__.py
from autofinetune.orchestrator.loop import run_experiment

__all__ = ["run_experiment"]
```

**Note for implementer:** Ensure `judge_qa` fake handler scores length is dynamic — adjust Fake handler to:

```python
"judge_qa": lambda s, u: {
    "scores": [
        {"question": it["question"], "score": 0.8, "rationale": "ok"}
        for it in __import__("json").loads(u)["items"]
    ]
},
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_orchestrator.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/orchestrator tests/test_orchestrator.py
git commit -m "feat: implement closed-loop orchestrator with budgets and pause"
```

---

### Task 10: CLI + end-to-end fake integration + README

**Files:**
- Create: `src/autofinetune/cli.py`
- Create: `tests/test_cli_integration.py`
- Create: `README.md`

**Interfaces:**
- CLI commands: `run`, `pause`, `resume`, `status`, `report`
- `run` flags: `--config`, `--base-model`, `--runs-dir`, `--trainer` (override backend)

- [ ] **Step 1: Write failing CLI test**

```python
# tests/test_cli_integration.py
from pathlib import Path

from typer.testing import CliRunner

from autofinetune.cli import app


def test_cli_run_with_fake_trainer(tmp_path: Path, monkeypatch):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("Domain: warehouse robots safety rules", encoding="utf-8")
    runs = tmp_path / "runs"
    # Force fake LLM via env understood by CLI
    monkeypatch.setenv("AUTOFINETUNE_LLM", "fake")
    monkeypatch.setenv("AUTOFINETUNE_TRAINER", "fake")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", str(inp), "--runs-dir", str(runs), "--base-model", "auto"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert any(runs.iterdir())
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_cli_integration.py -v`

- [ ] **Step 3: Implement CLI + README**

```python
# src/autofinetune/cli.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.pretty import pprint

from autofinetune.config import load_config
from autofinetune.llm.client import FakeLLMClient, LiteLLMClient
from autofinetune.orchestrator.loop import run_experiment
from autofinetune.store.run_store import RunStore
from autofinetune.trainer.base import get_trainer

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _build_fake_llm() -> FakeLLMClient:
    def judge(s, u):
        items = json.loads(u).get("items", [])
        return {
            "scores": [
                {"question": it["question"], "score": 0.7, "rationale": "fake"}
                for it in items
            ]
        }

    return FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "fake default",
            },
            "round_plan": lambda s, u: {
                "data_strategy": "synthesize",
                "target_train_size": 6,
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "epochs": 1,
                    "learning_rate": 0.0002,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                },
                "eval_focus": "facts",
                "notes": "",
            },
            "synthesize_qa": lambda s, u: {
                "items": [
                    {
                        "question": f"Q{i}",
                        "answer": f"A{i}",
                        "source": "synthetic",
                    }
                    for i in range(8)
                ]
            },
            "judge_qa": judge,
            "decide": lambda s, u: {
                "action": "stop",
                "hypothesis": "",
                "reason": "fake stop",
            },
        }
    )


def _llm_from_env(cfg):
    if os.getenv("AUTOFINETUNE_LLM", "").lower() == "fake":
        return _build_fake_llm()
    return LiteLLMClient(cfg.orchestrator)


@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
    base_model: Optional[str] = typer.Option(None, "--base-model"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
    trainer: Optional[str] = typer.Option(None, "--trainer"),
) -> None:
    cfg = load_config(config)
    if runs_dir:
        cfg.runs_dir = runs_dir
    if trainer:
        cfg.trainer.backend = trainer
    if os.getenv("AUTOFINETUNE_TRAINER"):
        cfg.trainer.backend = os.environ["AUTOFINETUNE_TRAINER"]

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=input_dir)
    console.print(f"Created run [bold]{rec.run_id}[/bold]")
    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        _llm_from_env(cfg),
        get_trainer(cfg.trainer.backend),
        base_model_arg=base_model,
    )
    console.print(f"Status: {final.status.value}")
    if final.base_model:
        console.print(f"Base model: {final.base_model.model_id} ({final.base_model.mode})")


@app.command()
def pause(run_id: str, runs_dir: Path = typer.Option(Path("runs"), "--runs-dir")) -> None:
    RunStore(runs_dir).request_pause(run_id)
    console.print(f"Pause requested for {run_id}")


@app.command()
def resume(
    run_id: str,
    note: Optional[str] = typer.Option(None, "--note"),
    config: Optional[Path] = typer.Option(None, "--config"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    cfg = load_config(config)
    cfg.runs_dir = runs_dir
    if os.getenv("AUTOFINETUNE_TRAINER"):
        cfg.trainer.backend = os.environ["AUTOFINETUNE_TRAINER"]
    store = RunStore(runs_dir)
    final = run_experiment(
        cfg,
        store,
        run_id,
        _llm_from_env(cfg),
        get_trainer(cfg.trainer.backend),
        resume_note=note,
    )
    console.print(f"Status: {final.status.value}")


@app.command()
def status(run_id: str, runs_dir: Path = typer.Option(Path("runs"), "--runs-dir")) -> None:
    rec = RunStore(runs_dir).load(run_id)
    pprint(rec.model_dump())


@app.command()
def report(run_id: str, runs_dir: Path = typer.Option(Path("runs"), "--runs-dir")) -> None:
    store = RunStore(runs_dir)
    rec = store.load(run_id)
    for i in range(1, rec.current_round + 1):
        path = store.round_dir(run_id, i) / "report.md"
        console.rule(f"Round {i}")
        if path.is_file():
            console.print(path.read_text(encoding="utf-8"))
```

```markdown
# AutoFineTune

Closed-loop **domain-knowledge** fine-tuning agent (CLI). Cloud LLM plans rounds; local LoRA trains; LLM-as-judge evaluates.

## Quick start (fake / CI)

```bash
pip install -e ".[dev]"
export AUTOFINETUNE_LLM=fake
export AUTOFINETUNE_TRAINER=fake
autofinetune run ./tests/fixtures/minimal_input --runs-dir ./runs --base-model auto
```

## Real local training

```bash
pip install -e ".[train]"
export OPENAI_API_KEY=...   # or other LiteLLM provider env
autofinetune run ./my_input --base-model Qwen/Qwen2.5-7B-Instruct --trainer trl
# or let the orchestrator recommend:
autofinetune run ./my_input --base-model auto
```

### Input layout

```text
input/
  brief.md      # optional but recommended
  docs/         # optional md/txt/pdf
  qa.jsonl      # optional {"question","answer"}
```

### Commands

- `autofinetune run`
- `autofinetune pause <run_id>`
- `autofinetune resume <run_id> --note "..."`
- `autofinetune status <run_id>`
- `autofinetune report <run_id>`

See `docs/superpowers/specs/2026-08-12-autofinetune-design.md`.
```

- [ ] **Step 4: Run full test suite**

Run: `pytest -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/cli.py tests/test_cli_integration.py README.md
git commit -m "feat: add CLI and README for fake and local training flows"
```

---

## Spec coverage checklist (self-review)

| Spec requirement | Task |
|------------------|------|
| Closed loop plan→data→train→eval→decide | 9 |
| Domain knowledge only | Global + datagen prompts |
| none/partial/full ingest | 3, 6 |
| Pause/resume + notes | 2, 9, 10 |
| CLI first | 10 |
| Trainer abstraction; local TRL; fake for CI | 7 |
| Base model pin or auto allowlist | 5 |
| LiteLLM orchestrator | 4, 10 |
| LLM-as-judge primary + aux | 8 |
| Budgets max_rounds/wall/cost | 9 |
| Holdout frozen / not trained on | 6, 9 |
| runs/ artifact layout | 2 |
| Success criteria 1–6 | 3,5,9,10 |

## Placeholder / consistency notes fixed in plan

- Single `evaluate_holdout` call on frozen holdout in the orchestrator round body.
- `RoundPlan` / `DecideResult` / `BaseModelChoice` names consistent across tasks.
- `get_trainer("fake"|"trl")` matches config `trainer.backend`.
- CLI env `AUTOFINETUNE_LLM=fake` / `AUTOFINETUNE_TRAINER=fake` for CI without GPU.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-12-autofinetune.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration  

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints  

Which approach?
