# Auto Trainer Backend (TRL + MLX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default `trainer.backend: auto` resolves to CUDA TRL or Apple Silicon MLX; add MLX train/predict, 1.5B allowlist entry, and Mac-friendly GPU defaults.

**Architecture:** Add `resolve_trainer_backend()` that maps `auto|trl|mlx|fake` → concrete backend. CLI resolves once before `get_trainer` / `get_predict_factory` and persists the concrete name. New `MLXTrainerBackend` + `mlx_predict_factory` use mlx-lm LoRA (completions dataset + adapter-path). Darwin without user `gpu_profile` defaults to 16GB unified-memory profile; recommend prompt biases ≤3B when mlx or `vram_gb<=16`.

**Tech Stack:** Python 3.11+, pytest, torch (optional), mlx + mlx-lm[train] (optional), existing Typer CLI

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-13-auto-trainer-mlx-design.md`
- Downstream never sees `auto` — only `trl` / `mlx` / `fake`
- Resume with stored backend vs re-resolved `auto` mismatch → `FatalError` (v1)
- MLX adapters ≠ PEFT; no conversion in v1
- Do not put `mlx` into default `[dev]` extras (CI stays fake/`[train]` optional)
- Prefer in-process mlx-lm calls; subprocess CLI only if no stable importable train API
- Commit author `mike <mliu36292@gmail.com>`; never leave `Co-authored-by: Cursor`
- CI must pass offline with `AUTOFINETUNE_TRAINER=fake` without installing mlx

## File Structure

| File | Responsibility |
|------|----------------|
| `src/autofinetune/trainer/resolve.py` | `resolve_trainer_backend`, cuda/mlx probes |
| `src/autofinetune/trainer/base.py` | `get_trainer` accepts `trl`/`mlx`/`fake` (concrete only) |
| `src/autofinetune/trainer/mlx_backend.py` | MLX LoRA train |
| `src/autofinetune/eval/predict.py` | `mlx_predict_factory` |
| `src/autofinetune/cli.py` | Resolve auto; resume mismatch; Darwin profile hook |
| `src/autofinetune/config.py` + `defaults/config.yaml` | Default `backend: auto`; Darwin gpu default helper |
| `src/autofinetune/model_select/selector.py` | Small-model recommend bias |
| `src/autofinetune/defaults/allowlist.yaml` | Add Qwen2.5-1.5B-Instruct |
| `pyproject.toml` | `[project.optional-dependencies].mlx` |
| `tests/test_trainer_resolve.py` | Resolution matrix |
| `tests/test_allowlist_filter.py` | 16GB filter + 1.5B present |
| `tests/test_mlx_backend.py` | Skip-unless-mlx smoke / mocked train |
| `tests/test_predict_factory.py` | mlx factory registration |
| `README.md` | auto / Mac / CUDA docs |

---

### Task 1: `resolve_trainer_backend` + `get_trainer("mlx")` stub wiring

**Files:**
- Create: `src/autofinetune/trainer/resolve.py`
- Modify: `src/autofinetune/trainer/base.py`
- Modify: `src/autofinetune/trainer/__init__.py`
- Create: `tests/test_trainer_resolve.py`

**Interfaces:**
- Produces:
  - `cuda_available() -> bool`
  - `mlx_available() -> bool`
  - `resolve_trainer_backend(name: str) -> str`  # returns concrete `fake|trl|mlx`
  - Raises `FatalError` for unknown names or `auto` with neither backend
- `get_trainer("mlx")` imports `MLXTrainerBackend` (Task 4 will implement; for Task 1 either lazy-import a thin stub class in `mlx_backend.py` that raises `FatalError("not implemented")` **only if called**, OR defer `get_trainer("mlx")` to Task 4 and only export resolver here)
- **Decision for this plan:** Task 1 creates `mlx_backend.py` with `MLXTrainerBackend` whose `train` raises `FatalError("MLXTrainerBackend not implemented")` so imports work; Task 4 replaces `train` body. Prefer: Task 1 only adds resolver + updates `get_trainer` to accept mlx importing the real module once Task 4 exists — **simpler:** Task 1 resolver only; `get_trainer` still trl/fake; Task 4 adds mlx to `get_trainer`.

**Revised Task 1 scope:** resolver + tests only. `get_trainer` unchanged until Task 4.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_trainer_resolve.py
import pytest

from autofinetune.errors import FatalError
from autofinetune.trainer.resolve import resolve_trainer_backend


def test_fake_passthrough():
    assert resolve_trainer_backend("fake") == "fake"
    assert resolve_trainer_backend("FAKE") == "fake"


def test_trl_forced():
    assert resolve_trainer_backend("trl") == "trl"


def test_mlx_forced():
    assert resolve_trainer_backend("mlx") == "mlx"


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.cuda_available", lambda: True
    )
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.mlx_available", lambda: True
    )
    assert resolve_trainer_backend("auto") == "trl"


def test_auto_falls_back_to_mlx(monkeypatch):
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.cuda_available", lambda: False
    )
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.mlx_available", lambda: True
    )
    assert resolve_trainer_backend("auto") == "mlx"


def test_auto_neither_raises(monkeypatch):
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.cuda_available", lambda: False
    )
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.mlx_available", lambda: False
    )
    with pytest.raises(FatalError, match="auto"):
        resolve_trainer_backend("auto")


def test_unknown_raises():
    with pytest.raises(FatalError, match="Unknown"):
        resolve_trainer_backend("bogus")
```

- [ ] **Step 2: Run tests — expect FAIL (import)**

Run: `pytest tests/test_trainer_resolve.py -v`

- [ ] **Step 3: Implement `resolve.py`**

```python
from __future__ import annotations

import platform

from autofinetune.errors import FatalError


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def mlx_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_trainer_backend(name: str) -> str:
    key = (name or "").strip().lower()
    if key == "fake":
        return "fake"
    if key == "trl":
        return "trl"
    if key == "mlx":
        return "mlx"
    if key == "auto":
        if cuda_available():
            return "trl"
        if mlx_available():
            return "mlx"
        raise FatalError(
            "trainer.backend=auto found neither CUDA (pip install 'autofinetune[train]') "
            "nor Apple Silicon MLX (pip install 'autofinetune[mlx]'). "
            "Set --trainer fake|trl|mlx explicitly."
        )
    raise FatalError(f"Unknown trainer backend: {name}")
```

Export from `trainer/__init__.py`:

```python
from autofinetune.trainer.base import TrainResult, TrainerBackend, get_trainer
from autofinetune.trainer.resolve import resolve_trainer_backend

__all__ = ["TrainResult", "TrainerBackend", "get_trainer", "resolve_trainer_backend"]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_trainer_resolve.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/autofinetune/trainer/resolve.py src/autofinetune/trainer/__init__.py tests/test_trainer_resolve.py
git commit -m "feat: resolve auto trainer backend to trl or mlx"
```

---

### Task 2: Config default `auto` + Darwin GPU profile + CLI resolve/resume

**Files:**
- Modify: `src/autofinetune/config.py`
- Modify: `src/autofinetune/defaults/config.yaml`
- Modify: `src/autofinetune/cli.py`
- Modify: `tests/test_config.py`
- Create or modify: `tests/test_cli_trainer_resolve.py` (small unit tests; avoid full run)

**Interfaces:**
- `TrainerConfig.backend` default `"auto"`
- `load_config(path)`: if `platform.system()=="Darwin"` and user yaml did **not** include `gpu_profile`, set `cfg.gpu_profile` to `GpuProfile(name="apple-unified-16gb", vram_gb=16)` (preserve other GpuProfile fields if any)
- CLI `run`/`resume`: after override logic, `cfg.trainer.backend = resolve_trainer_backend(cfg.trainer.backend)`; then persist; on resume if overridden to `auto` and resolved ≠ stored → FatalError

- [ ] **Step 1: Failing config test**

```python
# extend tests/test_config.py
def test_trainer_backend_defaults_to_auto():
    cfg = load_config(None)
    assert cfg.trainer.backend == "auto"
```

Add Darwin profile test with monkeypatch:

```python
import platform
from autofinetune.config import load_config

def test_darwin_default_gpu_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    cfg = load_config(None)
    assert cfg.gpu_profile.name == "apple-unified-16gb"
    assert cfg.gpu_profile.vram_gb == 16


def test_user_gpu_profile_not_overridden_on_darwin(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    p = tmp_path / "c.yaml"
    p.write_text("gpu_profile:\n  name: custom\n  vram_gb: 24\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.gpu_profile.name == "custom"
    assert cfg.gpu_profile.vram_gb == 24
```

CLI resolve unit test (pure helper preferred): extract `_finalize_trainer_backend(cfg, *, stored: str | None, overridden: bool) -> str` in cli or resolve module to keep Typer out of unit tests.

```python
# in resolve.py or cli helper
def finalize_trainer_backend(
    requested: str,
    *,
    stored: str | None,
    overridden: bool,
) -> str:
    """Return concrete backend; FatalError if resume auto re-resolve mismatches stored."""
    if not overridden and stored:
        return stored.strip().lower()
    resolved = resolve_trainer_backend(requested)
    if overridden and stored and requested.strip().lower() == "auto":
        if resolved != stored.strip().lower():
            raise FatalError(
                f"Resolved trainer backend {resolved!r} != stored run backend {stored!r}"
            )
    return resolved
```

Actually on resume without override we use stored (already concrete). On resume with `--trainer auto`, we re-resolve and must match stored. On run, no stored yet — just resolve.

```python
def finalize_trainer_backend(
    requested: str,
    *,
    stored: str | None = None,
    overridden: bool = False,
    is_resume: bool = False,
) -> str:
    if is_resume and not overridden and stored:
        return stored.strip().lower()
    resolved = resolve_trainer_backend(requested)
    if is_resume and overridden and stored:
        if resolved != stored.strip().lower():
            raise FatalError(
                f"trainer backend mismatch: resolved {resolved!r} vs stored {stored!r}"
            )
    return resolved
```

Tests for finalize in `test_trainer_resolve.py`.

Wire CLI:

```python
from autofinetune.trainer.resolve import finalize_trainer_backend

# run:
_apply_trainer_override(cfg, trainer)
cfg.trainer.backend = finalize_trainer_backend(cfg.trainer.backend)
store.set_trainer_backend(...)
get_trainer(cfg.trainer.backend)

# resume:
overridden = _apply_trainer_override(...)
if not overridden and rec.trainer_backend:
    cfg.trainer.backend = rec.trainer_backend
cfg.trainer.backend = finalize_trainer_backend(
    cfg.trainer.backend,
    stored=rec.trainer_backend,
    overridden=overridden,
    is_resume=True,
)
if overridden:
    store.set_trainer_backend(run_id, cfg.trainer.backend)
```

Note: when not overridden on resume, first assignment already sets concrete stored; `finalize` with `is_resume and not overridden` returns stored again — OK.

When `path is None` on Linux, keep yaml `single-24gb` / 24 — Darwin-only rewrite.

Implement Darwin apply inside `load_config` after validate:

```python
def load_config(path: Path | None = None) -> AppConfig:
    raw = _default_yaml("config.yaml")
    ...
    user = ...
    user_has_gpu = isinstance(user, dict) and "gpu_profile" in user if path else False
    ...
    cfg = AppConfig.model_validate(raw)
    ...
    if platform.system() == "Darwin" and not user_has_gpu:
        cfg.gpu_profile = GpuProfile(name="apple-unified-16gb", vram_gb=16)
    return cfg
```

When `path is None`, `user_has_gpu=False` → Darwin gets 16GB. Good.

Update defaults yaml:

```yaml
trainer:
  backend: auto
```

Update `test_load_config_defaults...` assertion from `trl` to `auto`.

- [ ] **Step 2–4: TDD implement, pass tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: default trainer auto with Darwin GPU profile and CLI resolve"
```

---

### Task 3: Allowlist 1.5B + recommend bias

**Files:**
- Modify: `src/autofinetune/defaults/allowlist.yaml`
- Modify: `src/autofinetune/model_select/selector.py`
- Create: `tests/test_allowlist_filter.py`
- Modify: `src/autofinetune/cli.py` fake recommend to prefer 1.5B id (CI fake still works on Darwin 16GB)

**Allowlist entry:**

```yaml
  - id: Qwen/Qwen2.5-1.5B-Instruct
    approx_params_b: 1.5
    min_vram_gb: 6
    chat_template: qwen
    notes: "Apple Silicon / low-VRAM friendly"
```

Place **before** 7B so listing order prefers smaller (optional). Keep 3B/7B/8B.

**Recommend bias:** `select_base_model` accepts optional `trainer_backend: str | None = None`. When `trainer_backend == "mlx"` or `cfg.gpu_profile.vram_gb <= 16`, append to system prompt:

```text
Prefer the smallest instruct model that fits unless the brief clearly needs larger capacity.
```

Also pass `trainer_backend` in user JSON payload for the LLM.

Update `run_experiment` / call site to pass `cfg.trainer.backend` (already concrete).

Fake CLI handler:

```python
"recommend_model": lambda s, u: {
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "rationale": "fake small default",
},
```

Tests:

```python
def test_filter_16gb_includes_small_excludes_8b():
    from autofinetune.config import load_config
    from autofinetune.model_select.selector import filter_allowlist
    from autofinetune.schemas import GpuProfile
    cfg = load_config(None)
    # force allowlist from defaults even on Darwin
    kept = filter_allowlist(cfg.allowlist, GpuProfile(name="t", vram_gb=16))
    ids = {e.id for e in kept}
    assert "Qwen/Qwen2.5-1.5B-Instruct" in ids
    assert "Qwen/Qwen2.5-3B-Instruct" in ids
    assert "meta-llama/Llama-3.1-8B-Instruct" not in ids  # min 18
```

- [ ] Commit: `feat: add 1.5B allowlist and small-model recommend bias`

---

### Task 4: `MLXTrainerBackend` + pyproject `[mlx]`

**Files:**
- Create: `src/autofinetune/trainer/mlx_backend.py`
- Modify: `src/autofinetune/trainer/base.py` (`get_trainer("mlx")`)
- Modify: `pyproject.toml`
- Create: `tests/test_mlx_backend.py`

**Data prep:** From AutoFineTune `train.jsonl` (`question`/`answer`), write a temp dir:

```
<data_dir>/train.jsonl
```

Each line mlx **completions** format (stable for instruct SFT):

```json
{"prompt": "### Question:\n{q}\n\n### Answer:\n", "completion": "{a}"}
```

(Keeps the same surface text as TRL’s concatenated template.)

**Training:** Prefer in-process. At implement time, inspect installed `mlx_lm`:

1. Try importing trainer helpers (e.g. modules under `mlx_lm.tuner` / documented train API).
2. If only CLI is stable, run:

```python
import subprocess, sys
cmd = [
    sys.executable, "-m", "mlx_lm", "lora",
    "--model", base_model_id,
    "--train",
    "--data", str(data_dir),
    "--adapter-path", str(output_dir),
    "--batch-size", str(plan.lora.per_device_train_batch_size),
    "--lora-rank", str(plan.lora.r),  # confirm flag name via --help at implement time
    "--learning-rate", str(plan.lora.learning_rate),
    "--iters", str(max(plan.lora.epochs * max(len(rows), 1), 1)),  # map epochs→iters conservatively
]
```

**Required behavior regardless of invoke style:**
- Missing `mlx` / `mlx_lm` → `FatalError` mentioning `pip install 'autofinetune[mlx]'`
- Other failures → `RoundError`
- `TrainResult(output_dir=output_dir, backend="mlx")`
- Log once and ignore unsupported knobs (e.g. dropout if unsupported)

**pyproject:**

```toml
mlx = [
  "mlx",
  "mlx-lm[train]",
]
```

**Tests:**
- Without mlx: instantiate backend and assert `train` raises `FatalError` matching `autofinetune\[mlx\]` (mock ImportError path) OR skip body.
- With mlx installed (optional): `@pytest.mark.skipif(not mlx_available(), reason="mlx not installed")` smoke on tiny synthetic jsonl — may be too heavy for CI; **default CI:** mock the train entrypoint and assert data file written + adapter-path arg.

```python
def test_mlx_backend_writes_completions_and_calls_train(tmp_path, monkeypatch):
    # mock heavy train function used by backend
    ...
```

- [ ] Commit: `feat: add MLX LoRA trainer backend`

---

### Task 5: `mlx_predict_factory`

**Files:**
- Modify: `src/autofinetune/eval/predict.py`
- Modify: `tests/test_predict_factory.py` (create if missing; extend)

**Implementation sketch:**

```python
def mlx_predict_factory(**kwargs):
    base_model_id = kwargs["base_model_id"]
    adapter_dir = Path(kwargs["adapter_dir"])
    try:
        from mlx_lm import load, generate
    except ImportError as e:
        raise FatalError(
            "MLX predict requires extras: pip install 'autofinetune[mlx]'"
        ) from e
    model, tokenizer = load(base_model_id, adapter_path=str(adapter_dir))

    def predict(q: str) -> str:
        prompt = f"### Question:\n{q}\n\n### Answer:\n"
        # use generate API; greedy / low temp; max_tokens~64
        ...
        return text.strip()

    return predict
```

Register in `get_predict_factory`:

```python
if key == "mlx":
    return mlx_predict_factory
```

Test: `get_predict_factory("mlx")` returns callable factory; calling without mlx raises FatalError on factory invocation (or on predict build).

- [ ] Commit: `feat: add MLX predict factory for eval`

---

### Task 6: README + full regression

**Files:**
- Modify: `README.md`

Document:
- `trainer.backend: auto` table (cuda→trl, else mlx, else error)
- Mac: `pip install -e '.[mlx]'`, `DEEPSEEK_API_KEY`, prefer ≤3B / auto
- CUDA: `pip install -e '.[train]'`
- MLX adapters ≠ PEFT

Run: `pytest -q` → all PASS (no mlx required).

- [ ] Commit: `docs: document auto trainer backend and MLX extras`

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| `auto` resolve cuda/mlx/error | 1–2 |
| Persist concrete backend; resume mismatch FatalError | 2 |
| MLX train + `[mlx]` extra | 4 |
| MLX predict | 5 |
| Allowlist 1.5B + bias + Darwin 16GB profile | 2–3 |
| README | 6 |
| CI fake offline | 6 |

## Self-review notes

- Flag names for mlx_lm.lora (`--lora-rank` vs `--rank`) **must be verified** at Task 4 implement time via `python -m mlx_lm.lora --help` or source — plan intentionally says confirm at implement time rather than guess wrong flags.
- Fake recommend model id must exist in allowlist after Task 3.
- `load_config` shallow merge: user omitting nested trainer keys still gets package defaults for missing keys via pydantic after yaml merge — switching default backend to `auto` affects users who relied on implicit `trl` without setting backend; README must mention.
