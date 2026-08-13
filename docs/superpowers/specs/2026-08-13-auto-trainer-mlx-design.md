# Auto Trainer Backend (CUDA TRL + MLX Fallback) Design

**Date:** 2026-08-13  
**Status:** Approved for implementation  
**Scope:** `trainer.backend: auto` resolves to TRL on CUDA or MLX on Apple Silicon; expand small-model allowlist for ~16GB unified memory

## Problem

AutoFineTune’s real trainer (`TRLTrainerBackend`) assumes NVIDIA CUDA via PyTorch/TRL/PEFT. Machines without CUDA (e.g. MacBook Pro M3, 16GB unified memory) cannot fine-tune through the current path. Users still want a closed loop on Apple Silicon for small models (~1.5B–3B LoRA).

## Goals

- Default `trainer.backend: auto`:
  - CUDA available → existing **TRL** path
  - else MLX available (Apple Silicon) → **MLX** LoRA path
  - else clear `FatalError` with install/hardware guidance
- Explicit overrides: `trl` | `mlx` | `fake` (CLI `--trainer`, env `AUTOFINETUNE_TRAINER`)
- Persist the **resolved** backend (`trl` / `mlx` / `fake`) in `run.json` so resume/eval stay consistent
- Matching predict factory per resolved backend
- Allowlist adds ~1.5B instruct model; on Mac / low VRAM profiles, recommendation favors ≤3B
- Optional deps: `[train]` (CUDA stack) and `[mlx]` remain separate

## Non-goals

- Guaranteeing MLX adapters interchange with Hugging Face PEFT
- Bitsandbytes / QLoRA on Mac
- Windows DirectML / ROCm backends
- Changing orchestrator LLM (DeepSeek preset is a separate spec)
- Making CI require MLX or a GPU

## Approach

**Separate `MLXTrainerBackend` + resolver** (not a single mega-class).

```text
get_trainer(name) / resolve_trainer_backend(name)
  fake → FakeTrainer
  trl  → TRLTrainerBackend
  mlx  → MLXTrainerBackend
  auto → cuda? trl : mlx_importable? mlx : FatalError
```

Resolve **once** at `run` / `resume` entry (after CLI/env overrides), write concrete backend into `cfg.trainer.backend` and `RunRecord.trainer_backend`. Downstream orchestrator and `get_predict_factory` only see concrete names—never `auto`.

## Config

```yaml
trainer:
  backend: auto   # auto | trl | mlx | fake
  # existing LoRA defaults unchanged
```

| Value | Behavior |
|-------|----------|
| `auto` | Resolve at start; persist concrete backend |
| `trl` | Force TRL; missing `[train]` / no usable torch → FatalError |
| `mlx` | Force MLX; missing `[mlx]` → FatalError |
| `fake` | Unchanged CI path |

**Detection (auto):**

1. Prefer `torch.cuda.is_available()` → `trl` (only if torch import succeeds; if torch missing, do not treat as CUDA).
2. Else try `import mlx.core` (and optionally require `platform.system() == "Darwin"`) → `mlx`.
3. Else FatalError listing both extras and that CUDA or Apple Silicon+MLX is required for real training.

## MLX training

**Module:** `src/autofinetune/trainer/mlx_backend.py` implementing `TrainerBackend`.

- Inputs: HF `base_model_id`, `train.jsonl` (`question`/`answer`), `output_dir`, `RoundPlan`
- Prompt format: match TRL text template for consistency:
  `### Question:\n...\n\n### Answer:\n...`
- Use **mlx-lm** LoRA fine-tuning APIs (prefer in-process library calls over shelling out to CLI)
- Map from `RoundPlan.lora` where possible: `r`, `alpha`, `dropout`, `epochs`, `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`
- Unsupported knobs: log once at info/warning and ignore
- Write adapter artifacts under round `adapter/` directory; `TrainResult.backend == "mlx"`
- Failures → `RoundError`; missing deps → `FatalError` with `pip install 'autofinetune[mlx]'`

**Extras (`pyproject.toml`):**

```toml
[project.optional-dependencies]
mlx = ["mlx", "mlx-lm"]
# train = unchanged CUDA stack
# consider: dev stays on [train] for Linux CI; do not force mlx into default dev
```

## MLX predict / eval

**Module:** extend `eval/predict.py` with `mlx_predict_factory`.

- Load base + MLX adapter from `adapter_dir`
- Greedy short generation (similar budget to TRL predict, e.g. ~64 new tokens)
- `get_predict_factory("mlx")` → this factory
- v1: no conversion between PEFT and MLX adapters

## Allowlist & GPU profile

Update `defaults/allowlist.yaml`:

- Add `Qwen/Qwen2.5-1.5B-Instruct` (approx ~1.5B, `min_vram_gb` ≈ 4–6)
- Keep `Qwen/Qwen2.5-3B-Instruct` (`min_vram_gb: 8`)
- Keep 7B/8B entries for CUDA boxes

**Recommendation bias (Mac / low memory):**

- When resolved backend is `mlx` **or** `gpu_profile.vram_gb <= 16`, augment the recommend system prompt (or candidate ordering notes) so the LLM prefers the smallest fitting instruct model unless the brief clearly needs larger capacity.
- Existing hard filter `min_vram_gb <= gpu.vram_gb` stays.

**Default profile on Apple Silicon (optional but recommended in same change):**

- If no user config overrides `gpu_profile` and platform is Darwin, default toward a unified-memory style profile (e.g. name `apple-unified-16gb`, `vram_gb: 16`) instead of `single-24gb`, so allowlist filtering matches M3 16GB machines. Document how to override.

## CLI / resume

- `run` / `resume`: resolve `auto` before constructing trainer/predict factory
- If resume has stored `trainer_backend` and user did not override CLI/env, use stored concrete backend (already partially implemented)
- If user passes `--trainer auto` on resume, re-resolve (should match machine); if re-resolve differs from stored, FatalError or warn+require explicit force—**v1: FatalError on mismatch** to avoid loading wrong adapter format

## Docs

README updates:

- `backend: auto` behavior table
- Mac: `pip install -e '.[mlx]'`, export orchestrator key, pin or auto-select ≤3B
- CUDA: `pip install -e '.[train]'`
- Explicit: MLX adapters ≠ PEFT adapters

## Testing

- Unit: `resolve_trainer_backend` with mocked cuda / mlx availability matrix
- Unit: `get_predict_factory` accepts `mlx` (import-guarded)
- MLX train smoke: skip unless `mlx` installed; optional local-only
- Allowlist contains 1.5B; filter with `vram_gb=16` includes 1.5B/3B, excludes models with higher mins as expected
- Existing fake/orchestrator/CLI tests green offline

## Acceptance

1. CUDA host + `auto` → persists `trl`, train/eval unchanged  
2. M3 Mac + `[mlx]` + `auto` → persists `mlx`; 1.5B/3B LoRA round completes on small fixture  
3. No CUDA and no MLX → clear FatalError  
4. Allowlist + 16GB profile favors small models in auto recommend path  
5. CI with `AUTOFINETUNE_TRAINER=fake` passes without mlx/torch GPU  

## Out of scope / follow-ups

- Export MLX adapter → PEFT for CUDA serving
- Quantized MLX training presets
- Auto-download of mlx-community converted weights if HF id needs conversion (document if mlx-lm requires specific repos)
