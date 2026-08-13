# AutoFineTune

Closed-loop **domain-knowledge** fine-tuning agent (CLI). Cloud LLM plans rounds; local LoRA trains; LLM-as-judge evaluates.

## Quick start (fake / CI)

Offline CI and local smoke tests use fake backends — no GPU, torch, or MLX required:

```bash
pip install -e ".[dev]"
export AUTOFINETUNE_LLM=fake
export AUTOFINETUNE_TRAINER=fake
autofinetune run ./tests/fixtures/minimal_input --runs-dir ./runs --base-model auto
```

## Trainer backend (`trainer.backend`)

The default is **`auto`** (breaking change vs earlier versions that implicitly used TRL when extras were installed). Set in `config.yaml`, `--trainer`, or `AUTOFINETUNE_TRAINER`. The resolved concrete backend (`trl`, `mlx`, or `fake`) is persisted in `run.json` for resume/eval consistency.

| Value | Resolves to | When |
|-------|-------------|------|
| `auto` | `trl` | PyTorch installed and `torch.cuda.is_available()` |
| `auto` | `mlx` | No CUDA, but Apple Silicon + `mlx` importable |
| `auto` | *(FatalError)* | Neither CUDA nor MLX available |
| `trl` | `trl` | Force TRL/PEFT (CUDA); requires `[train]` |
| `mlx` | `mlx` | Force MLX LoRA; requires `[mlx]` on Apple Silicon |
| `fake` | `fake` | No real training (CI / tests) |

**MLX adapters ≠ PEFT adapters.** TRL writes Hugging Face PEFT checkpoints; MLX writes mlx-lm adapter artifacts. They are not interchangeable — resume and eval must use the same backend that produced the adapter.

### CUDA (NVIDIA)

```bash
pip install -e ".[train]"
export DEEPSEEK_API_KEY=...   # default orchestrator: DeepSeek deepseek-v4-flash
# Optional: override model in config.yaml → orchestrator.model: deepseek-v4-pro
# Optional: provider: openai + export OPENAI_API_KEY=...
autofinetune run ./my_input --base-model Qwen/Qwen2.5-7B-Instruct
# or let the orchestrator recommend:
autofinetune run ./my_input --base-model auto
```

With `backend: auto` (default), CUDA hosts pick TRL automatically. Override with `--trainer trl` if needed.

### Mac (Apple Silicon)

```bash
pip install -e ".[mlx]"
export DEEPSEEK_API_KEY=...   # same orchestrator defaults as CUDA
autofinetune run ./my_input --base-model auto
# or pin a small instruct model:
autofinetune run ./my_input --base-model Qwen/Qwen2.5-1.5B-Instruct
```

On Darwin, `auto` resolves to MLX. Prefer **≤3B** instruct models (e.g. `Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`) — the allowlist and recommend path bias toward small models on ~16GB unified memory. Override with `--trainer mlx` or `trainer.backend: mlx` in config.

When your config omits `gpu_profile`, Darwin defaults to `apple-unified-16gb` with `vram_gb: 16` so allowlist filtering matches typical Apple Silicon unified memory. Override in `config.yaml`:

```yaml
gpu_profile:
  name: apple-unified-36gb
  vram_gb: 36
```

MLX maps RoundPlan `lora.alpha` / `lora.r` to mlx-lm `scale = alpha/r` (PEFT-compatible effective scale).

DeepSeek API docs: https://api-docs.deepseek.com/zh-cn/

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

### Budgets

`max_rounds` and `max_wall_time_sec` are enforced (wall clock uses `started_at` across resume).  
`max_llm_cost_usd` is **best-effort** in v1: FakeLLM/LiteLLM clients add a fixed estimate (~$0.001 per `complete_json` call). It is **not** provider-metered billing.

See `docs/superpowers/specs/2026-08-12-autofinetune-design.md`.
