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
export DEEPSEEK_API_KEY=...   # default orchestrator: DeepSeek deepseek-v4-flash
# Optional: override model in config.yaml → orchestrator.model: deepseek-v4-pro
# Optional: provider: openai + export OPENAI_API_KEY=...
autofinetune run ./my_input --base-model Qwen/Qwen2.5-7B-Instruct --trainer trl
# or let the orchestrator recommend:
autofinetune run ./my_input --base-model auto
```

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
