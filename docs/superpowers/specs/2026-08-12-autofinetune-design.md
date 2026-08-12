# AutoFineTune Design

**Date:** 2026-08-12  
**Status:** Draft for implementation  
**Scope:** v1 closed-loop domain-knowledge fine-tuning agent (CLI, local LoRA)

## Problem

Building full “automatic AI research” is too broad. We want a narrower product: an agent that iteratively improves a **domain-knowledge** fine-tune using strong cloud LLMs as the researcher, while training runs **locally**. Users may bring no data, partial data, or full datasets.

## Goals (v1)

- Closed-loop experimentation: plan → prepare data → train → evaluate → decide next round.
- Domain / vertical knowledge adaptation as the only supported task type.
- Flexible inputs: domain brief, documents, and/or QA pairs; auto-route by what is present.
- Default fully automatic runs, with pause / resume / optional user notes at round boundaries.
- CLI-first UX; artifacts and reports on disk.
- Pluggable trainer abstraction; v1 implements **local LoRA** only (default profile: ~7B on single 24GB GPU).
- Orchestrator via **configurable cloud LLM APIs**.
- Evaluation: **LLM-as-judge primary**, simple metrics secondary.
- Prefer suitable open-source libraries over custom reinvention.

## Non-goals (v1)

- Web UI
- Hosted / cloud training backends (interface only)
- General instruction SFT product surface (roadmap)
- Preference optimization (DPO/ORPO) (roadmap)
- Heavy multi-agent frameworks (LangGraph/AutoGen as core runtime)
- Full experiment platform (MLflow) as a hard dependency
- Guaranteeing edits to raw files under `rounds/` as a supported intervention API

## Approach

**Fixed outer loop + LLM planning inside each round** (not a free-form tool agent).

- Outer loop stages are fixed and resumable.
- Inside a round, the orchestrator LLM chooses data strategy, LoRA/hyperparameters, and evaluation focus.
- Budget guards force stop when limits are hit; best checkpoint is retained.

## Architecture

```text
User (CLI)
    │  run / pause / resume / status / report
    ▼
┌─────────────────────────────────────────┐
│  Orchestrator (outer loop: Round 1..N)  │
│  Cloud LLM via LiteLLM                  │
│  plan → execute → judge → decide        │
└───────────┬─────────────────────────────┘
            │ tool-like module calls
    ┌───────┼────────┬──────────┐
    ▼       ▼        ▼          ▼
 Ingest   DataGen   Trainer    Evaluator
 (route)  (synth)   (backend)  (judge+)
            │          │
            ▼          ▼
         datasets/   adapters/
            └──── Experiment Store (runs/) ────┘
```

### Components

| Component | Responsibility | v1 choice |
|-----------|----------------|-----------|
| CLI | Commands for lifecycle and inspection | Typer + Rich |
| Orchestrator | Round state machine; LLM plan/decide | Lightweight custom state machine; **LiteLLM** for providers |
| Ingest | Normalize brief / docs / QA; classify none / partial / full | Custom; docs: Markdown/text + pypdf (richer parsers optional later) |
| DataGen | Full synthesis, gap-fill, or QA QC + split | Orchestrator LLM + **Pydantic** schemas; optional **instructor** |
| Trainer | `TrainerBackend` interface; local LoRA | **HF transformers + PEFT + TRL**; optional **Unsloth** backend |
| Evaluator | Holdout QA; judge score primary | Custom pipeline; judge via LiteLLM |
| Experiment Store | Run metadata, per-round artifacts | Directory layout + JSON (`run.json`, `rounds/*.json`) |
| Config | Models, API keys, GPU profile, budgets | YAML + env via **pydantic-settings** |

## Data flow

### Input bundle

```text
input/
├── brief.md          # domain description (optional but recommended)
├── docs/             # knowledge sources (optional)
└── qa.jsonl          # existing QA (optional)
```

### Minimum input

At least one of: non-empty `brief.md`, files under `docs/`, or `qa.jsonl`. If all are missing/empty → fatal error before any round starts.

### Routing

| Classification | Behavior |
|----------------|----------|
| `none` | No usable QA; synthesize train + holdout from brief and/or docs |
| `partial` | Some QA present but below configured sufficiency threshold; keep user QA, synthesize to fill gaps; docs as knowledge source |
| `full` | QA meets sufficiency threshold; validate; split train/holdout (**holdout never used for training**) |

`partial` vs `full` uses a config threshold (e.g. minimum QA count); exact default is set in implementation config, not hardcoded in product logic beyond “configurable and documented”.

### Single round sequence

1. **Plan** — LLM reads prior report + data profile → structured plan (data strategy, hyperparams, stop hints).
2. **Prepare data** — Generate / augment / resample → `rounds/rN/train.jsonl`.
3. **Train** — `TrainerBackend` → `adapters/rN/`.
4. **Eval** — Base vs adapter on the same holdout; LLM-as-judge primary; auxiliary string/match metrics.
5. **Decide** — LLM chooses `continue` or `stop` given scores and budget; may write next-round hypothesis.
6. **Checkpoint** — Update `run.json`; honor pause at **round boundary**.

### Run directory layout

```text
runs/<run_id>/
  run.json
  input/
  holdout.jsonl
  rounds/r1/{plan.json, train.jsonl, metrics.json, report.md}
  adapters/r1/
  best.json
```

## Collaboration model

- Default: fully automatic until stop or budget.
- `pause`: finish current train/eval step if possible, then stop at round boundary (`paused`).
- `resume [--note "..."]`: continue from next Plan, injecting optional user note into planner context.
- Supported intervention: CLI notes + config overrides — not ad-hoc mutation of round files.

## Budgets and failure handling

**Budgets:** `max_rounds`, `max_wall_time`, optional `max_llm_cost` estimate. On hit → forced `stop`, keep `best`.

**Failures:**

| Class | Behavior |
|-------|----------|
| Retryable (rate limit, transient IO) | Exponential backoff; logged |
| Round failure (OOM, data validation) | Mark round `failed`; planner gets error summary; may replan or stop on budget |
| Fatal (bad API key, missing base weights, impossible GPU profile) | Fail CLI immediately with clear error |
| Judge failure | Keep auxiliary metrics and raw generations; do not discard training artifacts |

## Default training profile

- Base model class: ~7B instruct/chat checkpoint (configurable).
- Method: LoRA / PEFT.
- Hardware assumption: single ~24GB GPU; smaller/larger via config.
- Trainer backend selected in config (`trl` default; `unsloth` optional).

## Testing strategy

- Unit: ingest routing, plan/data schemas, metrics aggregation.
- Integration: fake Trainer + fake LLM drives a full outer-loop round (CI without GPU).
- Manual smoke: tiny step count / small model config on a real GPU when available.

## Roadmap (out of v1)

- Web UI
- Hosted trainer backends behind the same interface
- General instruction SFT mode
- Preference tuning (DPO/ORPO)
- Richer document ingestion (e.g. unstructured)
- MLflow / richer experiment tracking
- Stronger cost accounting and multi-GPU

## Success criteria for v1

1. User can start a run with any of: brief-only, docs±brief, QA±docs, or full QA.
2. System completes ≥1 automatic round locally and writes judge-primary metrics + report.
3. Pause/resume works across process restarts via `runs/<id>/`.
4. Swap orchestrator provider via config (LiteLLM) without code changes.
5. CI passes with mocked LLM/trainer; no GPU required in CI.

## Open decisions (resolved for v1)

- Task: domain knowledge first.
- Loop style: fixed outer loop + LLM in-round planning.
- Surface: CLI first.
- Train locus: local v1, abstract backend.
- Orchestrator: cloud API.
- Eval: LLM-as-judge primary.
- OSS: LiteLLM, HF PEFT/TRL, Typer/Rich, Pydantic; avoid heavy agent frameworks.
