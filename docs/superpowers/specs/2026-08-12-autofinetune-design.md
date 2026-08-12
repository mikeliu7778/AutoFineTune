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
- **Base model selection:** user-specified **or** orchestrator-recommended (constrained by GPU profile / allowlist).
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
- After ingest, **base model is resolved once** (user pin or LLM recommend).
- Inside a round, the orchestrator LLM chooses data strategy, LoRA/hyperparameters, and evaluation focus — not a new base model.
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
| Config | Models, API keys, GPU profile, budgets, base-model allowlist | YAML + env via **pydantic-settings** |
| ModelSelector | Resolve train base model: user pin vs LLM recommend | Custom; recommendation via orchestrator LLM + allowlist/GPU filters |

## Base model selection

The model being fine-tuned (train base) is distinct from the orchestrator LLM.

### Modes

| Mode | How | Behavior |
|------|-----|----------|
| **User-specified** | CLI `--base-model <id>` and/or config `base_model` | Use as-is after validation (exists / downloadable, fits GPU profile policy) |
| **LLM-recommended** | Omit pin; set `base_model: auto` (default when unset) | Before Round 1, orchestrator recommends from a **curated allowlist** given domain brief, data profile, and GPU profile; write choice + rationale into `run.json` |

### Rules (v1)

- Recommendation is **constrained**: only models on the allowlist that pass the active GPU profile (e.g. 24GB → prefer ~7B LoRA-capable entries). No open-ended “any HF repo” guessing in v1.
- Allowlist lives in config (Hugging Face repo ids + metadata: approx size, chat template family, notes). Users can extend the list.
- Selection happens **once per run** at start (after Ingest, before Round 1 Plan). Mid-run base switches are **out of scope** for v1 (would invalidate adapter comparability); user may start a new run or `resume --note` only affects plan/data/hyperparams, not base model.
- User pin always wins over auto. On `resume`, base model is whatever `run.json` already recorded.
- If auto-recommend fails validation (empty allowlist, none fit GPU), → fatal with actionable error (suggest pin or widen allowlist).

### CLI sketch

```text
autofinetune run ./input --base-model Qwen/Qwen2.5-7B-Instruct   # pin
autofinetune run ./input --base-model auto                         # recommend
autofinetune run ./input                                           # same as auto when config unset
```

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

### Run startup (before Round 1)

1. **Ingest** — normalize inputs; classify none / partial / full.
2. **Select base model** — user pin **or** LLM recommend from allowlist (see above); persist to `run.json`.

### Single round sequence

1. **Plan** — LLM reads prior report + data profile + fixed base model → structured plan (data strategy, LoRA/hyperparams, stop hints). Does **not** change base model in v1.
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

- Base model: user pin **or** auto-recommend from allowlist; allowlist default centers on ~7B instruct/chat checkpoints suitable for single-24GB LoRA.
- Method: LoRA / PEFT.
- Hardware assumption: single ~24GB GPU; smaller/larger via config (filters recommendations).
- Trainer backend selected in config (`trl` default; `unsloth` optional).

## Testing strategy

- Unit: ingest routing, plan/data schemas, metrics aggregation, model selection (pin vs auto, allowlist/GPU filter, fatal paths).
- Integration: fake Trainer + fake LLM drives a full outer-loop round (CI without GPU), including mocked recommend.
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
2. User can **pin** a base model or use **auto** recommendation; choice and rationale are recorded in `run.json`.
3. System completes ≥1 automatic round locally and writes judge-primary metrics + report.
4. Pause/resume works across process restarts via `runs/<id>/` (base model unchanged on resume).
5. Swap orchestrator provider via config (LiteLLM) without code changes.
6. CI passes with mocked LLM/trainer; no GPU required in CI.

## Open decisions (resolved for v1)

- Task: domain knowledge first.
- Loop style: fixed outer loop + LLM in-round planning.
- Surface: CLI first.
- Train locus: local v1, abstract backend.
- Base model: user-specified **or** LLM-recommended from GPU-filtered allowlist; selected once per run.
- Orchestrator: cloud API.
- Eval: LLM-as-judge primary.
- OSS: LiteLLM, HF PEFT/TRL, Typer/Rich, Pydantic; avoid heavy agent frameworks.
