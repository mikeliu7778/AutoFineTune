# RAG Query Intent Fine-tune with AutoFineTune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with AutoFineTune so user education RAG queries map to normalized metadata JSON (`intent` / `grade` / `volume` / `unit` / `subject`).

**Architecture:** Use existing AutoFineTune closed loop (ingest QA → LoRA train → eval → decide). Seed data lives under `datasets/rag_query_intent/`. Add a small offline field-accuracy evaluator for `holdout.jsonl` because v1 LLM-as-judge is not field-exact. Optional later: specialize judge prompts.

**Tech Stack:** AutoFineTune CLI, Qwen2.5-1.5B-Instruct, MLX (Mac) or TRL (CUDA), DeepSeek orchestrator (optional), pytest

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-13-rag-query-intent-normalize.md`
- Dataset: `datasets/rag_query_intent/` (`brief.md`, `qa.jsonl`, `holdout.jsonl`)
- Pin base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Do not merge `holdout.jsonl` into `qa.jsonl`
- Grade enum for v1 seed: only `七年级|八年级|九年级` (+ null)
- Commit author `mike <mliu36292@gmail.com>`; no Cursor co-author trailers

## File Structure

| Path | Responsibility |
|------|----------------|
| `datasets/rag_query_intent/*` | Seed train/holdout + brief (already generated) |
| `scripts/eval_rag_intent_holdout.py` | Parse model JSON answers vs holdout; print field accuracies |
| `docs/superpowers/specs/2026-08-13-rag-query-intent-normalize.md` | Update status + link to this plan |
| `README.md` (optional one-liner) | Point to dataset example |

---

### Task 1: Freeze enums + document dataset

**Files:**
- Verify: `datasets/rag_query_intent/{brief.md,qa.jsonl,holdout.jsonl,enums.json,README.md}`
- Modify: `docs/superpowers/specs/2026-08-13-rag-query-intent-normalize.md` (status → Ready for fine-tune; link plan + dataset)

- [ ] **Step 1: Sanity-check jsonl**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("datasets/rag_query_intent")
for name in ("qa.jsonl", "holdout.jsonl"):
    rows = [json.loads(l) for l in (root/name).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, name
    for r in rows:
        assert "question" in r and "answer" in r
        obj = json.loads(r["answer"])
        assert obj["intent"] in {"summary", "exercises", "knowledge", "unknown"}
        assert obj["grade"] in {"七年级", "八年级", "九年级", None}
        assert obj["volume"] in {"上", "下", None}
    print(name, len(rows), "ok")
q = {json.loads(l)["question"] for l in (root/"qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
h = {json.loads(l)["question"] for l in (root/"holdout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
assert not (q & h), f"leak {q & h}"
print("no train/holdout leakage")
PY
```

Expected: both files ok; no leakage.

- [ ] **Step 2: Update spec status block** to link this plan and dataset path.

- [ ] **Step 3: Commit**

```bash
git add datasets/rag_query_intent docs/superpowers/specs/2026-08-13-rag-query-intent-normalize.md docs/superpowers/plans/2026-08-13-rag-query-intent-finetune.md
git commit -m "docs: add RAG query-intent dataset and fine-tune plan"
```

---

### Task 2: Offline field-accuracy eval script

**Files:**
- Create: `scripts/eval_rag_intent_holdout.py`
- Create: `tests/test_eval_rag_intent_holdout.py` (unit-test pure scoring helper)

**Interfaces:**
- `score_prediction(gold: dict, pred: dict) -> dict[str, bool]` keys: intent, grade, volume, unit, subject, full
- CLI: `python scripts/eval_rag_intent_holdout.py --gold datasets/rag_query_intent/holdout.jsonl --pred path/to/preds.jsonl`
- `preds.jsonl` lines: `{"question": "...", "prediction": "{...json...}"}` or `{"question","answer"}` where answer is model output string

- [ ] **Step 1: Failing test for scorer**

```python
from scripts.eval_rag_intent_holdout import score_prediction  # or import from autofinetune package helper

def test_score_prediction_grade_alias_normalized():
    gold = {"intent": "summary", "grade": "八年级", "volume": "上", "unit": 1, "subject": None}
    pred = {"intent": "summary", "grade": "八年级", "volume": "上", "unit": 1, "subject": None}
    s = score_prediction(gold, pred)
    assert s["full"] is True
    assert s["grade"] is True
```

Prefer placing scorer in `src/autofinetune/eval/rag_intent_metrics.py` and thin CLI in `scripts/` if imports are cleaner.

- [ ] **Step 2–4: Implement scorer + CLI; pytest pass; commit**

```bash
git commit -m "feat: add RAG intent holdout field-accuracy evaluator"
```

---

### Task 3: Smoke AutoFineTune run (fake trainer)

**Files:** none required beyond dataset

- [ ] **Step 1: Fake end-to-end**

```bash
export AUTOFINETUNE_LLM=fake
export AUTOFINETUNE_TRAINER=fake
autofinetune run ./datasets/rag_query_intent \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --runs-dir ./runs
```

Expected: exit 0; run directory created.

- [ ] **Step 2: Commit nothing** (or commit a short note in dataset README if command differs). Document exact command in dataset README if missing.

---

### Task 4: Real LoRA fine-tune on this machine

- [ ] **Step 1: Install backend**

Mac:

```bash
pip install -e '.[mlx]'
export DEEPSEEK_API_KEY=...   # or AUTOFINETUNE_LLM=fake to skip cloud planning quality
```

CUDA:

```bash
pip install -e '.[train]'
```

- [ ] **Step 2: Run**

```bash
autofinetune run ./datasets/rag_query_intent \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --runs-dir ./runs \
  --trainer auto
```

Expected: resolves to `mlx` on Darwin / `trl` on CUDA; writes adapter under `runs/<id>/rounds/*/adapter/`.

- [ ] **Step 3: Generate holdout predictions** (manual or small script using mlx/trl predict). Save `runs/<id>/holdout_preds.jsonl`.

- [ ] **Step 4: Score**

```bash
python scripts/eval_rag_intent_holdout.py \
  --gold datasets/rag_query_intent/holdout.jsonl \
  --pred runs/<id>/holdout_preds.jsonl
```

**Acceptance (seed data):**

- `grade` exact match ≥ 0.85 on holdout  
- `intent` ≥ 0.80  
- `full` ≥ 0.60（缺槽位样本拉低满分率，可接受）

Record metrics in `runs/<id>/rag_intent_metrics.json` (script should write this).

---

### Task 5: Rule fallback note + optional expansion

- [ ] Document in `datasets/rag_query_intent/README.md` the A+C merge strategy:  
  - rules catch empty/low-confidence  
  - model primary for alias-heavy queries  
- [ ] Optional: expand `qa.jsonl` with more real logs (keep holdout frozen)

- [ ] Commit doc updates

```bash
git commit -m "docs: document RAG intent rule+model merge strategy"
```

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| Dataset for AutoFineTune | 1 (delivered with plan) |
| Enums / grade map | 1 + brief.md |
| Field-level metrics | 2, 4 |
| Run AutoFineTune 1.5B | 3–4 |
| Rules + model merge | 5 |

## Notes

- AutoFineTune internal holdout split is separate from `holdout.jsonl`; always evaluate aliases on the frozen file.
- If orchestrator synthesizes extra QA, ensure synthesizer prompt includes the same grade map (from `brief.md`).
- Product may later extend grades to 高中/小学 — treat as new dataset version, do not silently mix enums.
