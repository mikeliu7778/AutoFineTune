#!/usr/bin/env python3
"""Evaluate RAG query-intent field accuracy on a frozen holdout set.

Modes:
  1) Score an existing predictions file:
       python scripts/eval_rag_intent_holdout.py \\
         --gold datasets/rag_query_intent/holdout.jsonl \\
         --pred path/to/preds.jsonl

  2) Generate predictions from a completed AutoFineTune run (MLX/TRL), then score:
       python scripts/eval_rag_intent_holdout.py \\
         --gold datasets/rag_query_intent/holdout.jsonl \\
         --run runs/20260813-083721-22533979

pred jsonl lines accept:
  {"question": "...", "prediction": "{...json...}"}
  {"question": "...", "answer": "{...json...}"}   # answer = model output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from autofinetune.eval.rag_intent_metrics import (  # noqa: E402
    aggregate_scores,
    parse_json_answer,
    score_prediction,
)
from autofinetune.eval.rag_intent_validate import validate_and_fix  # noqa: E402

DEFAULT_GOLD = REPO_ROOT / "datasets" / "rag_query_intent" / "holdout.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise SystemExit(f"expected object lines in {path}")
        rows.append(obj)
    return rows


def _gold_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        q = str(r.get("question", "")).strip()
        if not q:
            continue
        ans = r.get("answer")
        gold = parse_json_answer(str(ans)) if not isinstance(ans, dict) else ans
        if gold is None:
            raise SystemExit(f"bad gold answer for question={q!r}")
        out[q] = gold
    return out


def _pred_text(row: dict[str, Any]) -> str:
    if "prediction" in row and row["prediction"] is not None:
        return str(row["prediction"])
    if "answer" in row and row["answer"] is not None:
        return str(row["answer"])
    raise SystemExit(f"pred row missing prediction/answer: {row!r}")


def _resolve_run_paths(run_dir: Path) -> tuple[str, Path, str]:
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise SystemExit(f"run.json not found: {run_json}")
    rec = json.loads(run_json.read_text(encoding="utf-8"))
    base = (rec.get("base_model") or {}).get("model_id")
    backend = (rec.get("trainer_backend") or "mlx").strip().lower()
    if not base:
        raise SystemExit(f"base_model.model_id missing in {run_json}")
    round_id = rec.get("best_round") or rec.get("current_round") or 1
    adapter = run_dir / "adapters" / f"r{round_id}"
    if not adapter.is_dir():
        # fallback: first adapters/r*
        cands = sorted((run_dir / "adapters").glob("r*"))
        if not cands:
            raise SystemExit(f"no adapter dir under {run_dir / 'adapters'}")
        adapter = cands[-1]
    return str(base), adapter, backend


def _build_predict_fn(
    *,
    backend: str,
    base_model_id: str,
    adapter_dir: Path,
    max_tokens: int,
) -> Callable[[str], str]:
    key = backend.strip().lower()
    if key == "mlx":
        try:
            from mlx_lm import generate, load
        except ImportError as e:
            raise SystemExit(
                "mlx_lm required for --run with mlx backend: pip install 'autofinetune[mlx]'"
            ) from e
        from autofinetune.eval.predict import _mlx_generation_prompt

        model, tokenizer = load(base_model_id, adapter_path=str(adapter_dir))

        def predict(q: str) -> str:
            prompt = _mlx_generation_prompt(tokenizer, q)
            return str(generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)).strip()

        return predict

    if key == "trl":
        from autofinetune.eval.predict import trl_predict_factory

        # factory uses max_new_tokens=64; wrap with longer decode via local copy
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise SystemExit(
                "TRL extras required: pip install 'autofinetune[train]'"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

        def predict(q: str) -> str:
            prompt = f"### Question:\n{q}\n\n### Answer:\n"
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[-1] :]
            return tokenizer.decode(gen, skip_special_tokens=True).strip()

        return predict

    raise SystemExit(f"unsupported backend for generation: {backend!r} (use mlx|trl)")


def _generate_preds(
    gold_rows: list[dict[str, Any]],
    predict_fn: Callable[[str], str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(gold_rows, 1):
        q = str(row["question"])
        print(f"[{i}/{len(gold_rows)}] {q}", flush=True)
        pred_text = predict_fn(q)
        out.append({"question": q, "prediction": pred_text})
    return out


def evaluate(
    gold_by_q: dict[str, dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    validate: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]], int]:
    per_row: list[dict[str, bool]] = []
    details: list[dict[str, Any]] = []
    parse_ok = 0
    missing = 0
    for row in pred_rows:
        q = str(row.get("question", "")).strip()
        if q not in gold_by_q:
            missing += 1
            continue
        gold = gold_by_q[q]
        raw = _pred_text(row)
        pred_obj = parse_json_answer(raw)
        if pred_obj is not None and validate:
            pred_obj = validate_and_fix(pred_obj)
        if pred_obj is None:
            scores = {
                "intent": False,
                "grade": False,
                "volume": False,
                "unit": False,
                "subject": False,
                "full": False,
            }
        else:
            parse_ok += 1
            scores = score_prediction(gold, pred_obj)
        per_row.append(scores)
        details.append(
            {
                "question": q,
                "gold": gold,
                "prediction_raw": raw,
                "prediction": pred_obj,
                "scores": scores,
            }
        )
    metrics = aggregate_scores(per_row)
    metrics["parse_ok"] = (parse_ok / len(per_row)) if per_row else 0.0
    metrics["missing_gold"] = float(missing)
    metrics["validated"] = 1.0 if validate else 0.0
    return metrics, details, parse_ok


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG intent holdout field-accuracy eval")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Gold holdout jsonl")
    p.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Predictions jsonl (skip generation)",
    )
    p.add_argument(
        "--run",
        type=Path,
        default=None,
        help="AutoFineTune run dir; generate preds with its adapter",
    )
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Explicit adapter dir (requires --base-model and --backend)",
    )
    p.add_argument("--base-model", default=None, help="HF model id for generation")
    p.add_argument(
        "--backend",
        default=None,
        choices=["mlx", "trl"],
        help="Generation backend when using --adapter-dir",
    )
    p.add_argument("--max-tokens", type=int, default=256, help="Generation max tokens")
    p.add_argument(
        "--out-pred",
        type=Path,
        default=None,
        help="Write generated/used predictions jsonl here",
    )
    p.add_argument(
        "--out-metrics",
        type=Path,
        default=None,
        help="Write metrics JSON here",
    )
    p.add_argument(
        "--out-details",
        type=Path,
        default=None,
        help="Write per-example details JSONL here",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Clamp pred fields to closed enums before scoring (recommended for serving)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.gold.is_file():
        raise SystemExit(f"gold file not found: {args.gold}")

    gold_rows = _load_jsonl(args.gold)
    gold_by_q = _gold_map(gold_rows)

    if args.pred is not None:
        pred_rows = _load_jsonl(args.pred)
    elif args.run is not None or args.adapter_dir is not None:
        if args.run is not None:
            base, adapter, backend = _resolve_run_paths(args.run.resolve())
        else:
            if not args.base_model or not args.backend or not args.adapter_dir:
                raise SystemExit(
                    "--adapter-dir requires --base-model and --backend"
                )
            base, adapter, backend = args.base_model, args.adapter_dir, args.backend
        print(f"generating with backend={backend} base={base} adapter={adapter}")
        predict_fn = _build_predict_fn(
            backend=backend,
            base_model_id=base,
            adapter_dir=adapter,
            max_tokens=args.max_tokens,
        )
        pred_rows = _generate_preds(gold_rows, predict_fn)
    else:
        raise SystemExit("provide --pred or --run (or --adapter-dir)")

    if args.out_pred is not None:
        args.out_pred.parent.mkdir(parents=True, exist_ok=True)
        args.out_pred.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in pred_rows) + "\n",
            encoding="utf-8",
        )
        print(f"wrote preds: {args.out_pred}")

    metrics, details, parse_ok = evaluate(gold_by_q, pred_rows, validate=args.validate)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        f"scored={int(metrics['n'])} parse_ok={parse_ok} "
        f"grade={metrics['grade']:.3f} intent={metrics['intent']:.3f} "
        f"full={metrics['full']:.3f}"
    )

    # Show a few failures
    fails = [d for d in details if not d["scores"]["full"]][:10]
    if fails:
        print("\nfailures (up to 10):")
        for d in fails:
            print("-", d["question"])
            print("  gold:", json.dumps(d["gold"], ensure_ascii=False))
            print("  pred:", json.dumps(d["prediction"], ensure_ascii=False))
            bad = [k for k, v in d["scores"].items() if k != "full" and not v]
            print("  miss:", ",".join(bad))

    if args.out_metrics is not None:
        args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote metrics: {args.out_metrics}")

    if args.out_details is not None:
        args.out_details.parent.mkdir(parents=True, exist_ok=True)
        args.out_details.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in details) + "\n",
            encoding="utf-8",
        )
        print(f"wrote details: {args.out_details}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
