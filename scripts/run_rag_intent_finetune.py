#!/usr/bin/env python3
"""Run AutoFineTune on the RAG query-intent dataset.

Reads DEEPSEEK_API_KEY from the environment for the real orchestrator
(unless --fake-llm is set). If the key is missing, falls back to a local
fake orchestrator with LoRA hyperparams matching the successful MLX run.

Examples:
  export DEEPSEEK_API_KEY=sk-...
  python scripts/run_rag_intent_finetune.py

  # no cloud LLM (still real MLX/TRL training unless --fake-trainer):
  python scripts/run_rag_intent_finetune.py --fake-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "datasets" / "rag_query_intent"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

# Match the successful 20260813 MLX LoRA plan (real training still uses these
# when orchestrator is fake — only plan/decide/judge are stubbed).
_GOOD_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.1,
    "epochs": 1,
    "learning_rate": 0.0002,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-1.5B-Instruct for RAG query intent via AutoFineTune"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"AutoFineTune input dir (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"HF model id (default: {DEFAULT_BASE_MODEL})",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Runs output dir (default: {DEFAULT_RUNS_DIR})",
    )
    p.add_argument(
        "--trainer",
        default="auto",
        choices=["auto", "trl", "mlx", "fake"],
        help="Trainer backend (default: auto)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path",
    )
    p.add_argument(
        "--fake-llm",
        action="store_true",
        help="Use local fake orchestrator (skip DeepSeek)",
    )
    p.add_argument(
        "--fake-trainer",
        action="store_true",
        help="Force fake trainer (no real LoRA)",
    )
    p.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Env var name for DeepSeek API key (default: DEEPSEEK_API_KEY)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and env checks, do not run",
    )
    return p.parse_args()


def _require_input(input_dir: Path) -> None:
    if not input_dir.is_dir():
        raise SystemExit(f"input dir not found: {input_dir}")
    brief = input_dir / "brief.md"
    qa = input_dir / "qa.jsonl"
    if not brief.is_file():
        raise SystemExit(f"missing {brief}")
    if not qa.is_file():
        raise SystemExit(f"missing {qa}")


def _build_training_fake_llm():
    from autofinetune.llm.client import FakeLLMClient

    def judge(_s, u):
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
                "model_id": DEFAULT_BASE_MODEL,
                "rationale": "rag intent default",
            },
            "round_plan": lambda s, u: {
                "data_strategy": (
                    "Use user qa.jsonl as-is (full route); curated holdout.jsonl frozen"
                ),
                "target_train_size": 2000,
                "lora": dict(_GOOD_LORA),
                "eval_focus": "intent + slot exact match; null/OOD robustness",
                "notes": "Prefer not inventing grade/subject when absent",
            },
            "synthesize_qa": lambda s, u: {"items": []},
            "judge_qa": judge,
            "decide": lambda s, u: {
                "action": "stop",
                "hypothesis": "",
                "reason": "single-round rag intent train",
            },
        }
    )


def main() -> int:
    args = _parse_args()
    _require_input(args.input_dir)

    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))

    key_name = args.api_key_env
    api_key = os.environ.get(key_name, "").strip()
    use_fake_llm = bool(args.fake_llm)
    if not use_fake_llm and not api_key:
        print(
            f"warning: {key_name} unset — falling back to --fake-llm "
            "(real trainer still runs; only orchestrator is stubbed)"
        )
        use_fake_llm = True

    from autofinetune.config import load_config
    from autofinetune.eval.predict import get_predict_factory
    from autofinetune.llm.client import LiteLLMClient
    from autofinetune.orchestrator.loop import run_experiment
    from autofinetune.store.run_store import RunStore
    from autofinetune.trainer.base import get_trainer
    from autofinetune.trainer.resolve import finalize_trainer_backend

    cfg = load_config(args.config)
    cfg.runs_dir = args.runs_dir.resolve()
    if args.fake_trainer:
        cfg.trainer.backend = "fake"
    else:
        cfg.trainer.backend = args.trainer
    cfg.trainer.backend = finalize_trainer_backend(cfg.trainer.backend)

    print("cwd:", REPO_ROOT)
    print("input:", args.input_dir.resolve())
    print("base_model:", args.base_model)
    print("trainer:", cfg.trainer.backend)
    print("llm:", "fake" if use_fake_llm else "deepseek/litellm")
    print(f"{key_name}={'set' if api_key else 'unset'}")
    if use_fake_llm:
        print("lora_plan:", json.dumps(_GOOD_LORA))

    if args.dry_run:
        return 0

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=args.input_dir.resolve())
    store.set_trainer_backend(rec.run_id, cfg.trainer.backend)
    print(f"Created run {rec.run_id}")

    llm = _build_training_fake_llm() if use_fake_llm else LiteLLMClient(cfg.orchestrator)
    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        get_trainer(cfg.trainer.backend),
        base_model_arg=args.base_model,
        predict_fn_factory=get_predict_factory(cfg.trainer.backend),
    )
    print(f"Status: {final.status.value}")
    if final.base_model:
        print(f"Base model: {final.base_model.model_id} ({final.base_model.mode})")
    print(f"run_dir: {store.run_dir(rec.run_id)}")
    return 0 if final.status.value in {"completed", "paused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
