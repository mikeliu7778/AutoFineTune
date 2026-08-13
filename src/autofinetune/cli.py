from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.pretty import pprint

from autofinetune.config import load_config
from autofinetune.eval.predict import get_predict_factory
from autofinetune.llm.client import FakeLLMClient, LiteLLMClient
from autofinetune.orchestrator.loop import run_experiment
from autofinetune.store.run_store import RunStore
from autofinetune.trainer.base import get_trainer
from autofinetune.trainer.resolve import finalize_trainer_backend

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
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "rationale": "fake small default",
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


def _apply_trainer_override(cfg, trainer: str | None) -> bool:
    """Apply CLI/env trainer override. Returns True if an override was applied."""
    if trainer:
        cfg.trainer.backend = trainer
        return True
    env = os.getenv("AUTOFINETUNE_TRAINER")
    if env:
        cfg.trainer.backend = env
        return True
    return False


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
    _apply_trainer_override(cfg, trainer)
    cfg.trainer.backend = finalize_trainer_backend(cfg.trainer.backend)

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=input_dir)
    store.set_trainer_backend(rec.run_id, cfg.trainer.backend)
    console.print(f"Created run [bold]{rec.run_id}[/bold]")
    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        _llm_from_env(cfg),
        get_trainer(cfg.trainer.backend),
        base_model_arg=base_model,
        predict_fn_factory=get_predict_factory(cfg.trainer.backend),
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
    trainer: Optional[str] = typer.Option(None, "--trainer"),
) -> None:
    cfg = load_config(config)
    cfg.runs_dir = runs_dir
    store = RunStore(runs_dir)
    rec = store.load(run_id)
    overridden = _apply_trainer_override(cfg, trainer)
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
    final = run_experiment(
        cfg,
        store,
        run_id,
        _llm_from_env(cfg),
        get_trainer(cfg.trainer.backend),
        resume_note=note,
        predict_fn_factory=get_predict_factory(cfg.trainer.backend),
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
