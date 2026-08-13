from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from autofinetune.config import AppConfig
from autofinetune.datagen.prepare import prepare_datasets, read_jsonl, write_jsonl
from autofinetune.errors import FatalError, RoundError
from autofinetune.eval.predict import get_predict_factory, lookup_predict_factory
from autofinetune.eval.runner import evaluate_holdout
from autofinetune.ingest.bundle import IngestResult, ingest_bundle
from autofinetune.llm.client import LLMClient
from autofinetune.model_select.selector import select_base_model
from autofinetune.schemas import DecideResult, RoundMetrics, RoundPlan, RunStatus
from autofinetune.store.run_store import RunRecord, RunStore
from autofinetune.trainer.base import TrainerBackend

PredictFactory = Callable[..., Callable[[str], str]]

# Back-compat alias for tests / callers that imported the default lookup factory.
_default_predict_factory = lookup_predict_factory


def _parse_started_at(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _sync_llm_cost(store: RunStore, run_id: str, llm: LLMClient) -> None:
    cost = getattr(llm, "cost_usd_est", None)
    if cost is None:
        return
    rec = store.load(run_id)
    rec.llm_cost_usd_est = float(cost)
    store.save(rec)


def run_experiment(
    cfg: AppConfig,
    store: RunStore,
    run_id: str,
    llm: LLMClient,
    trainer: TrainerBackend,
    base_model_arg: str | None = None,
    resume_note: str | None = None,
    predict_fn_factory: PredictFactory | None = None,
) -> RunRecord:
    if predict_fn_factory is None:
        predict_fn_factory = get_predict_factory(cfg.trainer.backend)
    rec = store.load(run_id)
    if resume_note:
        rec.user_note = resume_note
        store.save(rec)

    if rec.trainer_backend is None:
        store.set_trainer_backend(run_id, cfg.trainer.backend)
        rec = store.load(run_id)

    input_dir = store.run_dir(run_id) / "input"
    ingest = ingest_bundle(input_dir, cfg)
    store.set_route(run_id, ingest.route.value)

    # resume: never re-select if base_model already recorded
    if rec.base_model is None:
        choice = select_base_model(
            cfg, ingest, llm, base_model_arg, trainer_backend=cfg.trainer.backend
        )
        store.set_base_model(run_id, choice)
        rec = store.load(run_id)

    if rec.started_at is None:
        rec.started_at = datetime.now(timezone.utc).isoformat()
        store.save(rec)
        rec = store.load(run_id)

    store.set_status(run_id, RunStatus.running)
    assert rec.started_at is not None
    started_ts = _parse_started_at(rec.started_at)
    start_round = rec.current_round + 1

    for round_idx in range(start_round, cfg.budgets.max_rounds + 1):
        rec = store.load(run_id)
        _sync_llm_cost(store, run_id, llm)
        rec = store.load(run_id)
        if cfg.budgets.max_wall_time_sec and (time.time() - started_ts) > cfg.budgets.max_wall_time_sec:
            store.set_status(run_id, RunStatus.completed)
            break
        if (
            cfg.budgets.max_llm_cost_usd is not None
            and rec.llm_cost_usd_est >= cfg.budgets.max_llm_cost_usd
        ):
            store.set_status(run_id, RunStatus.completed)
            break

        try:
            _run_one_round(
                cfg,
                store,
                run_id,
                llm,
                trainer,
                ingest,
                rec,
                round_idx,
                predict_fn_factory,
            )
            _sync_llm_cost(store, run_id, llm)
            if store.load(run_id).pause_requested:
                store.set_status(run_id, RunStatus.paused)
                store.clear_pause(run_id)
                return store.load(run_id)

            decide_path = store.round_dir(run_id, round_idx) / "decide.json"
            decide = DecideResult.model_validate_json(
                decide_path.read_text(encoding="utf-8")
            )
            if decide.action == "stop":
                store.set_status(run_id, RunStatus.completed)
                return store.load(run_id)

        except RoundError as e:
            _sync_llm_cost(store, run_id, llm)
            rec = store.load(run_id)
            rec.current_round = round_idx
            rec.last_error = str(e)
            store.save(rec)
            store.save_report(run_id, round_idx, f"# Round {round_idx} FAILED\n\n{e}\n")
            if store.load(run_id).pause_requested:
                store.set_status(run_id, RunStatus.paused)
                store.clear_pause(run_id)
                return store.load(run_id)
            continue
        except FatalError:
            store.set_status(run_id, RunStatus.failed)
            raise

    store.set_status(run_id, RunStatus.completed)
    return store.load(run_id)


def _run_one_round(
    cfg: AppConfig,
    store: RunStore,
    run_id: str,
    llm: LLMClient,
    trainer: TrainerBackend,
    ingest: IngestResult,
    rec: RunRecord,
    round_idx: int,
    predict_fn_factory: PredictFactory,
) -> None:
    plan = _plan_round(llm, cfg, ingest, rec, round_idx, store, run_id)
    store.save_plan(run_id, round_idx, plan)

    existing_holdout = None
    hp = store.holdout_path(run_id)
    if not hp.is_file():
        # Prefer curated holdout shipped with the input bundle when present.
        curated = store.run_dir(run_id) / "input" / "holdout.jsonl"
        if curated.is_file():
            hp.write_text(curated.read_text(encoding="utf-8"), encoding="utf-8")
    if hp.is_file():
        existing_holdout = read_jsonl(hp)

    prepared = prepare_datasets(cfg, ingest, plan, llm, existing_holdout)
    if not hp.is_file():
        write_jsonl(hp, prepared.holdout)
    train_path = store.round_dir(run_id, round_idx) / "train.jsonl"
    write_jsonl(train_path, prepared.train)

    assert rec.base_model is not None
    adapter_dir = store.adapter_dir(run_id, round_idx)
    trainer.train(rec.base_model.model_id, train_path, adapter_dir, plan)

    predict = predict_fn_factory(
        base_model_id=rec.base_model.model_id,
        adapter_dir=adapter_dir,
        train_jsonl=train_path,
    )
    # Evaluate ONCE per round on frozen holdout from disk
    holdout_items = read_jsonl(store.holdout_path(run_id))
    metrics = evaluate_holdout(llm, holdout_items, predict)
    store.save_metrics(run_id, round_idx, metrics)

    report = (
        f"# Round {round_idx}\n\n"
        f"Base: `{rec.base_model.model_id}`\n\n"
        f"Judge: {metrics.judge_score}\n\n"
        f"Aux EM: {metrics.aux_exact_match}\n\n"
        f"Plan: {plan.data_strategy}\n"
    )
    store.save_report(run_id, round_idx, report)

    _maybe_update_best(store, run_id, round_idx, metrics)

    decide = _decide(llm, rec, round_idx, metrics, cfg)
    store.save_decide(run_id, round_idx, decide)

    rec = store.load(run_id)
    rec.current_round = round_idx
    rec.last_error = None
    store.save(rec)


def _prior_round_context(store: RunStore, run_id: str, round_idx: int) -> dict[str, Any]:
    """Artifacts from the previous round for planning context (round_idx > 1)."""
    prev = round_idx - 1
    rd = store.round_dir(run_id, prev)
    ctx: dict[str, Any] = {"prior_round": prev}

    metrics_path = rd / "metrics.json"
    if metrics_path.is_file():
        ctx["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))

    report_path = rd / "report.md"
    if report_path.is_file():
        ctx["report"] = report_path.read_text(encoding="utf-8")[:4000]

    decide_path = rd / "decide.json"
    if decide_path.is_file():
        decide = json.loads(decide_path.read_text(encoding="utf-8"))
        if "hypothesis" in decide:
            ctx["last_hypothesis"] = decide["hypothesis"]
        ctx["last_decide"] = decide

    return ctx


def _plan_round(
    llm: LLMClient,
    cfg: AppConfig,
    ingest: IngestResult,
    rec: RunRecord,
    round_idx: int,
    store: RunStore,
    run_id: str,
) -> RoundPlan:
    user: dict[str, Any] = {
        "round": round_idx,
        "base_model": rec.base_model.model_dump() if rec.base_model else None,
        "route": ingest.route.value,
        "brief": ingest.brief[:3000],
        "user_note": rec.user_note,
        "last_error": rec.last_error,
        "defaults": {
            "target_train_size": cfg.data.target_train_size,
            "lora_r": cfg.trainer.default_lora_r,
        },
    }
    if round_idx > 1:
        user["prior"] = _prior_round_context(store, run_id, round_idx)
    out = llm.complete_json(
        system=(
            "Plan one fine-tuning round for domain knowledge. "
            "Do not change base model. Return JSON matching RoundPlan fields: "
            "data_strategy, target_train_size, lora{...}, eval_focus, notes."
        ),
        user=json.dumps(user, ensure_ascii=False),
        schema_name="round_plan",
    )
    try:
        return RoundPlan.model_validate(out)
    except ValidationError as e:
        raise RoundError(f"Invalid RoundPlan from LLM: {e}") from e


def _decide(
    llm: LLMClient,
    rec: RunRecord,
    round_idx: int,
    metrics: RoundMetrics,
    cfg: AppConfig,
) -> DecideResult:
    out = llm.complete_json(
        system=(
            "Decide whether to continue fine-tuning rounds. "
            "Return JSON: action(continue|stop), hypothesis, reason."
        ),
        user=json.dumps(
            {
                "round": round_idx,
                "metrics": metrics.model_dump(),
                "max_rounds": cfg.budgets.max_rounds,
                "best_round": rec.best_round,
            },
            ensure_ascii=False,
        ),
        schema_name="decide",
    )
    try:
        return DecideResult.model_validate(out)
    except ValidationError as e:
        raise RoundError(f"Invalid DecideResult from LLM: {e}") from e


def _maybe_update_best(
    store: RunStore,
    run_id: str,
    round_idx: int,
    metrics: RoundMetrics,
) -> None:
    rec = store.load(run_id)
    score = metrics.judge_score
    if score is None:
        score = metrics.aux_exact_match or 0.0
    if rec.best_round is None:
        store.update_best(run_id, round_idx, metrics)
        return
    best = json.loads(store.best_path(run_id).read_text(encoding="utf-8"))
    prev = best.get("metrics", {})
    prev_score = prev.get("judge_score")
    if prev_score is None:
        prev_score = prev.get("aux_exact_match") or 0.0
    if score >= prev_score:
        store.update_best(run_id, round_idx, metrics)
