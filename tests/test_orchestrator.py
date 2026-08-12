import json
from pathlib import Path

from autofinetune.config import load_config
from autofinetune.errors import FatalError, RoundError
from autofinetune.llm.client import FakeLLMClient
from autofinetune.orchestrator.loop import run_experiment
from autofinetune.schemas import BaseModelChoice, RunStatus
from autofinetune.store.run_store import RunStore
from autofinetune.trainer.fake import FakeTrainer


def _plan_payload(s: str, u: str) -> dict:
    return {
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
    }


def _synth_payload(s: str, u: str) -> dict:
    return {
        "items": [
            {"question": f"Q{i}", "answer": f"A{i}", "source": "synthetic"}
            for i in range(8)
        ]
    }


def _judge_payload(s: str, u: str) -> dict:
    items = json.loads(u)["items"]
    return {
        "scores": [
            {"question": it["question"], "score": 0.8, "rationale": "ok"} for it in items
        ]
    }


def test_one_round_stop_with_fakes(tmp_path: Path):
    cfg = load_config(None)
    cfg.trainer.backend = "fake"
    cfg.budgets.max_rounds = 2
    cfg.runs_dir = tmp_path / "runs"
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("ACME billing domain knowledge", encoding="utf-8")

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=inp)

    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "default",
            },
            "round_plan": _plan_payload,
            "synthesize_qa": _synth_payload,
            "judge_qa": _judge_payload,
            "decide": lambda s, u: {
                "action": "stop",
                "hypothesis": "",
                "reason": "good enough",
            },
        }
    )

    def predict(q: str) -> str:
        return q.replace("Q", "A") if q.startswith("Q") else "A0"

    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        FakeTrainer(),
        base_model_arg="auto",
        predict_fn_factory=lambda **kwargs: predict,
    )
    assert final.status == RunStatus.completed
    assert final.base_model is not None
    assert final.current_round >= 1
    assert (store.best_path(rec.run_id)).is_file()
    judge_calls = [c for c in llm.calls if c[2] == "judge_qa"]
    assert len(judge_calls) == 1


def test_resume_skips_base_model_reselect(tmp_path: Path):
    cfg = load_config(None)
    cfg.trainer.backend = "fake"
    cfg.budgets.max_rounds = 1
    cfg.runs_dir = tmp_path / "runs"
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("ACME billing", encoding="utf-8")

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=inp)
    store.set_base_model(
        rec.run_id,
        BaseModelChoice(model_id="already/set", mode="user", rationale="pinned"),
    )

    recommend_calls = {"n": 0}

    def recommend(s: str, u: str) -> dict:
        recommend_calls["n"] += 1
        return {"model_id": "Qwen/Qwen2.5-7B-Instruct", "rationale": "should not run"}

    llm = FakeLLMClient(
        handlers={
            "recommend_model": recommend,
            "round_plan": _plan_payload,
            "synthesize_qa": _synth_payload,
            "judge_qa": _judge_payload,
            "decide": lambda s, u: {"action": "stop", "hypothesis": "", "reason": "done"},
        }
    )

    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        FakeTrainer(),
        base_model_arg="auto",
        resume_note="tweak data",
        predict_fn_factory=lambda **kwargs: (lambda q: "A0"),
    )
    assert recommend_calls["n"] == 0
    assert final.base_model is not None
    assert final.base_model.model_id == "already/set"
    assert final.user_note == "tweak data"


def test_pause_after_round_boundary(tmp_path: Path):
    cfg = load_config(None)
    cfg.trainer.backend = "fake"
    cfg.budgets.max_rounds = 3
    cfg.runs_dir = tmp_path / "runs"
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("ACME billing", encoding="utf-8")

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=inp)

    decide_n = {"n": 0}

    def decide(s: str, u: str) -> dict:
        decide_n["n"] += 1
        if decide_n["n"] == 1:
            store.request_pause(rec.run_id)
            return {"action": "continue", "hypothesis": "more", "reason": "continue"}
        return {"action": "stop", "hypothesis": "", "reason": "should not reach"}

    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "default",
            },
            "round_plan": _plan_payload,
            "synthesize_qa": _synth_payload,
            "judge_qa": _judge_payload,
            "decide": decide,
        }
    )

    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        FakeTrainer(),
        base_model_arg="auto",
        predict_fn_factory=lambda **kwargs: (lambda q: "A0"),
    )
    assert final.status == RunStatus.paused
    assert final.current_round == 1
    assert final.pause_requested is False
    assert decide_n["n"] == 1


def test_round_error_continues_fatal_fails(tmp_path: Path):
    cfg = load_config(None)
    cfg.trainer.backend = "fake"
    cfg.budgets.max_rounds = 2
    cfg.runs_dir = tmp_path / "runs"
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("ACME billing", encoding="utf-8")

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=inp)

    plan_n = {"n": 0}

    def plan(s: str, u: str) -> dict:
        plan_n["n"] += 1
        if plan_n["n"] == 1:
            raise RoundError("transient plan failure")
        return _plan_payload(s, u)

    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "default",
            },
            "round_plan": plan,
            "synthesize_qa": _synth_payload,
            "judge_qa": _judge_payload,
            "decide": lambda s, u: {"action": "stop", "hypothesis": "", "reason": "ok"},
        }
    )

    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        FakeTrainer(),
        base_model_arg="auto",
        predict_fn_factory=lambda **kwargs: (lambda q: "A0"),
    )
    assert final.status == RunStatus.completed
    assert final.current_round == 2
    assert (store.round_dir(rec.run_id, 1) / "report.md").is_file()

    # FatalError path
    store2 = RunStore(tmp_path / "runs2")
    rec2 = store2.create(input_dir=inp)
    cfg2 = load_config(None)
    cfg2.budgets.max_rounds = 1
    cfg2.runs_dir = tmp_path / "runs2"

    llm_fatal = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "default",
            },
            "round_plan": lambda s, u: (_ for _ in ()).throw(FatalError("boom")),
        }
    )
    try:
        run_experiment(
            cfg2,
            store2,
            rec2.run_id,
            llm_fatal,
            FakeTrainer(),
            base_model_arg="auto",
        )
        assert False, "expected FatalError"
    except FatalError:
        assert store2.load(rec2.run_id).status == RunStatus.failed


def test_pause_after_round_error(tmp_path: Path):
    cfg = load_config(None)
    cfg.trainer.backend = "fake"
    cfg.budgets.max_rounds = 3
    cfg.runs_dir = tmp_path / "runs"
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("ACME billing", encoding="utf-8")

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=inp)

    plan_n = {"n": 0}

    def plan(s: str, u: str) -> dict:
        plan_n["n"] += 1
        store.request_pause(rec.run_id)
        raise RoundError("plan failed mid-flight")

    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "default",
            },
            "round_plan": plan,
            "synthesize_qa": _synth_payload,
            "judge_qa": _judge_payload,
            "decide": lambda s, u: {
                "action": "continue",
                "hypothesis": "",
                "reason": "should not reach",
            },
        }
    )

    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        FakeTrainer(),
        base_model_arg="auto",
        predict_fn_factory=lambda **kwargs: (lambda q: "A0"),
    )
    assert final.status == RunStatus.paused
    assert final.current_round == 1
    assert final.pause_requested is False
    assert final.last_error == "plan failed mid-flight"
    assert plan_n["n"] == 1
    assert not (store.round_dir(rec.run_id, 2) / "plan.json").exists()


def test_plan_includes_prior_round_context(tmp_path: Path):
    cfg = load_config(None)
    cfg.trainer.backend = "fake"
    cfg.budgets.max_rounds = 2
    cfg.runs_dir = tmp_path / "runs"
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("ACME billing", encoding="utf-8")

    store = RunStore(cfg.runs_dir)
    rec = store.create(input_dir=inp)

    plan_users: list[dict] = []
    decide_n = {"n": 0}

    def plan(s: str, u: str) -> dict:
        plan_users.append(json.loads(u))
        return _plan_payload(s, u)

    def decide(s: str, u: str) -> dict:
        decide_n["n"] += 1
        if decide_n["n"] == 1:
            return {
                "action": "continue",
                "hypothesis": "try harder facts",
                "reason": "need another round",
            }
        return {"action": "stop", "hypothesis": "", "reason": "done"}

    llm = FakeLLMClient(
        handlers={
            "recommend_model": lambda s, u: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "default",
            },
            "round_plan": plan,
            "synthesize_qa": _synth_payload,
            "judge_qa": _judge_payload,
            "decide": decide,
        }
    )

    final = run_experiment(
        cfg,
        store,
        rec.run_id,
        llm,
        FakeTrainer(),
        base_model_arg="auto",
        predict_fn_factory=lambda **kwargs: (lambda q: "A0"),
    )
    assert final.status == RunStatus.completed
    assert len(plan_users) == 2
    assert "prior" not in plan_users[0]
    prior = plan_users[1]["prior"]
    assert prior["prior_round"] == 1
    assert "metrics" in prior
    assert "judge_score" in prior["metrics"]
    assert "report" in prior
    assert "Round 1" in prior["report"]
    assert prior["last_hypothesis"] == "try harder facts"
