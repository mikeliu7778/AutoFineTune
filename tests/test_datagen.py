import pytest

from autofinetune.config import load_config
from autofinetune.datagen.prepare import prepare_datasets
from autofinetune.errors import RoundError
from autofinetune.ingest.bundle import IngestResult
from autofinetune.llm.client import FakeLLMClient
from autofinetune.schemas import DataRoute, LoraHyperparams, QAItem, RoundPlan


def test_full_route_splits_without_llm():
    cfg = load_config(None)
    qa = [QAItem(question=f"Q{i}", answer=f"A{i}") for i in range(100)]
    ingest = IngestResult(route=DataRoute.full, brief="b", qa=qa)
    plan = RoundPlan(data_strategy="use_user_qa", target_train_size=80)
    llm = FakeLLMClient(handlers={})
    result = prepare_datasets(cfg, ingest, plan, llm, existing_holdout=None)
    assert len(result.holdout) >= 1
    assert len(result.train) >= 1
    hold_q = {x.question for x in result.holdout}
    assert all(t.question not in hold_q for t in result.train)


def test_full_route_single_qa_raises():
    cfg = load_config(None)
    qa = [QAItem(question="Q0", answer="A0")]
    ingest = IngestResult(route=DataRoute.full, brief="b", qa=qa)
    plan = RoundPlan(data_strategy="use_user_qa", target_train_size=1)
    llm = FakeLLMClient(handlers={})
    with pytest.raises(RoundError, match="at least 2"):
        prepare_datasets(cfg, ingest, plan, llm, existing_holdout=None)


def test_none_route_synthesizes_train_and_holdout():
    cfg = load_config(None)
    ingest = IngestResult(route=DataRoute.none, brief="ACME refunds", docs_text="Refunds in 14 days")
    plan = RoundPlan(data_strategy="synthesize", target_train_size=5, lora=LoraHyperparams())
    llm = FakeLLMClient(
        handlers={
            "synthesize_qa": lambda s, u: {
                "items": [
                    {"question": f"Q{i}", "answer": f"A{i}", "source": "synthetic"}
                    for i in range(8)
                ]
            }
        }
    )
    result = prepare_datasets(cfg, ingest, plan, llm, existing_holdout=None)
    assert len(result.train) >= 1
    assert len(result.holdout) >= 1


def test_none_route_existing_holdout_all_synth_overlap_raises():
    cfg = load_config(None)
    ingest = IngestResult(route=DataRoute.none, brief="ACME refunds", docs_text="Refunds in 14 days")
    plan = RoundPlan(data_strategy="synthesize", target_train_size=5, lora=LoraHyperparams())
    holdout = [QAItem(question=f"Q{i}", answer=f"H{i}") for i in range(3)]
    llm = FakeLLMClient(
        handlers={
            "synthesize_qa": lambda s, u: {
                "items": [
                    {"question": f"Q{i}", "answer": f"A{i}", "source": "synthetic"}
                    for i in range(3)
                ]
            }
        }
    )
    with pytest.raises(RoundError, match="holdout"):
        prepare_datasets(cfg, ingest, plan, llm, existing_holdout=holdout)
