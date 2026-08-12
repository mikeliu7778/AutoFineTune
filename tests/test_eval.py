from autofinetune.eval.runner import evaluate_holdout
from autofinetune.llm.client import FakeLLMClient
from autofinetune.schemas import QAItem


def test_judge_primary_and_aux():
    holdout = [
        QAItem(question="Q1", answer="yes"),
        QAItem(question="Q2", answer="no"),
    ]
    llm = FakeLLMClient(
        handlers={
            "judge_qa": lambda s, u: {
                "scores": [
                    {"question": "Q1", "score": 1.0, "rationale": "ok"},
                    {"question": "Q2", "score": 0.0, "rationale": "bad"},
                ]
            }
        }
    )

    def predict(q: str) -> str:
        return "yes" if q == "Q1" else "maybe"

    metrics = evaluate_holdout(llm, holdout, predict)
    assert metrics.n_eval == 2
    assert metrics.judge_score == 0.5
    assert metrics.aux_exact_match == 0.5


def test_judge_failure_keeps_aux():
    holdout = [QAItem(question="Q1", answer="yes")]

    def predict(q: str) -> str:
        return "yes"

    from autofinetune.errors import RoundError
    from autofinetune.llm.client import FakeLLMClient as F

    class Boom(F):
        def complete_json(self, system, user, schema_name):
            raise RoundError("judge down")

    metrics = evaluate_holdout(Boom(handlers={}), holdout, predict)
    assert metrics.judge_score is None
    assert metrics.judge_error
    assert metrics.aux_exact_match == 1.0
