"""Unit tests for trainer backend finalization used by CLI run/resume."""

from autofinetune.trainer.resolve import finalize_trainer_backend


def test_run_path_resolves_before_persist(monkeypatch):
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.cuda_available", lambda: True
    )
    backend = finalize_trainer_backend("auto")
    assert backend == "trl"


def test_resume_path_keeps_stored_when_not_overridden():
    backend = finalize_trainer_backend(
        "auto",
        stored="fake",
        overridden=False,
        is_resume=True,
    )
    assert backend == "fake"
