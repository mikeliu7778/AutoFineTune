import pytest

from autofinetune.errors import FatalError
from autofinetune.trainer.resolve import resolve_trainer_backend


def test_fake_passthrough():
    assert resolve_trainer_backend("fake") == "fake"
    assert resolve_trainer_backend("FAKE") == "fake"


def test_trl_forced():
    assert resolve_trainer_backend("trl") == "trl"


def test_mlx_forced():
    assert resolve_trainer_backend("mlx") == "mlx"


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.cuda_available", lambda: True
    )
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.mlx_available", lambda: True
    )
    assert resolve_trainer_backend("auto") == "trl"


def test_auto_falls_back_to_mlx(monkeypatch):
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.cuda_available", lambda: False
    )
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.mlx_available", lambda: True
    )
    assert resolve_trainer_backend("auto") == "mlx"


def test_auto_neither_raises(monkeypatch):
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.cuda_available", lambda: False
    )
    monkeypatch.setattr(
        "autofinetune.trainer.resolve.mlx_available", lambda: False
    )
    with pytest.raises(FatalError, match="auto"):
        resolve_trainer_backend("auto")


def test_unknown_raises():
    with pytest.raises(FatalError, match="Unknown"):
        resolve_trainer_backend("bogus")
