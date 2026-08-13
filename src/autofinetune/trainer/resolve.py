from __future__ import annotations

import platform

from autofinetune.errors import FatalError


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def mlx_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_trainer_backend(name: str) -> str:
    key = (name or "").strip().lower()
    if key == "fake":
        return "fake"
    if key == "trl":
        return "trl"
    if key == "mlx":
        return "mlx"
    if key == "auto":
        if cuda_available():
            return "trl"
        if mlx_available():
            return "mlx"
        raise FatalError(
            "trainer.backend=auto found neither CUDA (pip install 'autofinetune[train]') "
            "nor Apple Silicon MLX (pip install 'autofinetune[mlx]'). "
            "Set --trainer fake|trl|mlx explicitly."
        )
    raise FatalError(f"Unknown trainer backend: {name}")
