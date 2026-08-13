from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from autofinetune.datagen.prepare import read_jsonl
from autofinetune.errors import FatalError, RoundError

PredictFactory = Callable[..., Callable[[str], str]]


def lookup_predict_factory(**kwargs: Any) -> Callable[[str], str]:
    """Fake/default path: answer lookup from train.jsonl."""
    train_path: Path | None = kwargs.get("train_jsonl")
    lookup: dict[str, str] = {}
    if train_path and train_path.is_file():
        for item in read_jsonl(train_path):
            lookup[item.question] = item.answer

    def predict(q: str) -> str:
        return lookup.get(q, "")

    return predict


def trl_predict_factory(**kwargs: Any) -> Callable[[str], str]:
    """Load base model + PEFT adapter and generate short greedy answers."""
    base_model_id: str = kwargs["base_model_id"]
    adapter_dir: Path = Path(kwargs["adapter_dir"])
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise FatalError(
            "TRL predict requires extras: pip install 'autofinetune[train]'"
        ) from e

    try:
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
    except FatalError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RoundError(f"TRL predict load failed: {e}") from e

    def predict(q: str) -> str:
        prompt = f"### Question:\n{q}\n\n### Answer:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    return predict


def _mlx_generation_prompt(tokenizer: Any, question: str) -> Any:
    """Match mlx-lm CompletionsDataset: chat-wrap the ### Question prompt."""
    user_content = f"### Question:\n{question}\n\n### Answer:\n"
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return user_content
    messages = [{"role": "user", "content": user_content}]
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        return user_content


def mlx_predict_factory(**kwargs: Any) -> Callable[[str], str]:
    """Load base model + MLX adapter and generate short greedy answers."""
    base_model_id: str = kwargs["base_model_id"]
    adapter_dir: Path = Path(kwargs["adapter_dir"])
    try:
        from mlx_lm import generate, load
    except ImportError as e:
        raise FatalError(
            "MLX predict requires extras: pip install 'autofinetune[mlx]'"
        ) from e

    try:
        model, tokenizer = load(base_model_id, adapter_path=str(adapter_dir))
    except FatalError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RoundError(f"MLX predict load failed: {e}") from e

    def predict(q: str) -> str:
        prompt = _mlx_generation_prompt(tokenizer, q)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=64)
        return str(text).strip()

    return predict


def get_predict_factory(backend: str) -> PredictFactory:
    key = backend.strip().lower()
    if key == "fake":
        return lookup_predict_factory
    if key == "trl":
        return trl_predict_factory
    if key == "mlx":
        return mlx_predict_factory
    raise FatalError(f"Unknown predict factory for trainer backend: {backend}")
