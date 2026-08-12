from __future__ import annotations

from pathlib import Path

from autofinetune.errors import FatalError, RoundError
from autofinetune.schemas import RoundPlan
from autofinetune.trainer.base import TrainResult


class TRLTrainerBackend:
    def train(
        self,
        base_model_id: str,
        train_jsonl: Path,
        output_dir: Path,
        plan: RoundPlan,
    ) -> TrainResult:
        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer
        except ImportError as e:
            raise FatalError(
                "TRL backend requires extras: pip install 'autofinetune[train]'"
            ) from e

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            ds = load_dataset("json", data_files=str(train_jsonl), split="train")

            def to_text(example):
                return {
                    "text": (
                        f"### Question:\n{example['question']}\n\n"
                        f"### Answer:\n{example['answer']}"
                    )
                }

            ds = ds.map(to_text)
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            lora = LoraConfig(
                r=plan.lora.r,
                lora_alpha=plan.lora.alpha,
                lora_dropout=plan.lora.dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            args = SFTConfig(
                output_dir=str(output_dir),
                num_train_epochs=plan.lora.epochs,
                learning_rate=plan.lora.learning_rate,
                per_device_train_batch_size=plan.lora.per_device_train_batch_size,
                gradient_accumulation_steps=plan.lora.gradient_accumulation_steps,
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                max_seq_length=2048,
            )
            trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=ds,
                peft_config=lora,
                processing_class=tokenizer,
            )
            trainer.train()
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            return TrainResult(output_dir=output_dir, backend="trl")
        except FatalError:
            raise
        except Exception as e:  # noqa: BLE001
            raise RoundError(f"TRL training failed: {e}") from e
