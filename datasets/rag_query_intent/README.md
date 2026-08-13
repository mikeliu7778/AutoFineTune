# RAG Query Intent 数据集（AutoFineTune 输入）

供 AutoFineTune 微调 [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 做教育 RAG 前置解析。

## 布局

```text
datasets/rag_query_intent/
  brief.md          # 领域说明 + 枚举/映射（给编排与合成用）
  qa.jsonl          # 训练集（question=用户query, answer=标准JSON字符串）
  holdout.jsonl     # 固定离线评测集（勿并入训练）
  enums.json        # 枚举与条数元信息
```

## 快速开跑

```bash
# 需要 DeepSeek 编排时，从环境变量读 key：
export DEEPSEEK_API_KEY=sk-...

# 推荐：用仓库脚本（自动检查 DEEPSEEK_API_KEY）
python scripts/run_rag_intent_finetune.py

# 仅冒烟（假 LLM + 假 trainer，不需要 key / GPU）：
python scripts/run_rag_intent_finetune.py --fake-llm --fake-trainer

# 等价手写命令：
pip install -e '.[mlx]'   # Mac；CUDA 用 '.[train]'
autofinetune run ./datasets/rag_query_intent \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --runs-dir ./runs \
  --trainer auto
```

字段级准确率请用固定 holdout + 评测脚本：

```bash
# 用已完成的 run 直接推理并打分（推荐）
python scripts/eval_rag_intent_holdout.py \
  --gold datasets/rag_query_intent/holdout.jsonl \
  --run runs/20260813-083721-22533979 \
  --out-metrics runs/20260813-083721-22533979/rag_intent_holdout_metrics.json \
  --out-pred runs/20260813-083721-22533979/rag_intent_holdout_preds.jsonl \
  --out-details runs/20260813-083721-22533979/rag_intent_holdout_details.jsonl

# 或只对已有预测文件打分；线上建议加 --validate（非法枚举钳制）
python scripts/eval_rag_intent_holdout.py \
  --gold datasets/rag_query_intent/holdout.jsonl \
  --pred path/to/preds.jsonl \
  --validate
```

服务侧可用 `autofinetune.eval.rag_intent_validate.parse_and_validate` 对模型输出做枚举钳制（`weather`→`unknown`、非法 grade→`null` 等）。

（AutoFineTune 内置 LLM-judge 仅作辅助，不等于字段 exact match。）

## 数据侧重

训练集刻意加强：空槽不编造、OOD→`unknown`、考点/重点→`knowledge`、高中/小学 `grade=null`。holdout 含同类硬例，勿并入训练。

## 统计

见 `enums.json` 中的 `train_count` / `holdout_count`（当前约 2.4k 训练 / 56 holdout，刻意加重空槽与 OOD）。
