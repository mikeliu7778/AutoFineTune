#!/usr/bin/env bash
# 重新跑 RAG query-intent 微调 + 固定 holdout 字段评测
#
# 用法:
#   ./scripts/rerun_rag_intent.sh
#   DEEPSEEK_API_KEY=sk-... ./scripts/rerun_rag_intent.sh   # 有 key 时走真实编排
#   ./scripts/rerun_rag_intent.sh --fake-llm                 # 显式假编排（仍真实 MLX 训练）
#   ./scripts/rerun_rag_intent.sh --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "==> python: $PYTHON"
echo "==> cwd:    $ROOT"

# 1) 微调（无 DEEPSEEK_API_KEY 时脚本会自动 --fake-llm）
echo "==> fine-tune"
"$PYTHON" scripts/run_rag_intent_finetune.py "$@"

# dry-run 不评测
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    echo "==> dry-run done"
    exit 0
  fi
done

# 取最新 run 目录
RUN_DIR="$(ls -1dt runs/*/run.json 2>/dev/null | head -1 | xargs -I{} dirname {})"
if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "error: no run found under ./runs" >&2
  exit 1
fi

echo "==> eval holdout on $RUN_DIR"
"$PYTHON" scripts/eval_rag_intent_holdout.py \
  --gold datasets/rag_query_intent/holdout.jsonl \
  --run "$RUN_DIR" \
  --validate \
  --out-metrics "$RUN_DIR/rag_intent_holdout_metrics.json" \
  --out-pred "$RUN_DIR/rag_intent_holdout_preds.jsonl" \
  --out-details "$RUN_DIR/rag_intent_holdout_details.jsonl"

echo "==> done: $RUN_DIR"
