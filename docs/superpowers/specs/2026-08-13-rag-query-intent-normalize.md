# RAG 用户 Query 意向判断与元数据归一化

**Date:** 2026-08-13  
**Status:** Ready for fine-tune — seed dataset + plan available  
**Target model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)  
**Related product:** AutoFineTune（领域知识 / Instruct LM LoRA 闭环）  
**Dataset:** `datasets/rag_query_intent/`  
**Plan:** `docs/superpowers/plans/2026-08-13-rag-query-intent-finetune.md`

## 问题

在教育内容 RAG 场景中，用户口语查询与文件元数据用词不一致。例如：

| 用户说法 | 库内可能元数据 |
|----------|----------------|
| 初中2年级上册第一单元的总结 | `年级=初二` 或 `年级=八年级`，`册别=上`，`单元=1` |
| 八年级上第一单元总结 | 同上 |
| 初二上册 Unit 1 summary | 同上（中英混用） |

需要系统在检索前完成：

1. **意图（intent）**：总结 / 习题 / 知识点 / 其他  
2. **槽位（slots）**：学段、年级、册别、单元、学科等  
3. **归一化（normalize）**：多种说法映射到**统一枚举值**，再用归一化字段过滤/检索文档

这不是「背课文知识」为主，而是 **结构化理解 + 词表对齐**。

## 结论：1.5B Instruct 是否够用

**够用。** Qwen2.5-1.5B-Instruct 是 Causal Instruct LM，适合作为 RAG 前置的轻量意图/槽位模型；也在 AutoFineTune 默认 allowlist 中（Mac / 低显存友好）。

适合：

- 封闭或半封闭的教材元数据枚举  
- 输出固定 JSON  
- 本机 LoRA 微调强化别称对齐  

不适合单独指望模型：

- 元数据本身混乱、无统一枚举  
- 库中不存在的书/单元（微调无法凭空召回）

## 推荐落地路径（由浅到深）

| 阶段 | 方案 | 做法 | 何时用 |
|------|------|------|--------|
| A | 规则 + 同义词表 | `初二=八年级=初中二年级` 等字典 + 正则抽「上册 / 第 N 单元」 | 说法相对封闭、要先上线 |
| B | 零样本 / 少样本 LLM | 给出合法枚举，强制 JSON 输出 | 说法多、先验证效果 |
| C | AutoFineTune LoRA | 用 `(query → 标准元数据 JSON)` 微调 1.5B | B 不够稳、线上说法杂 |

**建议顺序：A 打底 + B 试效果 → 不够再 C（AutoFineTune）。**

## 目标 JSON Schema（草案）

模型输出应约束为固定结构，**枚举字段只能从合法集合中选**：

```json
{
  "intent": "summary",
  "grade": "八年级",
  "volume": "上",
  "unit": 1,
  "subject": null,
  "confidence": 0.86,
  "raw_spans": {
    "grade_mention": "初中2年级",
    "volume_mention": "上册",
    "unit_mention": "第一单元"
  }
}
```

字段说明（可按业务增删）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `intent` | enum | 如 `summary` / `exercises` / `knowledge` / `unknown` |
| `grade` | enum \| null | **归一化后的年级**（如统一用「八年级」，不要自由生成「二年级上」） |
| `volume` | enum \| null | `上` / `下` / `全` 等 |
| `unit` | int \| null | 单元号 |
| `subject` | enum \| null | 语文 / 数学 / … |
| `confidence` | float | 0–1，低置信度可走澄清或规则兜底 |
| `raw_spans` | object | 可选，便于调试与规则复核 |

### 年级映射表示例（需产品定稿）

写入 system prompt 与/或规则表，微调数据也要一致：

| 用户/口语 | 标准 `grade` |
|-----------|----------------|
| 初中一年级 / 初一 / 七年级 | 七年级 |
| 初中二年级 / 初二 / 八年级 / 初中2年级 | 八年级 |
| 初中三年级 / 初三 / 九年级 | 九年级 |
| （小学、高中等同理扩展） | … |

**原则：** RAG 检索与过滤一律使用标准 `grade`，不要用原始 query 字符串去撞文件名。

## 少样本 Prompt 要点（阶段 B）

System 中应包含：

1. 任务：从用户 query 抽取意图与槽位，并归一化到给定枚举  
2. 完整枚举列表（grade / volume / intent / subject）  
3. 明确映射表（如上）  
4. 「无法确定则该字段为 null，intent 可为 unknown」  
5. 「只输出 JSON，不要解释」

User 示例：

- 输入：`初中2年级上册第一单元的总结`  
- 输出：`grade=八年级, volume=上, unit=1, intent=summary`

## 与 AutoFineTune 的关系（阶段 C）

AutoFineTune 定位：领域知识 Instruct LM 的闭环 LoRA（plan → 数据 → 训练 → 评测 → 决定）。

对本任务：

| 项 | 建议 |
|----|------|
| 基座 | `Qwen/Qwen2.5-1.5B-Instruct`（`--base-model` pin 或 allowlist auto） |
| 后端 | Mac：`trainer.backend=auto` → MLX；NVIDIA：TRL |
| 数据形态 | 转为 AutoFineTune 可消费的 QA / jsonl：`question=用户query`，`answer=目标 JSON 字符串` |
| 评测 | 主指标用 **字段级准确率**（grade/volume/unit/intent exact match）；LLM-as-judge 仅作辅 |
| 编排器 | DeepSeek 等可继续做轮次规划；judge schema 可后续特化为「JSON 字段是否正确」 |

### 训练数据格式（对接 AutoFineTune 的草案）

`qa.jsonl` 每行：

```json
{"question": "初中2年级上册第一单元的总结", "answer": "{\"intent\":\"summary\",\"grade\":\"八年级\",\"volume\":\"上\",\"unit\":1,\"subject\":null,\"confidence\":1.0}"}
```

配套 `brief.md` 可写：教育 RAG 前置解析器；只输出标准 JSON；年级必须归一化到枚举。

数据来源建议：

1. 线上真实 query + 人工标注标准 JSON  
2. 用模板从枚举笛卡尔积生成（「{年级说法}{册}第{N}单元的总结」）再混入真实说法  
3. 难例：缺槽位、冲突说法、中英混用、错别字

规模直觉（可在计划里细化）：先 **数百～几千** 条高质量标注验证增益，再扩模板合成。

### 评测集建议

Holdout 按「映射类型」分层，至少覆盖：

- 标准说法（八年级上册第一单元）  
- 别称（初二 / 初中2年级）  
- 缺字段（只要「第一单元总结」）  
- 干扰（超纲年级、库外书名）

主指标：

- `grade` exact match  
- `volume` / `unit` / `intent` exact match  
- 全字段 exact match（严格）  
- 低置信度时「拒识」是否合理（可选）

## 明确非目标（本笔记范围）

- 不替代全文 RAG 检索与重排序  
- 不微调 ACE-Step / ASR / TTS 等非文本 Instruct 栈  
- 不在本文件中展开完整 AutoFineTune 实现计划（见下方「后续」）

## 后续：用 AutoFineTune 做微调计划

已完成：

- 实现计划：`docs/superpowers/plans/2026-08-13-rag-query-intent-finetune.md`
- 种子数据：`datasets/rag_query_intent/`（`qa.jsonl` 训练 + `holdout.jsonl` 固定评测）

执行计划中的 Task 2（字段评测脚本）与 Task 3–4（fake / 真 LoRA 开跑）即可开始微调。

## 参考

- 模型卡：https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct  
- AutoFineTune 设计：`docs/superpowers/specs/2026-08-12-autofinetune-design.md`  
- MLX / 小模型训练：`docs/superpowers/specs/2026-08-13-auto-trainer-mlx-design.md`
