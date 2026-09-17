# 本地评测工具包

在**任何机器**（云端 ECS / 本机 Windows PC）上对 docs-seeker 跑批量评估：
不依赖 core 服务，只要仓库代码 + `.env` 即可。

## 目录

```
scripts/eval/
├── run_local_baseline.py   # 跑一轮：逐题问答 + 子串判分，结果落 JSON
├── run_agent_eval.py       # 跑一轮（AgentRunner / Agentic M1 路径）+ agent 专属指标
├── probe_latency_breakdown.py  # 单题延迟分解（按 LLM/检索调用点；必先预热检索）
├── probe_case_evidence.py      # 单题证据链定位（①证据→②成文输入→③答案，逐环找断点）
├── compare_eval_runs.py    # 跨轮对照：逐题分数矩阵 + 摆幅/总分
├── cases/                  # 离线用例（纳入 git，跨机器共享）
│   └── eval-set-v2.json    #   亚马逊卖家侧语料用例（出题后提交于此）
└── README.md
```

跑批产物默认落在仓库根 `eval-runs/`（已 gitignore，不提交）。

## 前置

1. 依赖：`uv sync --extra dev`（Windows）或 `.venv` 已就绪（云端）
2. 仓库根 `.env` 至少包含：

   ```
   MILVUS_URI=...                # 指向同一 Zilliz 集群 → 数据天然共享，不用同步向量
   MILVUS_TOKEN=...
   COLLECTION_NAME=chartermate_docs_insightforge
   SUMMARY_COLLECTION_NAME=chartermate_summaries_insightforge
   DEEPSEEK_API_KEY=...
   DASHSCOPE_API_KEY=...
   LLM_TEMPERATURE=0
   LLM_DECOMPOSE_TEMPERATURE=0
   LLM_GENERATE_MODEL=deepseek-chat
   SEMANTIC_CACHE_ENABLED=false
   LANGFUSE_TRACING_ENABLED=false
   ```

## 跑一轮

```bash
# Git Bash / Linux / macOS
PYTHONPATH=src LLM_TEMPERATURE=0 LLM_DECOMPOSE_TEMPERATURE=0 \
  .venv/bin/python scripts/eval/run_local_baseline.py \
  --cases-file scripts/eval/cases/eval-set-v2.json --label local-a

# Windows PowerShell
$env:PYTHONPATH="src"; $env:LLM_TEMPERATURE="0"; $env:LLM_DECOMPOSE_TEMPERATURE="0"
uv run python scripts/eval/run_local_baseline.py `
  --cases-file scripts/eval/cases/eval-set-v2.json --label local-a
```

云端习惯把结果放到仓库外：追加 `--out-dir /opt/datapilot-backups/eval-runs`。

不传 `--cases-file` 时回退读 core API（`127.0.0.1:3002`，仅云端）。

## 跑 AgentRunner（Agentic M1）一轮

```bash
LANGFUSE_TRACING_ENABLED=false .venv/bin/python scripts/eval/run_agent_eval.py \
  --cases-file scripts/eval/cases/eval-set-v2.json --label agent-m1 --workers 2
```

与基线的唯一区别是「谁回答问题」：本脚本**直接驱动 `AgentRunner`**（ReAct 循环 + 检索工具 +
两段式充分性裁决），不经过 `ChatService`（因此不受 `AGENT_ENABLED` 影响，也不需要起服务）。
判分**复用 `run_local_baseline.py` 的同一套实现**，结果 JSON 同构、可直接喂
`compare_eval_runs.py` 做跨管线对照；此外多记：

- `agent_steps` / `agent_actions` / `agent_parse_errors`：循环步数与动作轨迹
- `agent_sufficient`：agent 自己的充分性终态（与判分的拒答启发式对照看）
- `agent_stats`：平均步数、P50/P95 耗时、空答案数、异常题数
- `params` 额外写死 agent 口径（`agent_max_steps` / `agent_judge_max_tokens` /
  `agent_evidence_char_budget`），可用 `--max-steps` 临时覆盖步数上限

⚠️ agent 每题 LLM 调用次数是单轮的多倍（决策 + 成文 + 可能的核实），`--workers` 建议 2 起，
别按单轮的并发开。

## 跨轮对照

```bash
.venv/bin/python scripts/eval/compare_eval_runs.py eval-runs/a-*.json eval-runs/b-*.json
```

判分口径复刻 core：关键词**纯子串**命中率；`expected_chapter` 不匹配 ×0.7；
空答案 0 分；`score ≥ 0.8` 算通过。

**拒答题（本工具包对 core 判分的扩展）**：用例带 `"expected_answer_type": "abstain"` 时，
系统**空答案或明确表示语料不足**（"未提及/没有相关/无法回答…"等 30+ 措辞）→ 1.0；
硬编答案 → 0.0（结果里标 `HALLUCINATED`）。汇总行单独输出「拒答正确率」（Agentic M1 核心指标）。
局限：当前只认措辞，不认"先否认再硬编"——M1 充分性判断上线后收紧。

## 用例文件格式

```json
[
  {"case_id": "T001", "question": "...", "expected_keywords": ["原文连续短语"],
   "expected_chapter": "第三章", "category": "事实查询"},
  {"case_id": "T027", "question": "语料里没有答案的问题...", "expected_keywords": [],
   "expected_chapter": null, "category": "拒答", "expected_answer_type": "abstain"}
]
```

- 关键词必须逐词是**原文连续子串**、≥2 字（T008 单字关键词教训）；出题后用原文检索逐题核对
- `eval-set-v2.json`：亚马逊卖家侧语料（`chartermate_docs_insightforge`），30 题（26 可答 + 4 拒答）

## 纪律（别省）

- **温度必须 0**：否则同代码同语料单题摆幅可达 0.75，改动读不出来
- **同配置至少 2~3 轮**再下结论；单轮总分差 < 1 题（≈4.5pp）不下结论
- 结果 JSON 记录了生效参数（温度/模型/集合名），对照前先核对参数一致
- 会真实调用 Zilliz + DeepSeek + DashScope，消耗少量额度
- 用例文件格式：`[{"case_id","question","expected_keywords":[],"expected_chapter":"第X章"}]`
  （也兼容 core 导出的 `{data:{cases:[...]}}` 结构）
