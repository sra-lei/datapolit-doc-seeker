# 本地评测工具包

在**任何机器**（云端 ECS / 本机 Windows PC）上对 docs-seeker 跑批量评估：
不依赖 core 服务，只要仓库代码 + `.env` 即可。

## 目录

```
scripts/eval/
├── run_local_baseline.py   # 跑一轮：逐题问答 + 子串判分，结果落 JSON
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

## 跨轮对照

```bash
.venv/bin/python scripts/eval/compare_eval_runs.py eval-runs/a-*.json eval-runs/b-*.json
```

判分口径复刻 core：关键词**纯子串**命中率；`expected_chapter` 不匹配 ×0.7；
空答案 0 分；`score ≥ 0.8` 算通过。

## 纪律（别省）

- **温度必须 0**：否则同代码同语料单题摆幅可达 0.75，改动读不出来
- **同配置至少 2~3 轮**再下结论；单轮总分差 < 1 题（≈4.5pp）不下结论
- 结果 JSON 记录了生效参数（温度/模型/集合名），对照前先核对参数一致
- 会真实调用 Zilliz + DeepSeek + DashScope，消耗少量额度
- 用例文件格式：`[{"case_id","question","expected_keywords":[],"expected_chapter":"第X章"}]`
  （也兼容 core 导出的 `{data:{cases:[...]}}` 结构）
