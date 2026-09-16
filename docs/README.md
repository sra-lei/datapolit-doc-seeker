# 设计文档

| 文档 | 内容 |
|---|---|
| [agentic-rag-upgrade.md](agentic-rag-upgrade.md) | 上位决策（2026-08-27 定稿）：路径 A = docs-seeker 从单轮检索升级为 Agent 循环；M1/M2/M3 分层 |
| [agentic-rag-m1-plan.md](agentic-rag-m1-plan.md) | M1 启动方案与全程实录：目标/验收定义、确定性缺陷修复、评估降噪、换语料、优化总表（§11/§12/§13） |

配套实现与评估资产：

- 评估跑批：`scripts/eval/`（`README.md` 含用法与纪律）
- 评估集：`scripts/eval/cases/eval-set-v2.json`（30 题，2026-09-16，基线 29/30、拒答 4/4）
