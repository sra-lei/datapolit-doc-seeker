# docs-seeker 项目现状与修复计划

> 本文档基于当前工作区盘点生成，用于后续照单修复。
> 盘点基线：git commit `a038f84`（initial commit）+ 工作区未提交改动（仅 `README.md`）。
> 文档约定：每项问题带【位置】【问题】【影响】【修复建议】【验收标准】；修复路线图用勾选框跟踪。

---

## 1. 项目概述

在线检索与问答微服务（RAG）：

- **检索**：Milvus（只读，与 doc-kit 共享）语义检索 + 本地 jieba/BM25 稀疏检索 + 摘要引导检索，三路 RRF 融合
- **生成**：DeepSeek LLM 答案生成（带重试/熔断/降级网关）
- **向量化**：阿里百炼 text-embedding-v4（OpenAI 兼容协议）
- **缓存**：Redis Stack 向量语义缓存（按问题相似度命中）
- **安全**：输入提示注入检测 + 输出敏感信息脱敏（guard.py）
- **技术栈**：Python 3.10+ / FastAPI / pydantic v2 / pymilvus / redis / jieba / openai / loguru / langfuse(链路追踪已接入，见 2.5)

**数据流**：`POST /v1/chat` → 注入检测 → 语义缓存查询 → LLM 查询分解 → 三路检索(RRF 融合) → LLM 生成 → 脱敏 → 写缓存

---

## 2. 当前工程状态

### 2.0 架构调整记录（本次已完成）

- 目录结构调整为 README 目标架构（见 2.1）：config 拆分 `settings.py` + `prompts.yaml` + `retrieval.yaml`；domain 只保留 `entities/` 与 `interfaces/`；检索实现移至 `retrieval/`；新增 `application/services` + `pipelines`；infra 按 `vector_store/cache/llm/embedding/security` 分子目录；api 拆 `routes/v1` + `schemas/{request,response}` + `middleware`；新增 `infra/observability/logger.py`、`infra/observability/metrics.py`
- 行为保持不变的机械搬移：dense/bm25/summary/composite/decomposer/hybrid_router/generator、milvus/embedder/semantic_cache/llm_gateway/guard
- 检索结果由 dict 改为领域实体 `Chunk/Document/Query`（dict → 实体在检索器出口完成，行为等价）
- 配置外部化落地：generator/query_decomposer 的 Prompt 从 `prompts.yaml` 读取（缺失时回退代码内默认值）；RRF 权重/k/fetch 参数从 `retrieval.yaml` 读取
- 新增：`GET /metrics`（Prometheus，prometheus-client）；请求日志中间件（request_id + 耗时 + 指标）；`infra/cache/redis_client.py` 单例
- 依赖新增：`pyyaml`、`prometheus-client`（pyproject.toml 与 requirements.txt 同步）；`package-data` 补 `config/*.yaml`（wheel 打包不丢 yaml 配置）
- 依赖安装（uv）：创建 `.venv`（CPython 3.12.12）+ 全量依赖及 dev extras 安装完成（含 `pydantic-settings==2.15.0`、`python-dotenv==1.2.3`，P1-1/P1-2 顺带解决）；冒烟通过：`import docs_seeker.api.main` OK、`GET /`、`GET /metrics` 200、`POST /v1/chat`（注入拦截）正常、中间件 request_id 日志正常
  - 注：uv 构建 sdist 时需临时目录 chmod 权限，DSH workspace-write 沙箱会拦截（WinError 5），须以完整文件系统权限运行安装；`.uv-cache/` 已加入 .gitignore
- 语义缓存修复（P1-3/P1-4）：`array('f')` 字节编码、`doc["sources"]` 用 `[]` 访问、KNN `dialect=2`、索引维度动态化 + 漂移自愈；新增 `SEMANTIC_CACHE_ENABLED` 环境变量开关（默认开启，`.env.example` 已注释说明）；新增 `tests/test_semantic_cache.py`（5 例全过）；顺手修 P2-9（Settings 改用 `SettingsConfigDict`）
- 前端对接（client 仓库，独立 git）：新增 `GET /v1/stats` 只读指标端点（语义缓存 + LLM 网关统计，供前端看板）；修复 `_require_dim` 空文本探测 bug（百炼拒绝空 input，改用"维度探测"，实测 text-embedding-v4 = 1024 维）
- Docker 运维基线（P3-7）：非 root 用户（app:1000）、Dockerfile/compose 双份 healthcheck（urllib 调 `/v1/health`）、新增 `.dockerignore`（防 .venv/.uv-cache/.env/tests 进镜像）；注：DSH 沙箱无法连接 docker daemon（named pipe 限制），未实际 build，待真机验证
- Milvus 监控看板：新增 `GET /v1/milvus/stats`（集合状态/行数/向量维度/索引）；修复 `MilvusStore.count`（改用 `get_collection_stats`，原 count 在 Milvus 2.6 报 "pagination not allowed"）、`describe_index`（pymilvus 3.x 需 index_name，先 list_indexes 枚举）；前端 Dashboard 新增 Milvus 监控面板（与数据库管理同行）。⚠️ 发现库向量维度 1536 vs embedding 1024 不匹配，见 P2-13
- RAG 使用统计（按用户维度）：中间件埋点（`X-User-ID` + chat/retrieve + 状态码）写 Redis 独立键 `rag:usage:*`（不受语义缓存开关影响，Redis 不可用时静默降级）；新增 `GET /v1/usage/stats`（总次数/成功率/活跃用户/用户 Top）；前端 `docsSeekerFetch` 附带 `X-User-ID`，Dashboard 右侧"数据库配置与统计"替换为 RAG 使用统计面板
- P2-13 修复：embedding 模型统一为 `text-embedding-v2`（与 doc-kit 入库维度 1536 对齐），dense/summary 检索恢复；语义缓存维度自动重探为 1536（漂移自愈）
- 静态审查（后台子代理）结论：重构后导入图自洽、无循环导入、无旧扁平模块残留引用、实体迁移全链路一致；严重项均为既有 P1（P1-3 已细化三重故障诊断，见 3.1）
- 未纳入本次范围（按清单后续修）：P1-1/P1-2 依赖缺失、P1-3 语义缓存 bytes、P1-4 缓存维度、P2 系列等
- 新发现问题：BM25 结果无 id → /chat 去重坍缩，见 P2-12
- 依赖单一来源（P3-3）：删除 `requirements.txt`；Dockerfile 迁移到 uv（`python:3.12-slim` 基础镜像 + 构建期 `pip install uv` 引导工具，依赖安装用 `uv sync --frozen --no-dev`）；注：ghcr.io 官方 uv 镜像国内网络拉取失败，故弃用；未实机 docker build 验证
- 二次结构重构（目标架构对齐，行为零变化）：`app.py` → `api/main.py`（uvicorn 入口 `docs_seeker.api.main:app`）；`config/` + `infra/{observability,security}` → `core/`（config.py/logging.py/metrics.py/security.py，yaml 并入 core/）；`domain/entities/` → `domain/models/`；`application/services` + `application/pipelines` → `domain/services/`（含 rag_pipeline）；`retrieval/` → `infrastructure/retrieval/`；`infra/` → `infrastructure/`（`vector_store/` → `database/`）；`api/routes/v1/` 拍平为 `api/routes/`（`/v1` 前缀保留在聚合处）；所有 import 同步更新；工程化：ruff/mypy 配置 + dev extras（ruff/mypy 入 uv.lock）、路由 async→def（线程池）、新增 guard/composite/bm25/usage 单元测试（总计 27 例）；测试重组为 `tests/unit/` + `tests/integration/` + `conftest.py`

### 2.1 实际目录结构（与 README 一致，重构后）

```
src/docs_seeker/
├── api/                          # 接口层（HTTP 适配）
│   ├── main.py                   # FastAPI 应用实例 + 中间件 + lifespan（uvicorn 入口）
│   ├── deps.py                   # 依赖注入组装点（单例管理）
│   ├── middleware.py             # 请求日志 + 指标中间件
│   ├── routes/                   # 路由（文件拍平，对外保留 /v1 前缀）
│   │   ├── __init__.py           # v1 路由聚合
│   │   ├── chat.py / health.py
│   │   └── stats.py / milvus.py / usage.py
│   └── schemas/                  # request.py / response.py
├── core/                         # 跨模块共享
│   ├── config.py                 # Pydantic Settings + yaml 加载（settings/prompts/retrieval_config）
│   ├── logging.py                # 结构化日志（loguru）
│   ├── metrics.py                # Prometheus 指标
│   ├── security.py               # 安全护栏（输入注入检测 / 输出脱敏）
│   ├── prompts.yaml / retrieval.yaml
├── domain/                       # 核心业务层
│   ├── models/                   # Chunk / Document / Query
│   ├── services/                 # chat_service / generator / rag_pipeline / top_warmup
│   └── interfaces/               # Retriever / EmbeddingProvider / LLMProvider
└── infrastructure/               # 基础设施层（外部依赖实现）
    ├── database/                 # milvus_client.py（只读）
    ├── cache/                    # redis_client.py + semantic_cache.py
    ├── llm/                      # gateway.py（重试/熔断/降级）
    ├── embedding/                # embedder.py（查询向量化）
    ├── retrieval/                # dense/bm25/summary/composite/query_decomposer/hybrid_router
    └── usage/                    # tracker.py（RAG 使用统计）+ __init__.py（兼容导出）
```

> 结论：README 目录树与实际代码已对齐（重构落地）。README 声称但尚未实现的部分仅剩：认证/限流中间件（P3-4）、HybridRouter 接线（P2-3）。

### 2.2 依赖现状

- `pyproject.toml` + `uv.lock`（唯一正源，`requirements.txt` 已删除，镜像构建同样走 uv）：fastapi / uvicorn / loguru / pydantic / pymilvus / redis / jieba / openai / httpx / langfuse；dev extras：pytest、pytest-asyncio
- **缺失**：`pydantic-settings`（config.py 直接 import，未声明 → 全新环境 ImportError）、`python-dotenv`（llm_gateway.py import，仅靠 pydantic-settings 传递安装）
- **本次重构新增**：`pyyaml`（config yaml 加载）、`prometheus-client`（/metrics 端点）
- 本机环境无可用 Python 解释器（`python` 为 WindowsApps 占位符、`py` 不存在），未能做实际 import 验证

### 2.3 配置与环境

- `.env` 已存在且配置完整（Milvus URI/Token、DASHSCOPE_KEY、DEEPSEEK_KEY、REDIS_URL、两个 collection 名），**未被 git 跟踪**（`.gitignore` 正确，`git ls-files` 确认无 .env）✅
- `LANGFUSE_*` 三个变量为空
- `.env.example` 与 `.env` 键位一致

### 2.4 Git 状态

- 仅 1 个 commit（initial commit）
- 工作区有未提交改动：README 的目标架构改写 + 本次完整目录重构（见 2.0），建议作为一次 commit 提交
- `tests/` 目录存在但**完全为空**；无 CI、无 lint/type 配置（.gitignore 里预留了 .mypy_cache/.ruff_cache 但无对应配置）

### 2.5 Langfuse 链路追踪（P3-2 已接入）

遵循 [langfuse/skills](https://github.com/langfuse/skills) 官方 Agent Skill 与[追踪最佳实践](https://langfuse.com/docs/observability/best-practices)实现：

- **接入点**：`infrastructure/tracing.py`（环境变量加载 + `tracing_enabled()` / `shutdown_langfuse()`）
- **LLM 调用**：`infrastructure/llm/gateway.py`、`infrastructure/embedding/embedder.py` 改用 `langfuse.openai.OpenAI` drop-in 包装，自动记录 generation/embedding 观测（模型名、token 用量、耗时、错误）；流式开启 `stream_options.include_usage` 采集 token
- **流程观测**：`ChatService.chat/chat_stream`（根 trace `chat-response`）、`RAGPipeline.prepare`（`retrieve-context`）、三路检索器与 RRF 融合（`retriever` 类型）、语义缓存查询（`retriever` 类型）
- **属性**：`session_id`/`user_id`（ChatRequest 新增可选字段）→ propagate_attributes 传播；tags=`chat`；environment 取 `ENVIRONMENT`；metadata 含路由
- **降级**：未配置 `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` 时客户端自动 no-op，不影响业务；测试环境 `LANGFUSE_TRACING_ENABLED=false`（tests/conftest.py）
- **验证**：已端到端跑通并审计真实 trace（chat 非流式 + 流式各一条），结构符合基线要求（命名/类型/层级/输入输出/用量）

---

## 3. 遗留问题清单

> 路径对应说明：重构后旧路径已迁移 —— `domain/*_retriever.py` → `infrastructure/retrieval/`；`infra/milvus_store.py` → `infrastructure/database/milvus_client.py`；`infra/semantic_cache.py` → `infrastructure/cache/semantic_cache.py`；`infra/llm_gateway.py` → `infrastructure/llm/gateway.py`；`infra/embedder.py` → `infrastructure/embedding/embedder.py`；`infra/guard.py` → `core/security.py`；`infra/usage_tracker.py` → `infrastructure/usage/tracker.py`；`api/routes.py` → `api/routes/*.py`（v1 拍平，前缀保留）；`api/schemas.py` → `api/schemas/{request,response}.py`；`config/config.py` → `core/config.py`；`domain/generator.py` → `domain/services/generator.py`；`application/*` → `domain/services/`

### 3.1 🔴 P1 阻断级（装不上 / 启动失败 / 核心功能失效）

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P1-1 | ~~缺失 `pydantic-settings` 依赖~~ | `pyproject.toml`、`requirements.txt` | ✅ 已解决（uv 安装 `pydantic-settings==2.15.0`，`import docs_seeker.app` 通过） |
| P1-2 | ~~`python-dotenv` 未显式声明~~ | `pyproject.toml`、`requirements.txt` | ✅ 已解决：显式声明 `python-dotenv>=1.0.0`（实际安装 1.2.3） |
| P1-3 | ~~语义缓存三重故障~~ | `infra/cache/semantic_cache.py` | ✅ 已解决：① `array('f', vec).tobytes()` 字节编码；② `doc["sources"]` 改用 `[]` 访问；③ KNN 加 `dialect=2`；另新增 `SEMANTIC_CACHE_ENABLED` 开关（默认开）；`tests/test_semantic_cache.py` 5 例通过 |
| P1-4 | ~~缓存索引维度硬编码 `DIM: 1536`~~ | `infra/cache/semantic_cache.py` | ✅ 已解决：维度改为按真实 embedding 长度动态创建，模型/维度变化时自动重建索引（漂移自愈） |
| P1-5 | ~~README（未提交版）与代码脱节~~ | `README.md` | ✅ 已解决：按 README 架构完成重构（见 2.0），目录树与实际一致，游离 ``` 已删；"提交 git" 待做 |

### 3.2 🟠 P2 功能缺陷 / 潜在 bug

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P2-1 | async 路由内跑同步阻塞代码（Milvus/Redis/LLM/jieba/BM25 建索引） | `api/routes.py:13,29,35` 及各 domain 实现 | 阻塞事件循环，并发吞吐劣化；三路检索串行无 `asyncio.gather` |
| P2-2 | BM25 索引首次 search 才 `build_index()`，之后永不过期 | `domain/bm25_retriever.py:56-57` | 文档更新后检索结果陈旧（README 的"惰性刷新"方案未实现） |
| P2-3 | `HybridRouter` 死代码：deps 提供注入器但无路由使用 | `domain/hybrid_router.py`、`api/deps.py:37-40` | BM25 路由决策从未生效，composite 无条件三路全跑 |
| P2-4 | `MilvusStore.query_by_chapter` 死代码且必崩（传 `query_vector=[]`） | `infra/milvus_store.py:86-107` | 调用即报错；无调用方（summary_retriever 实际用内存后过滤） |
| P2-5 | score 语义不统一：dense 用 `1-distance`(≈0~1)，BM25 无界原始分 | `dense_retriever.py:40` vs `bm25_retriever.py:84`；`generator.py:25` | `_score_avg > 0.3` 的置信度判定被 BM25 高分污染，结果失真 |
| P2-6 | 健康检查 `status` 恒为 `"ok"`，即使 Milvus/Redis 全挂 | `api/routes.py:13-25` | 监控误判；`redis_connected` 依赖缓存模块私有属性 `_available` |
| P2-7 | guard 正则一刀切：`(翻译|translate)` 拒绝所有翻译请求；`(怎么|如何).*(攻击|破解|入侵)` 可能误杀正常问题 | `infra/guard.py:25-31` | 正常用户请求被拒（策略问题，需人工确认口径） |
| P2-8 | CORS `allow_origins=["*"]` + `allow_credentials=True` 非法组合 | `app.py:23` | 浏览器规范不允许，跨域带凭据请求行为异常 |
| P2-9 | ~~pydantic v2 弃用写法 `class Config`~~ | `config/settings.py` | ✅ 已解决：改用 `model_config = SettingsConfigDict(...)` |
| P2-10 | `MilvusStore.search` 异常时静默返回 `[]` | `infra/milvus_store.py:82-84` | 上层无法区分"无结果"与"失败"，chat 拿空上下文仍生成 → 幻觉风险 |
| P2-11 | 每次 chat 都调 LLM 做查询分解，无缓存/无简单问题短路 | `retrieval/query_decomposer.py`、`application/services/chat_service.py` | 每次问答额外 1 次 LLM 调用 + 延迟 + 成本 |
| P2-12 | BM25 检索结果无 id（`get_all_documents` 未取 id 字段）→ /chat 按 id 去重时全部以 "" 归并，BM25 命中几乎全部被去重掉 | `retrieval/bm25_retriever.py` + `application/pipelines/rag_pipeline.py` | 问答链路召回受损；重构时保持原行为，修复方向：`get_all_documents` 补 id 或用内容哈希兜底 |
| P2-13 | ~~向量维度不匹配（库 1536 vs embedding 1024）~~ | `config/settings.py`、`.env.example` | ✅ 已解决：根因 doc-kit 用 `text-embedding-v2`（1536 维）入库，docs-seeker 误配 `text-embedding-v4`（1024 维）导致 dense/summary 检索报 `vector dimension mismatch (6144 vs 4096 bytes)`；已将 embedding 模型统一为 `text-embedding-v2`，实测 `/v1/retrieve` 恢复。⚠️ 服务器部署 .env 需同步改 `EMBEDDING_MODEL=text-embedding-v2`；doc-kit 入库侧模型必须同为 v2（本地 doc-kit/.env 现为 v4，重新入库前需改）；长远可统一 v4 + 重建库 |

### 3.3 🟡 P3 工程化欠账

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P3-1 | 零测试：tests/ 为空，核心纯逻辑（RRF/BM25/guard/脱敏）无保障 | `tests/` | 修复无回归防线 |
| P3-2 | ~~langfuse 声明未使用~~ | `pyproject.toml`、`.env.example` | ✅ 已解决：chat 全链路 Langfuse 追踪已接入（见 2.5），未配置时自动 no-op |
| P3-3 | ~~依赖清单双份维护（pyproject + requirements.txt 内容重复）~~ | `Dockerfile`、`requirements.txt` | ✅ 已解决：删除 `requirements.txt`，Dockerfile 迁移到 uv（`python:3.12-slim` + 构建期 `pip install uv`，依赖安装 `uv sync --frozen`），pyproject.toml + uv.lock 单一正源 |
| P3-4 | 无认证/限流中间件 | api 层 | 请求日志中间件（request_id + 耗时 + 指标）已落地；认证/限流仍未实现 |
| P3-5 | 其他死代码：`Embedder.get_embeddings_batch/reset`、`SemanticCache.clear/stats`、`MilvusStore.count` 均无调用方 | 对应文件 | 清理或接线（如暴露 metrics 端点） |
| P3-6 | BM25 `_tokenize` 过滤长度>1，中文单字 token 被丢弃 | `domain/bm25_retriever.py:26` | 单字查询召回差 |
| P3-7 | ~~Docker：非 root 用户、无 healthcheck~~ | `Dockerfile`、`docker-compose.yml`、`.dockerignore` | ✅ 已解决：非 root（app 用户 uid 1000）+ Dockerfile/compose 双份 healthcheck（urllib 调 /v1/health）+ 新增 `.dockerignore`（排除 .venv/.uv-cache/.env 等进镜像） |

---

## 4. 修复路线图（照单执行）

> 每项完成后勾选，并跑对应验收标准。建议顺序：Phase 0 → 1 → 2。

### Phase 0 — 可运行性修复（P1，做完能装、能启动、缓存可用）

- [x] **P1-1** 在 `pyproject.toml`（及 `requirements.txt`）补 `pydantic-settings>=2.0.0`
  - 验收：✅ 通过（uv 安装 `pydantic-settings==2.15.0`；`import docs_seeker.api.main` 无 ImportError；TestClient 冒烟 OK）
- [x] **P1-2** 显式补 `python-dotenv>=1.0.0`（与 P1-1 同批，实际安装 1.2.3）
- [x] **P1-3** 修复 `infra/cache/semantic_cache.py` 三处：① `array('f', query_embedding).tobytes()` 编码；② `doc["sources"]` 用 `[]` 访问；③ KNN 加 `dialect=2`
  - 验收：✅ 通过（`tests/test_semantic_cache.py` 5 例全过，覆盖编码/开关/dialect/命中/维度漂移）
- [x] **P1-4** 维度动态化：索引按真实 embedding 长度创建（`_require_dim`），维度漂移自动 `dropindex` 重建
  - 验收：✅ 通过（test_dim_drift_rebuilds_index；无需手工 `FT.DROPINDEX`）
- [x] **P1-5** README 对齐（已解决：按方案 B 完成架构重构，README 目录树与实际一致；游离 ``` 已删；"提交 git" 待做）

### Phase 1 — 正确性修复（P2）

- [ ] **P2-1** 阻塞调用改造：三路检索改 `asyncio.gather`（或至少将 BM25 建索引、LLM 调用放入 `run_in_executor`/线程池）
  - 验收：`ab -c 20` 并发 /chat 压测 P99 相比改造前下降；事件循环无阻塞（用 loop 监控或简单计时）
- [ ] **P2-2** BM25 惰性刷新：search 前对比 Milvus `count()` 指纹（记录数+最新 id 哈希），变化才重建索引
  - 验收：向 Milvus 插入一条新文档后再次检索能查到（无需重启）
- [ ] **P2-3** 接线或删除 HybridRouter：在 composite 中按路由结果动态调整三路权重/是否走 BM25；或删除死代码
- [ ] **P2-4** 删除 `query_by_chapter`（或改用 Milvus `query()` + filter 重写）；确认 `count()` 是否保留
- [ ] **P2-5** 统一 score：BM25 输出做归一化（如 min-max 或 rank 映射）后再参与置信度计算；confidence 逻辑单独测试
- [ ] **P2-6** 健康检查：任一下游不可用时 `status` 返回 `degraded`/`unhealthy`；不依赖缓存模块私有属性
- [ ] **P2-7** guard 口径确认：与业务方确认翻译是否应禁止；正则改为更精确（如"破解/越狱/入侵系统"限定词）
- [ ] **P2-8** CORS：显式 origin 列表或去掉 `allow_credentials=True`
- [x] **P2-9** `Settings` 改用 `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`（`class Config` 弃用写法已移除，无 DeprecationWarning）
- [ ] **P2-10** 区分失败与空结果：`search` 失败时抛异常或返回带标记的结果，chat 侧在检索失败时直接报错而非拿空上下文生成
- [ ] **P2-11** 查询分解降本：简单问题（长度<阈值/无连词）短路直接检索；或复用语义缓存
- [ ] **P2-12** BM25 结果补 id：`get_all_documents` 增加 `id` 字段输出（或检索结果用内容哈希兜底 id），修复 /chat 去重坍缩
  - 验收：/chat 响应 sources 中出现 `bm25` 来源的文档

### Phase 2 — 工程化（P3）

- [ ] **P3-1** 补最小测试集：`test_guard.py`（注入/脱敏/误杀回归）、`test_bm25.py`（建索引/检索/刷新）、`test_composite.py`（RRF 融合/去重/权重）、`test_semantic_cache.py`（mock Redis + 修复后的向量编码）；pytest 跑绿
- [x] **P3-2** langfuse 真正接入：chat 全链路 Langfuse 追踪（见 2.5：`infrastructure/tracing.py` + `@observe` 观测 + langfuse.openai 包装 LLM/向量化；未配置 LANGFUSE_* 时自动降级 no-op；测试环境经 `LANGFUSE_TRACING_ENABLED=false` 禁用上报）；已端到端验证并审计真实 trace
- [x] **P3-3** 依赖单一来源：已删 `requirements.txt`，Dockerfile 迁移到 uv（`uv sync --frozen`），pyproject.toml + uv.lock 为唯一正源
- [ ] **P3-4** 补中间件（日志 request_id 已落地；剩余：限流、鉴权如 API Key）
- [ ] **P3-5** 死代码清理或接线（见 3.3 P3-5）
- [ ] **P3-6** BM25 tokenize 策略调优（保留单字或按需）
- [x] **P3-7** Docker：非 root 用户（app:1000）+ healthcheck（Dockerfile 与 compose 双份，urllib 调 /v1/health，interval 30s / timeout 5s / start_period 30s）+ `.dockerignore`
  - 验收：待真机 `docker compose build` 后 `docker compose up`，`docker ps` 显示 healthy（沙箱无法连接 docker daemon，未实机验证）

---

## 5. 回归与验收基线

- 启动冒烟：`uvicorn docs_seeker.api.main:app` 启动无异常日志；`GET /`、`GET /v1/health`、`GET /metrics` 返回 200
- 链路冒烟：`POST /v1/chat` 返回 answer + sources
- 缓存验证：同一问题连续问两次，第二次响应 `cached=true`
- 每次 Phase 完成后全量跑 pytest + 上述冒烟，再进入下一 Phase

---

## 6. P4 候选需求：热门问题 Top10（ChatWidget 欢迎语 + 预热省 token）

> 状态：**已实施完成**（方案见下，实施清单全部勾选）

### 6.1 目标

- **主用途**：ChatWidget 打开时展示 Top10 热门问题作为欢迎语快捷按钮，用户点击直接提问
- **附加价值**：对 Top10 问题主动预热语义缓存，减少高频问题重复检索 + LLM 生成的 token 花费

### 6.2 已确认的决策

| 决策点 | 结论 |
|---|---|
| 归并粒度 | 问题文本**精确匹配**（归一化：strip + 压缩空白 + 小写）；匹配不到的走语义缓存兜底（已有，阈值 0.92） |
| 存储 | **Redis**（不引入 SQLite）——与现有 usage 统计同体系（`rag:usage:top` ZSet） |
| 隐私 | 不做脱敏（都是针对文档的提问，无个人问题） |
| 记录范围 | chat 请求的问题文本（归一化后 ≤200 字符，过短/空跳过） |

### 6.3 方案设计

**① 记录层（`infra/usage/` 扩展）**
- `record()` 增加 `question` 参数（chat_service 传入）；归一化后 `ZINCRBY rag:usage:top 1 <问题>`
- Redis 不可用时降级跳过（与现 usage 一致）

**② Top 查询**
- 新端点 `GET /v1/usage/top?limit=10`（或并入 `/v1/usage/stats` 加 `top` 字段，待定）
- 返回 `[{question, count, cached}]`，`cached` = 语义缓存 search 命中判定（供预热器与前端标记）

**③ 预热器（省 token 核心，`infra/warmup.py`）**
- 后台线程 + 定时（默认每 6h）：对 Top10 中 `cached=false` 的问题，**复用 `chat_service.pipeline.run` + `cache.store`** 跑一遍写缓存
- Redis 锁防多实例重复预热；Top 列表变化（hash 对比）才重预热
- 预热失败不阻塞（try/except + 日志）
- 配置：`TOP_WARMUP_ENABLED`（默认 true）、`TOP_WARMUP_INTERVAL`、`TOP_WARMUP_SIZE`（默认 10）

**④ 前端（ChatWidget 欢迎语）**
- ChatWidget 打开且消息为空时拉取 `/v1/usage/top` → 展示 Top 问题快捷按钮
- 点击按钮 → 复用 `handleSend` 逻辑直接提问（该问题大概率已预热命中缓存）

### 6.4 效果与成本

| 场景 | 现状（仅被动缓存） | 加预热后 |
|---|---|---|
| top 问题首次被问 | 全量检索 + LLM | 预热后直接命中缓存（0 LLM） |
| top 问题重复问 | 命中缓存 | 命中缓存 |
| 预热成本 | — | 每周期 ≤10 次 LLM 调用 |

### 6.5 实施清单（照单执行）

- [x] `usage_tracker.record` 增加 `question` 参数 + ZINCRBY 记录（归一化 + 长度过滤）——实际为独立 `record_question()`（chat_service 调用，middleware 记录不动）
- [x] `usage_tracker.top_questions(n)` 聚合（含 `cached` 标记）
- [x] `GET /v1/usage/top` 端点 + schema（`limit` 1~50，默认 10）
- [x] `infra`→`application/services/top_warmup.py` 预热器（后台线程 + Redis 锁 + 配置开关；直接走 pipeline + cache.store，不重复计数）
- [x] 配置项：`TOP_WARMUP_ENABLED` / `TOP_WARMUP_INTERVAL_HOURS` / `TOP_WARMUP_SIZE`
- [x] ChatWidget 欢迎语：打开时拉取 Top6 快捷按钮（点击即问，handleSend 参数化）
- [x] 验证：pytest 通过、/v1/usage/top 200（Redis 降级返回空）、tsc 通过
