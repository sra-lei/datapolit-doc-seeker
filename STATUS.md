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
- **技术栈**：Python 3.10+ / FastAPI / pydantic v2 / pymilvus / redis / jieba / openai / loguru / langfuse(未用)

**数据流**：`POST /v1/chat` → 注入检测 → 语义缓存查询 → LLM 查询分解 → 三路检索(RRF 融合) → LLM 生成 → 脱敏 → 写缓存

---

## 2. 当前工程状态

### 2.0 架构调整记录（本次已完成）

- 目录结构调整为 README 目标架构（见 2.1）：config 拆分 `settings.py` + `prompts.yaml` + `retrieval.yaml`；domain 只保留 `entities/` 与 `interfaces/`；检索实现移至 `retrieval/`；新增 `application/services` + `pipelines`；infra 按 `vector_store/cache/llm/embedding/security` 分子目录；api 拆 `routes/v1` + `schemas/{request,response}` + `middleware`；新增 `utils/logger.py`、`utils/metrics.py`
- 行为保持不变的机械搬移：dense/bm25/summary/composite/decomposer/hybrid_router/generator、milvus/embedder/semantic_cache/llm_gateway/guard
- 检索结果由 dict 改为领域实体 `Chunk/Document/Query`（dict → 实体在检索器出口完成，行为等价）
- 配置外部化落地：generator/query_decomposer 的 Prompt 从 `prompts.yaml` 读取（缺失时回退代码内默认值）；RRF 权重/k/fetch 参数从 `retrieval.yaml` 读取
- 新增：`GET /metrics`（Prometheus，prometheus-client）；请求日志中间件（request_id + 耗时 + 指标）；`infra/cache/redis_client.py` 单例
- 依赖新增：`pyyaml`、`prometheus-client`（pyproject.toml 与 requirements.txt 同步）；`package-data` 补 `config/*.yaml`（wheel 打包不丢 yaml 配置）
- 依赖安装（uv）：创建 `.venv`（CPython 3.12.12）+ 全量依赖及 dev extras 安装完成（含 `pydantic-settings==2.15.0`、`python-dotenv==1.2.3`，P1-1/P1-2 顺带解决）；冒烟通过：`import docs_seeker.app` OK、`GET /`、`GET /metrics` 200、`POST /v1/chat`（注入拦截）正常、中间件 request_id 日志正常
  - 注：uv 构建 sdist 时需临时目录 chmod 权限，DSH workspace-write 沙箱会拦截（WinError 5），须以完整文件系统权限运行安装；`.uv-cache/` 已加入 .gitignore
- 语义缓存修复（P1-3/P1-4）：`array('f')` 字节编码、`doc["sources"]` 用 `[]` 访问、KNN `dialect=2`、索引维度动态化 + 漂移自愈；新增 `SEMANTIC_CACHE_ENABLED` 环境变量开关（默认开启，`.env.example` 已注释说明）；新增 `tests/test_semantic_cache.py`（5 例全过）；顺手修 P2-9（Settings 改用 `SettingsConfigDict`）
- 前端对接（client 仓库，独立 git）：新增 `GET /v1/stats` 只读指标端点（语义缓存 + LLM 网关统计，供前端看板）；修复 `_require_dim` 空文本探测 bug（百炼拒绝空 input，改用"维度探测"，实测 text-embedding-v4 = 1024 维）
- Docker 运维基线（P3-7）：非 root 用户（app:1000）、Dockerfile/compose 双份 healthcheck（urllib 调 `/v1/health`）、新增 `.dockerignore`（防 .venv/.uv-cache/.env/tests 进镜像）；注：DSH 沙箱无法连接 docker daemon（named pipe 限制），未实际 build，待真机验证
- Milvus 监控看板：新增 `GET /v1/milvus/stats`（集合状态/行数/向量维度/索引）；修复 `MilvusStore.count`（改用 `get_collection_stats`，原 count 在 Milvus 2.6 报 "pagination not allowed"）、`describe_index`（pymilvus 3.x 需 index_name，先 list_indexes 枚举）；前端 Dashboard 新增 Milvus 监控面板（与数据库管理同行）。⚠️ 发现库向量维度 1536 vs embedding 1024 不匹配，见 P2-13
- RAG 使用统计（按用户维度）：中间件埋点（`X-User-ID` + chat/retrieve + 状态码）写 Redis 独立键 `rag:usage:*`（不受语义缓存开关影响，Redis 不可用时静默降级）；新增 `GET /v1/usage/stats`（总次数/成功率/活跃用户/用户 Top）；前端 `docsSeekerFetch` 附带 `X-User-ID`，Dashboard 右侧"数据库配置与统计"替换为 RAG 使用统计面板
- 静态审查（后台子代理）结论：重构后导入图自洽、无循环导入、无旧扁平模块残留引用、实体迁移全链路一致；严重项均为既有 P1（P1-3 已细化三重故障诊断，见 3.1）
- 未纳入本次范围（按清单后续修）：P1-1/P1-2 依赖缺失、P1-3 语义缓存 bytes、P1-4 缓存维度、P2 系列等
- 新发现问题：BM25 结果无 id → /chat 去重坍缩，见 P2-12

### 2.1 实际目录结构（与 README 一致，重构后）

```
src/docs_seeker/
├── app.py                      # FastAPI 入口（lifespan/CORS/root，含 GET /metrics）
├── config/                     # 配置管理
│   ├── __init__.py             # 导出 settings / prompts / retrieval_config（yaml 加载）
│   ├── settings.py             # Pydantic Settings（环境变量）
│   ├── prompts.yaml            # Prompt 模板（generator / query_decomposer）
│   └── retrieval.yaml          # 检索策略（RRF 权重/k、fetch 参数）
├── domain/                     # 领域层（核心实体 + 接口定义）
│   ├── entities/               # Chunk / Document / Query
│   └── interfaces/             # Retriever / EmbeddingProvider / LLMProvider
├── retrieval/                  # 检索策略实现（原 domain/ 迁移）
│   ├── dense_retriever.py      # 语义检索（Milvus）
│   ├── bm25_retriever.py       # BM25 稀疏检索
│   ├── summary_retriever.py    # 摘要引导检索
│   ├── composite_retriever.py  # 多路融合（RRF）
│   ├── query_decomposer.py     # 查询分解
│   └── hybrid_router.py        # BM25 路由（尚未接线）
├── application/                # 应用层（业务编排）
│   ├── services/               # chat_service / search_service / generator
│   └── pipelines/              # rag_pipeline.py（RAG 完整流程）
├── infra/                      # 基础设施层（外部依赖实现）
│   ├── vector_store/           # milvus_client.py（只读）
│   ├── cache/                  # redis_client.py + semantic_cache.py
│   ├── llm/                    # gateway.py（重试/熔断/降级）
│   ├── embedding/              # embedder.py（查询向量化）
│   └── security/               # guard.py（输入/输出护栏）
├── api/                        # HTTP 层
│   ├── routes/v1/              # chat.py / retrieve.py / health.py
│   ├── schemas/                # request.py / response.py
│   ├── deps.py                 # 单例依赖注入
│   └── middleware.py           # 请求日志 + 指标中间件
└── utils/                      # logger.py（结构化日志）/ metrics.py（Prometheus）
```

> 结论：README 目录树与实际代码已对齐（重构落地）。README 声称但尚未实现的部分仅剩：认证/限流中间件（P3-4）、HybridRouter 接线（P2-3）。

### 2.2 依赖现状

- `pyproject.toml`（唯一正源，另有重复的 `requirements.txt`）：fastapi / uvicorn / loguru / pydantic / pymilvus / redis / jieba / openai / httpx / langfuse；dev extras：pytest、pytest-asyncio
- **缺失**：`pydantic-settings`（config.py 直接 import，未声明 → 全新环境 ImportError）、`python-dotenv`（llm_gateway.py import，仅靠 pydantic-settings 传递安装）
- **多余**：`langfuse`（声明且 .env.example 有变量，代码零引用）
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

---

## 3. 遗留问题清单

> 路径对应说明：重构后旧路径已迁移 —— `domain/*_retriever.py` → `retrieval/`；`infra/milvus_store.py` → `infra/vector_store/milvus_client.py`；`infra/semantic_cache.py` → `infra/cache/semantic_cache.py`；`infra/llm_gateway.py` → `infra/llm/gateway.py`；`infra/embedder.py` → `infra/embedding/embedder.py`；`infra/guard.py` → `infra/security/guard.py`；`api/routes.py` → `api/routes/v1/*.py`；`api/schemas.py` → `api/schemas/{request,response}.py`；`config/config.py` → `config/settings.py`；`domain/generator.py` → `application/services/generator.py`

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
| P2-13 | 🔴 向量维度不匹配：Milvus 库中向量字段 dim=1536（doc-kit 入库），而 docs-seeker embedder 输出 1024（text-embedding-v4 默认）→ dense/summary 检索向量维度不符，检索失败或结果异常 | `infra/embedding/embedder.py` | 检索质量根因级问题（已实测确认：describe_collection dim=1536 vs embedding len=1024）；修复方向：embedder 调用加 `dimensions=1536`（需实测百炼兼容），或 doc-kit 重新以 1024 维入库 |

### 3.3 🟡 P3 工程化欠账

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| P3-1 | 零测试：tests/ 为空，核心纯逻辑（RRF/BM25/guard/脱敏）无保障 | `tests/` | 修复无回归防线 |
| P3-2 | langfuse 声明未使用 | `pyproject.toml`、`.env.example` | 依赖冗余；`utils/logger.py`、`utils/metrics.py` 与 `GET /metrics` 已随重构落地（部分解决） |
| P3-3 | 依赖清单双份维护（pyproject + requirements.txt 内容重复） | 两份文件 | 改一处漏一处的风险 |
| P3-4 | 无认证/限流中间件 | api 层 | 请求日志中间件（request_id + 耗时 + 指标）已落地；认证/限流仍未实现 |
| P3-5 | 其他死代码：`Embedder.get_embeddings_batch/reset`、`SemanticCache.clear/stats`、`MilvusStore.count` 均无调用方 | 对应文件 | 清理或接线（如暴露 metrics 端点） |
| P3-6 | BM25 `_tokenize` 过滤长度>1，中文单字 token 被丢弃 | `domain/bm25_retriever.py:26` | 单字查询召回差 |
| P3-7 | ~~Docker：非 root 用户、无 healthcheck~~ | `Dockerfile`、`docker-compose.yml`、`.dockerignore` | ✅ 已解决：非 root（app 用户 uid 1000）+ Dockerfile/compose 双份 healthcheck（urllib 调 /v1/health）+ 新增 `.dockerignore`（排除 .venv/.uv-cache/.env 等进镜像） |

---

## 4. 修复路线图（照单执行）

> 每项完成后勾选，并跑对应验收标准。建议顺序：Phase 0 → 1 → 2。

### Phase 0 — 可运行性修复（P1，做完能装、能启动、缓存可用）

- [x] **P1-1** 在 `pyproject.toml`（及 `requirements.txt`）补 `pydantic-settings>=2.0.0`
  - 验收：✅ 通过（uv 安装 `pydantic-settings==2.15.0`；`import docs_seeker.app` 无 ImportError；TestClient 冒烟 OK）
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
- [ ] **P3-2** 移除 langfuse 依赖与 .env.example 变量，或真正接入（`utils/metrics.py` + `/metrics` 端点已落地；剩余：langfuse 清理/接线、LLM 调用与检索延迟指标接入，数据源已具备：LLMGateway.stats、SemanticCache.stats）
- [ ] **P3-3** 依赖单一来源：删 `requirements.txt` 或改为 `-e .[dev]` 入口，以 pyproject 为唯一正源
- [ ] **P3-4** 补中间件（日志 request_id 已落地；剩余：限流、鉴权如 API Key）
- [ ] **P3-5** 死代码清理或接线（见 3.3 P3-5）
- [ ] **P3-6** BM25 tokenize 策略调优（保留单字或按需）
- [x] **P3-7** Docker：非 root 用户（app:1000）+ healthcheck（Dockerfile 与 compose 双份，urllib 调 /v1/health，interval 30s / timeout 5s / start_period 30s）+ `.dockerignore`
  - 验收：待真机 `docker compose build` 后 `docker compose up`，`docker ps` 显示 healthy（沙箱无法连接 docker daemon，未实机验证）

---

## 5. 回归与验收基线

- 启动冒烟：`uvicorn docs_seeker.app:app` 启动无异常日志；`GET /`、`GET /v1/health`、`GET /metrics` 返回 200
- 链路冒烟：`POST /v1/retrieve` 返回带 `chapter/source` 的文档；`POST /v1/chat` 返回 answer + sources
- 缓存验证：同一问题连续问两次，第二次响应 `cached=true`
- 每次 Phase 完成后全量跑 pytest + 上述冒烟，再进入下一 Phase
