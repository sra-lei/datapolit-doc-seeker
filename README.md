# docs-seeker

在线检索与问答微服务（语义/稀疏/摘要三路融合检索 + LLM 答案生成）

## 目录结构

```
src/docs_seeker/
├── api/                          # 接口层（HTTP 适配）
│   ├── main.py                   # FastAPI 应用实例 + 中间件 + lifespan（uvicorn 入口）
│   ├── deps.py                   # 依赖注入组装点（单例管理）
│   ├── middleware.py             # 中间件（请求日志 / 指标）
│   ├── routes/                   # 路由定义（文件拍平，对外保留 /v1 前缀）
│   │   ├── __init__.py           # v1 路由聚合
│   │   ├── chat.py               # /v1/chat
│   │   ├── health.py             # /v1/health
│   │   ├── stats.py              # /v1/stats（缓存 + LLM 网关指标）
│   │   ├── milvus.py             # /v1/milvus/stats（集合监控）
│   │   └── usage.py              # /v1/usage/stats、/v1/usage/top
│   └── schemas/                  # 请求/响应模型（Pydantic）
│       ├── request.py
│       └── response.py
├── core/                         # 跨模块共享的通用代码
│   ├── config.py                 # Pydantic Settings + yaml 加载（settings/prompts/retrieval_config）
│   ├── logging.py                # 结构化日志（loguru）
│   ├── metrics.py                # Prometheus 指标
│   ├── security.py               # 安全护栏（输入注入检测 / 输出脱敏）
│   ├── prompts.yaml              # Prompt 模板管理
│   └── retrieval.yaml            # 检索策略配置（权重/阈值等）
├── domain/                       # 核心业务层（独立于外部）
│   ├── models/                   # 领域模型
│   │   ├── document.py           # 文档实体
│   │   ├── chunk.py              # 文档块实体
│   │   └── query.py              # 查询实体
│   ├── services/                 # 业务服务
│   │   ├── chat_service.py       # 问答用例：检索 + 生成
│   │   ├── generator.py          # 答案生成（LLM 调用 + 置信度）
│   │   ├── rag_pipeline.py       # RAG 完整流程编排
│   │   └── top_warmup.py         # 热门问题预热器（后台线程）
│   └── interfaces/               # 抽象接口（依赖倒置）
│       ├── retriever.py          # 检索器抽象接口
│       ├── embedder.py           # 向量化接口
│       └── llm.py                # LLM 接口
└── infrastructure/               # 基础设施层（外部依赖实现）
    ├── database/                 # 数据库实现
    │   └── milvus_client.py      # Milvus 只读客户端
    ├── cache/                    # 缓存实现
    │   ├── redis_client.py       # Redis 基础客户端
    │   └── semantic_cache.py     # 语义缓存策略
    ├── llm/                      # LLM 实现
    │   └── gateway.py            # 重试/熔断/降级网关
    ├── embedding/                # 向量化实现
    │   └── embedder.py           # 查询向量化（只读）
    ├── retrieval/                # 检索策略实现
    │   ├── dense_retriever.py    # 语义检索（Milvus）
    │   ├── bm25_retriever.py     # BM25 稀疏检索
    │   ├── summary_retriever.py  # 摘要引导检索
    │   ├── composite_retriever.py# 多路融合（RRF）
    │   ├── query_decomposer.py   # 查询分解
    │   └── hybrid_router.py      # BM25 路由
    └── usage/                    # RAG 使用统计（Redis 持久化）
        ├── tracker.py            # UsageTracker 实现
        └── __init__.py           # 兼容导出 get_usage_tracker
```

## 设计原则

### 分层架构（整洁架构）

```
HTTP 层 (api/) 
    ↓ 依赖
业务服务层 (domain/services/) 
    ↓ 依赖
领域层 (domain/models + interfaces) ← 核心，定义接口
    ↑ 实现
基础设施层 (infrastructure/) ← 实现领域接口
```

- **依赖方向**：外层依赖内层，内层不依赖外层
- **领域层独立**：不依赖任何框架或外部库，只包含业务实体和接口定义
- **可测试性**：每层可独立 Mock 测试

### 接口隔离（依赖倒置）

- `domain/interfaces/` 定义抽象接口
- `infrastructure/` 实现这些接口（Milvus、Redis、LLM 等）
- 上层只依赖接口，不依赖具体实现
- 便于替换组件（如 Milvus → Qdrant）

### 关注点分离

| 层级 | 职责 | 示例 |
|------|------|------|
| **API 层** | HTTP 协议适配、参数校验、响应格式化 | FastAPI 路由 |
| **业务服务层** | 用例编排、业务流程控制 | 检索→融合→生成 |
| **领域层** | 核心业务逻辑、实体定义 | 检索策略接口、文档实体 |
| **基础设施层** | 外部依赖适配 | 数据库客户端、LLM SDK |

### 配置外部化

- 所有环境相关配置放在 `core/config.py`（Pydantic Settings + yaml）
- 敏感信息通过环境变量注入（`.env`）
- Prompt 模板与代码分离，便于调优

### 包设计原则

- `src/` 容器目录 + `docs_seeker/` Python 包
- 遵循 Python Packaging 规范，使用 `uv` 管理依赖（`uv sync` 安装项目与依赖，`uv.lock` 锁定版本）
- 明确的命名空间，避免导入冲突

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat` | 问答：检索 + LLM 生成答案 |
| GET | `/v1/health` | 健康检查 |
| GET | `/v1/stats` | 运行指标（语义缓存 + LLM 网关统计） |
| GET | `/v1/milvus/stats` | Milvus 集合监控（状态/行数/向量维度/索引） |
| GET | `/v1/usage/stats` | RAG 使用统计（总次数/成功率/活跃用户/用户 Top） |
| GET | `/v1/usage/top` | 热门问题 TopN（含语义缓存命中标记） |
| GET | `/metrics` | Prometheus 指标 |

## 启动

依赖统一由 `uv` 管理（`pyproject.toml` + `uv.lock` 为唯一正源，镜像构建同样走 uv）：

```bash
# 同步依赖（含 dev extra：pytest / ruff / mypy）
uv sync --extra dev

# 配置环境变量：复制并填写密钥（DeepSeek / 百炼 / Milvus 等）
cp .env.example .env

# 启动服务
uv run uvicorn docs_seeker.api.main:app --host 0.0.0.0 --port 8001
```

Docker 构建与启动（镜像内依赖安装使用 `uv sync --frozen`，不再使用 pip / requirements.txt）：

```bash
docker compose build
docker compose up -d
```

## 代码规范（提交前自动检查）

提交钩子基于 [pre-commit](https://pre-commit.com)，自动执行 `ruff check --fix`、`ruff format`，并用 [gitleaks](https://github.com/gitleaks/gitleaks) 扫描暂存内容中的硬编码密钥（API Key / Token / 私钥等，配置见 `.pre-commit-config.yaml`）：

```bash
# 安装钩子（一次性，dev 依赖已通过 uv sync 安装）
pre-commit install

# 手动触发一次全量检查（首次运行需联网下载 ruff-pre-commit 环境，并从源码构建 gitleaks，耗时较长）
pre-commit run --all-files
```

钩子安装后，每次 `git commit` 都会先检查/格式化改动文件并做密钥扫描；未通过则提交被拦截，修复后重新提交即可。若确有需要提交的测试占位密钥，可在 `.gitleaks.toml` 中配置 allowlist（勿把真实密钥加入）。

## 依赖

- Milvus（只读，和 doc-kit 共享）
- Redis Stack（语义缓存）
- 阿里百炼（向量化）
- DeepSeek（LLM 答案生成）

## 未来迭代方向

### BM25 索引优化

当前 BM25 检索器在启动时从 Milvus 拉全量文档，一次性构建本地倒排索引，全部存内存。
对于个人项目 / 文档量几千篇以内完全够用，但文档量大或需要实时刷新时有以下优化方向：

1. **惰性刷新（Lazy Refresh）**
   - 每次检索前对比 Milvus 的 `collection.count()` 指纹（记录数 + 最新 id 哈希）
   - 指纹未变 → 直接用内存索引（零开销）
   - 指纹变化 → 惰性重建索引，再检索

2. **分页/增量加载**
   - 启动时只加载最近 N 篇文档建索引
   - 查询时用 Milvus filter 补充检索历史文档
   - 适用于文档库持续增长且历史文档访问频率低的场景

3. **持久化索引**
   - 将 BM25 倒排索引序列化到磁盘（pickle/JSON）
   - 启动时从磁盘加载，增量同步 Milvus 变更
   - 减少冷启动时间

### 检索链路优化

4. **查询向量化缓存**
   - 相同问题复用 embedding（Redis 缓存 query→vector 映射）
   - 减少百炼 API 调用

5. **并发检索**
   - 当前 dense/bm25/summary 三路串行
   - 改为 `asyncio.gather` 并行，降低 P99 延迟

6. **检索结果持久化**
   - 热门查询的结果缓存到 Milvus（独立 collection）
   - 减少重复检索开销

## 主要改动说明

| 改动 | 原结构 | 新结构 | 原因 |
|------|--------|--------|------|
| 配置集中 | `config/settings.py` + `config/loader.py` | `core/config.py`（Pydantic Settings + yaml 加载） | 配置与日志/指标/安全同属跨模块通用代码，收敛到 `core/` |
| 领域模型 | `domain/entities/` | `domain/models/` | 命名与"业务实体/数据模型"一致 |
| 业务服务 | `application/services/` + `application/pipelines/` | `domain/services/` | 用例编排（ChatService/Generator/RAGPipeline 等）归入领域层服务 |
| 检索实现 | `retrieval/` 顶层 | `infrastructure/retrieval/` | 检索策略依赖 Milvus/embedding，属基础设施实现 |
| 基础设施 | `infra/` | `infrastructure/` | 命名规范化；`vector_store/` → `database/`；`observability/`、`security/` → `core/` |
| API 入口 | `docs_seeker/app.py` | `docs_seeker/api/main.py` | FastAPI 应用实例与中间件归入接口层 |
| 路由目录 | `api/routes/v1/` 子目录 | `api/routes/` 拍平（`/v1` 前缀保留在聚合处） | 路由按模块组织，版本前缀由前缀管理 |
| 测试组织 | `tests/*.py` 扁平 | `tests/unit/` + `tests/integration/` + `conftest.py` | 单测与集成测试分层 |

这样调整后，目录结构与目标架构保持一致，并且 README 中的设计原则说明可以帮助团队成员理解架构决策的缘由。