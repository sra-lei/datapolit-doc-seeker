# docs-seeker

在线检索与问答微服务（语义/稀疏/摘要三路融合检索 + LLM 答案生成）

## 目录结构

```
src/docs_seeker/
├── app.py                      # FastAPI 应用入口
├── config/                     # 配置管理
│   ├── settings.py             # Pydantic Settings（环境变量/多环境）
│   ├── prompts.yaml            # Prompt 模板管理
│   └── retrieval.yaml          # 检索策略配置（权重/阈值等）
├── domain/                     # 领域层（核心实体 + 接口定义）
│   ├── entities/               # 领域实体
│   │   ├── document.py         # 文档实体
│   │   ├── chunk.py            # 文档块实体
│   │   └── query.py            # 查询实体
│   └── interfaces/             # 接口定义（依赖倒置）
│       ├── retriever.py        # 检索器抽象接口
│       ├── embedder.py         # 向量化接口
│       └── llm.py              # LLM 接口
├── retrieval/                  # 检索策略实现（原 domain/ 重命名）
│   ├── dense_retriever.py      # 语义检索（Milvus）
│   ├── bm25_retriever.py       # BM25 稀疏检索
│   ├── summary_retriever.py    # 摘要引导检索
│   ├── composite_retriever.py  # 多路融合（RRF）
│   ├── query_decomposer.py     # 查询分解
│   └── hybrid_router.py        # BM25 路由
├── application/                # 应用层/用例层（业务编排）
│   ├── services/               # 应用服务
│   │   ├── chat_service.py     # 问答用例：检索 + 生成
│   │   ├── search_service.py   # 纯检索用例
│   │   └── generator.py        # 答案生成（LLM 调用 + 置信度）
│   └── pipelines/              # 流程管道
│       └── rag_pipeline.py     # RAG 完整流程编排
├── infra/                      # 基础设施层（外部依赖实现）
│   ├── vector_store/           # 向量存储
│   │   └── milvus_client.py    # Milvus 只读客户端
│   ├── cache/                  # 缓存
│   │   ├── redis_client.py     # Redis 基础客户端
│   │   └── semantic_cache.py   # 语义缓存策略
│   ├── llm/                    # LLM 网关
│   │   └── gateway.py          # 重试/熔断/降级
│   ├── embedding/              # 向量化
│   │   └── embedder.py         # 查询向量化（只读）
│   └── security/               # 安全
│       └── guard.py            # 安全护栏（输入/输出过滤）
├── api/                        # HTTP 层（接口适配器）
│   ├── routes/                 # 路由定义
│   │   ├── v1/                 # API 版本
│   │   │   ├── chat.py         # /v1/chat
│   │   │   ├── retrieve.py     # /v1/retrieve
│   │   │   └── health.py       # /v1/health
│   │   └── __init__.py
│   ├── schemas/                # 请求/响应模型
│   │   ├── request.py
│   │   └── response.py
│   ├── deps.py                 # 依赖注入（单例管理）
│   └── middleware.py           # 中间件（请求日志 / 指标）
└── utils/                      # 工具函数
    ├── logger.py               # 结构化日志
    └── metrics.py              # Prometheus 指标
```

## 设计原则

### 分层架构（整洁架构）

```
HTTP 层 (api/) 
    ↓ 依赖
应用层 (application/) 
    ↓ 依赖
领域层 (domain/) ← 核心，定义接口
    ↑ 实现
基础设施层 (infra/) ← 实现领域接口
```

- **依赖方向**：外层依赖内层，内层不依赖外层
- **领域层独立**：不依赖任何框架或外部库，只包含业务实体和接口定义
- **可测试性**：每层可独立 Mock 测试

### 接口隔离（依赖倒置）

- `domain/interfaces/` 定义抽象接口
- `infra/` 实现这些接口（Milvus、Redis、LLM 等）
- 上层只依赖接口，不依赖具体实现
- 便于替换组件（如 Milvus → Qdrant）

### 关注点分离

| 层级 | 职责 | 示例 |
|------|------|------|
| **API 层** | HTTP 协议适配、参数校验、响应格式化 | FastAPI 路由 |
| **应用层** | 用例编排、业务流程控制 | 检索→融合→生成 |
| **领域层** | 核心业务逻辑、实体定义 | 检索策略接口、文档实体 |
| **基础设施层** | 外部依赖适配 | 数据库客户端、LLM SDK |

### 配置外部化

- 所有环境相关配置放在 `config/` 下
- 敏感信息通过环境变量注入（`.env`）
- Prompt 模板与代码分离，便于调优

### 包设计原则

- `src/` 容器目录 + `docs_seeker/` Python 包
- 遵循 Python Packaging 规范，支持 `pip install -e .` 开发模式
- 明确的命名空间，避免导入冲突

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat` | 问答：检索 + LLM 生成答案 |
| POST | `/v1/retrieve` | 纯检索：返回文档列表 |
| GET | `/v1/health` | 健康检查 |
| GET | `/v1/stats` | 运行指标（语义缓存 + LLM 网关统计） |
| GET | `/v1/milvus/stats` | Milvus 集合监控（状态/行数/向量维度/索引） |
| GET | `/v1/usage/stats` | RAG 使用统计（总次数/成功率/活跃用户/用户 Top） |
| GET | `/v1/usage/top` | 热门问题 TopN（含语义缓存命中标记） |
| GET | `/metrics` | Prometheus 指标 |

## 启动

```bash
pip install -e .
uvicorn docs_seeker.app:app --host 0.0.0.0 --port 8001
```

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
   - 参考 chartermate 原 `BM25Retriever.ensure_fresh()` 方案

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
| 配置拆分 | `config/config.py` | `config/settings.py` + `*.yaml` | 配置外部化，环境与业务配置分离 |
| 领域层重构 | `domain/*_retriever.py` | `domain/entities/` + `domain/interfaces/` | 接口与实现分离，符合依赖倒置 |
| 检索策略独立 | 放在 `domain/` | 移至 `retrieval/` | `domain/` 只保留核心实体和接口 |
| 新增应用层 | 无 | `application/services/` + `pipelines/` | 业务编排层，避免 API 直接调用检索逻辑 |
| 基础设施细化 | `infra/*.py` | `infra/vector_store/`、`cache/`、`llm/` 等子目录 | 按职责分类，便于扩展 |
| API 路由拆分 | `api/routes.py` | `api/routes/v1/` 目录 | 支持 API 版本管理 |
| 新增工具层 | 无 | `utils/` | 横切关注点（日志、监控） |

这样调整后，目录结构与之前讨论的架构建议保持一致，并且 README 中的设计原则说明可以帮助团队成员理解架构决策的缘由。