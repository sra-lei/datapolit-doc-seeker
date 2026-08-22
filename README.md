# docs-seeker

在线检索与问答微服务（语义/稀疏/摘要三路融合检索 + LLM 答案生成）

## 目录结构

```
src/docs_seeker/
├── app.py               # FastAPI 应用入口
├── config/               # 配置
│   └── config.py
├── domain/               # 领域层
│   ├── dense_retriever.py    # 语义检索（Milvus）
│   ├── bm25_retriever.py      # BM25 稀疏检索
│   ├── summary_retriever.py   # 摘要引导检索
│   ├── composite_retriever.py # 多路融合（RRF）
│   ├── query_decomposer.py   # 查询分解
│   ├── hybrid_router.py       # BM25 路由
│   └── generator.py            # 答案生成
├── infra/                # 基础设施
│   ├── milvus_store.py        # Milvus 只读客户端
│   ├── embedder.py            # 查询向量化（只读）
│   ├── semantic_cache.py      # Redis 语义缓存
│   ├── llm_gateway.py         # LLM 网关（重试/熔断）
│   └── guard.py               # 安全护栏
└── api/                  # HTTP 层
    ├── routes.py              # /v1/chat, /v1/retrieve, /v1/health
    ├── schemas.py             # 请求/响应模型
    └── deps.py                # 依赖注入
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/chat` | 问答：检索 + LLM 生成答案 |
| POST | `/v1/retrieve` | 纯检索：返回文档列表 |
| GET | `/v1/health` | 健康检查 |

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
