"""docs-seeker - FastAPI 应用入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from docs_seeker.api.deps import get_chat_service, get_composite_retriever
from docs_seeker.api.middleware import RequestLoggingMiddleware
from docs_seeker.api.routes import router
from docs_seeker.core.config import settings
from docs_seeker.core.logging import setup_logging
from docs_seeker.core.metrics import metrics_response
from docs_seeker.domain.services.top_warmup import get_top_warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info("docs-seeker 启动中...")
    # 启动即构建 BM25 索引（进程内共享，首次请求不再承担全量扫描耗时）；
    # Milvus 暂不可用时不阻塞启动，首次检索会自动懒构建/周期重建
    try:
        get_composite_retriever().bm25.build_index()
        logger.info("BM25 索引启动预热完成")
    except Exception as e:
        logger.warning(f"BM25 索引启动预热失败（首次检索时将自动构建）: {e}")
    # 预热器注入与 API 路径共享的 ChatService，避免另建一套检索实例
    get_top_warmup().start(service=get_chat_service())
    yield
    get_top_warmup().stop()
    logger.info("docs-seeker 已关闭")


# 生产环境不暴露 API 文档（/docs、/redoc、/openapi.json）
_docs_enabled = settings.environment.strip().lower() not in {"production", "prod"}

app = FastAPI(
    title="docs-seeker",
    description="在线检索与问答微服务（语义/稀疏/摘要三路融合检索 + LLM 答案生成）",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(router)


@app.get("/metrics")
async def metrics():
    return metrics_response()
