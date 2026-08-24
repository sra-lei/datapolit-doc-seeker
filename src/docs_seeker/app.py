"""docs-seeker - FastAPI 应用入口"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from docs_seeker.api.middleware import RequestLoggingMiddleware
from docs_seeker.api.routes import router
from docs_seeker.application.services.top_warmup import get_top_warmup
from docs_seeker.config import settings
from docs_seeker.utils.logger import setup_logging
from docs_seeker.utils.metrics import metrics_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info("docs-seeker 启动中...")
    get_top_warmup().start()
    yield
    get_top_warmup().stop()
    logger.info("docs-seeker 已关闭")


app = FastAPI(
    title="docs-seeker",
    description="在线检索与问答微服务（语义/稀疏/摘要三路融合检索 + LLM 答案生成）",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)
app.include_router(router)


@app.get("/")
async def root():
    return {"service": "docs-seeker", "version": "0.1.0", "docs": "/docs"}


@app.get("/metrics")
async def metrics():
    return metrics_response()
