"""docs-seeker - API 中间件（请求日志 / 指标）"""
import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from docs_seeker.core.metrics import http_request_duration_seconds, http_requests_total
from docs_seeker.infrastructure.usage import get_usage_tracker


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """为每个请求生成 request_id，记录访问日志并采集 Prometheus 指标"""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} "
            f"({duration * 1000:.1f}ms)"
        )
        response.headers["X-Request-ID"] = request_id

        http_requests_total.labels(
            method=request.method, path=request.url.path, status=response.status_code
        ).inc()
        http_request_duration_seconds.labels(method=request.method, path=request.url.path).observe(duration)

        # RAG 使用统计埋点（仅 /v1/chat、/v1/retrieve；Redis 不可用时静默降级）
        get_usage_tracker().record(
            request.headers.get("X-User-ID", ""),
            request.url.path,
            response.status_code,
        )
        return response
