"""docs-seeker - API 中间件（请求日志 / 指标）"""

import time
import uuid

from loguru import logger
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

from docs_seeker.services.usage import UsageTracker

# HTTP 指标（由 api/middleware.py 采集）
http_requests_total = Counter("docs_seeker_http_requests_total", "HTTP 请求总数", ["method", "path", "status"])
http_request_duration_seconds = Histogram(
    "docs_seeker_http_request_duration_seconds", "HTTP 请求耗时（秒）", ["method", "path"]
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """为每个请求生成 request_id，记录访问日志并采集 Prometheus 指标"""
    def __init__(self, app, tracker: UsageTracker | None = None):
        super().__init__(app)
        self.tracker = tracker

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({duration * 1000:.1f}ms)"
        )
        response.headers["X-Request-ID"] = request_id

        http_requests_total.labels(method=request.method, path=request.url.path, status=response.status_code).inc()
        http_request_duration_seconds.labels(method=request.method, path=request.url.path).observe(duration)

        # RAG 使用统计埋点（仅 /v1/chat；Redis 不可用时静默降级）
        if self.tracker is not None:
            self.tracker.record(
                request.headers.get("X-User-ID", ""),
                request.url.path,
                response.status_code,
            )
        return response
