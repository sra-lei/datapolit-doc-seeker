"""docs-seeker - Prometheus 指标"""

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# HTTP 指标（由 api/middleware.py 采集）
http_requests_total = Counter("docs_seeker_http_requests_total", "HTTP 请求总数", ["method", "path", "status"])
http_request_duration_seconds = Histogram(
    "docs_seeker_http_request_duration_seconds", "HTTP 请求耗时（秒）", ["method", "path"]
)

# 语义缓存指标（由 infra/cache/semantic_cache.py 采集）
cache_hits_total = Counter("docs_seeker_cache_hits_total", "语义缓存命中次数")
cache_misses_total = Counter("docs_seeker_cache_misses_total", "语义缓存未命中次数")


def metrics_response() -> Response:
    """生成 /metrics 端点响应（Prometheus text 格式）"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
