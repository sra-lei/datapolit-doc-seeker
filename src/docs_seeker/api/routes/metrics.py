from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docs_seeker.api.routes import router


@router.get("/metrics")
async def metrics():
    """生成 /metrics 端点响应（Prometheus text 格式）"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)