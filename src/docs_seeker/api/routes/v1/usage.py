"""docs-seeker - /v1/usage/stats 路由（RAG 使用统计，按用户维度）"""
from fastapi import APIRouter

from docs_seeker.api.schemas import UsageStatsResponse, UsageUserStat
from docs_seeker.infra.usage_tracker import get_usage_tracker

router = APIRouter(tags=["usage"])


@router.get("/usage/stats", response_model=UsageStatsResponse)
async def usage_stats():
    """RAG 使用统计：总次数/成功率/活跃用户/用户 Top（按调用次数降序）"""
    data = get_usage_tracker().stats()
    return UsageStatsResponse(
        total_calls=data["total_calls"],
        success_calls=data["success_calls"],
        success_rate=data["success_rate"],
        active_users=data["active_users"],
        users=[UsageUserStat(**u) for u in data["users"]],
    )
