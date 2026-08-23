"""v1 路由聚合"""
from fastapi import APIRouter

from docs_seeker.api.routes.v1.chat import router as chat_router
from docs_seeker.api.routes.v1.health import router as health_router
from docs_seeker.api.routes.v1.retrieve import router as retrieve_router
from docs_seeker.api.routes.v1.stats import router as stats_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(chat_router)
v1_router.include_router(retrieve_router)
v1_router.include_router(health_router)
v1_router.include_router(stats_router)

__all__ = ["v1_router"]
