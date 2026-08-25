"""docs-seeker - /v1 聚合路由（统一挂载各子路由，对外路径保留 /v1 前缀）"""

from fastapi import APIRouter

from docs_seeker.api.routes.chat import router as chat_router
from docs_seeker.api.routes.health import router as health_router
from docs_seeker.api.routes.milvus import router as milvus_router
from docs_seeker.api.routes.retrieve import router as retrieve_router
from docs_seeker.api.routes.stats import router as stats_router
from docs_seeker.api.routes.usage import router as usage_router

router = APIRouter(prefix="/v1")
router.include_router(chat_router)
router.include_router(health_router)
router.include_router(milvus_router)
router.include_router(retrieve_router)
router.include_router(stats_router)
router.include_router(usage_router)
