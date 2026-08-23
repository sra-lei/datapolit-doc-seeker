"""docs-seeker - /v1/health 路由"""
from fastapi import APIRouter

from docs_seeker.api.schemas import HealthResponse
from docs_seeker.infra.cache.semantic_cache import get_semantic_cache
from docs_seeker.infra.vector_store.milvus_client import get_milvus_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    milvus_ok, redis_ok = False, False
    try:
        store = get_milvus_store()
        milvus_ok = store.has_collection(store.collection_name)
    except Exception:
        pass
    try:
        redis_ok = get_semantic_cache()._available
    except Exception:
        pass
    return HealthResponse(status="ok", milvus_connected=milvus_ok, redis_connected=redis_ok)
