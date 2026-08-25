"""docs-seeker - /v1/milvus/stats 路由（Milvus 集合级监控，只读）"""

from fastapi import APIRouter

from docs_seeker.api.schemas import (
    MilvusCollectionStats,
    MilvusIndexInfo,
    MilvusStatsResponse,
)
from docs_seeker.infrastructure.database.milvus_client import get_milvus_store

router = APIRouter(tags=["milvus"])


def _collection_stats(store, name: str) -> MilvusCollectionStats:
    desc = store.describe_collection(name)
    exists = bool(desc)
    vector_field = next((f for f in desc.get("fields", []) if f.get("dim")), None)
    index = store.describe_index(name) if exists else {}
    return MilvusCollectionStats(
        name=name,
        exists=exists,
        count=store.count(name) if exists else -1,
        dim=vector_field.get("dim") if vector_field else None,
        index=MilvusIndexInfo(**index) if index else None,
    )


# 同步 handler：FastAPI 自动放入线程池执行，避免阻塞事件循环
@router.get("/milvus/stats", response_model=MilvusStatsResponse)
def milvus_stats():
    """Milvus 集合监控：在线状态 / 实体数 / 向量维度 / 索引信息"""
    store = get_milvus_store()
    docs = _collection_stats(store, store.collection_name)
    summaries = _collection_stats(store, store.summary_collection_name)
    return MilvusStatsResponse(
        connected=docs.exists or summaries.exists,
        server_version=store.get_server_version(),
        collections={"docs": docs, "summaries": summaries},
    )
