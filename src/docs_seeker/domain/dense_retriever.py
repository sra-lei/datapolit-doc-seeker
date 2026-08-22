"""
docs-seeker - 语义检索（Dense Retrieval）
基于 Milvus 向量检索
"""
from loguru import logger

from docs_seeker.infra.milvus_store import get_milvus_store
from docs_seeker.infra.embedder import get_embedder
from docs_seeker.config import settings


class DenseRetriever:
    """语义检索器：查询 → embedding → Milvus search"""

    def __init__(self):
        self.milvus = get_milvus_store()
        self.embedder = get_embedder()
        self.collection_name = settings.collection_name

    def search(self, query: str, top_k: int = 10, filter_expr: str = "") -> list[dict]:
        """语义检索

        Args:
            query: 用户查询文本
            top_k: 返回数量
            filter_expr: Milvus filter 表达式

        Returns:
            [{id, text, source, chapter, ..., score}, ...]
        """
        query_vector = self.embedder.get_embedding(query)
        results = self.milvus.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr,
        )
        # Milvus 返回 distance（越小越相似），转为 score（越大越好）
        for doc in results:
            doc["score"] = max(0, 1 - doc.get("distance", 0))
        logger.info(f"DenseRetriever 检索完成: query='{query[:30]}...' top_k={top_k} hits={len(results)}")
        return results
