"""
docs-seeker - 语义检索（Dense Retrieval）
基于 Milvus 向量检索
"""

from typing import Any

from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.retriever import Retriever
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.infrastructure.database.milvus_client import get_milvus_store
from docs_seeker.infrastructure.embedding.embedder import get_embedder


class DenseRetriever(Retriever):
    """语义检索器：查询 → embedding → Milvus search"""

    def __init__(self):
        self.milvus = get_milvus_store()
        self.embedder = get_embedder()
        self.collection_name = settings.collection_name

    def search(self, query: str, top_k: int = 10, filter_expr: str = "", **kwargs: Any) -> list[Chunk]:
        """语义检索

        Args:
            query: 用户查询文本
            top_k: 返回数量
            filter_expr: Milvus filter 表达式

        Returns:
            按相关性降序的 Chunk 列表
        """
        query_vector = self.embedder.get_embedding(query)
        results = self.milvus.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr,
        )
        chunks = []
        for doc in results:
            # Milvus 返回 distance（越小越相似），转为 score（越大越好）
            doc["score"] = max(0, 1 - doc.get("distance", 0))
            chunks.append(Chunk.from_dict(doc))
        logger.info(f"DenseRetriever 检索完成: query='{query[:30]}...' top_k={top_k} hits={len(chunks)}")
        return chunks
