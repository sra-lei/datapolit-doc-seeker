"""
docs-seeker - 摘要引导检索（Summary-guided Retrieval）
先检索摘要集合，找到相关章节后，再检索该章节的正文
"""
from typing import Any

from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.retriever import Retriever
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.domain.models.document import Document
from docs_seeker.infrastructure.database.milvus_client import get_milvus_store
from docs_seeker.infrastructure.embedding.embedder import get_embedder


class SummaryRetriever(Retriever):
    """摘要引导检索器

    两阶段：
    1. 在摘要集合中做向量检索，找到相关章节（Document）
    2. 用章节名过滤，在正文集合中检索（Chunk）
    """

    def __init__(self):
        self.milvus = get_milvus_store()
        self.embedder = get_embedder()
        self.collection_name = settings.collection_name
        self.summary_collection_name = settings.summary_collection_name

    def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[Chunk]:
        """摘要引导检索

        Args:
            query: 用户查询文本
            top_k: 最终返回正文数量

        Returns:
            按相关性降序的 Chunk 列表
        """
        query_vector = self.embedder.get_embedding(query)

        # 阶段1：摘要检索
        summary_hits = self.milvus.search(
            collection_name=self.summary_collection_name,
            query_vector=query_vector,
            top_k=3,
            output_fields=["text", "chapter", "chapter_title", "chunk_ids"],
        )
        if not summary_hits:
            logger.info("摘要引导检索：摘要集合为空，跳过")
            return []

        documents = [
            Document(
                id=h.get("id", ""),
                chapter=h.get("chapter", ""),
                chapter_title=h.get("chapter_title", ""),
                summary=h.get("text", ""),
            )
            for h in summary_hits
        ]

        # 提取相关章节
        chapters = list(set(d.chapter for d in documents if d.chapter))
        if not chapters:
            return []

        logger.info(f"摘要引导：相关章节 {chapters}")

        # 阶段2：正文检索（按章节过滤）
        results = self.milvus.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            output_fields=["text", "source", "pages", "chapter", "chapter_title",
                           "section", "section_title", "article", "article_title"],
        )
        chunks = [Chunk.from_dict(r) for r in results]

        # 过滤出相关章节
        filtered = [c for c in chunks if c.chapter in chapters]
        if not filtered:
            filtered = chunks  # fallback：如果过滤后为空，返回全部

        for chunk in filtered:
            chunk.score = max(0, 1 - chunk.distance)

        logger.info(f"摘要引导检索完成: query='{query[:30]}...' hits={len(filtered)}")
        return filtered[:top_k]
