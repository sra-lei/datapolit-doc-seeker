"""
docs-seeker - 摘要引导检索（Summary-guided Retrieval）
先检索摘要集合，找到相关章节后，再检索该章节的正文
"""
from loguru import logger

from docs_seeker.infra.milvus_store import get_milvus_store
from docs_seeker.infra.embedder import get_embedder
from docs_seeker.config import settings


class SummaryRetriever:
    """摘要引导检索器

    两阶段：
    1. 在摘要集合中做向量检索，找到相关章节
    2. 用章节名作为 filter，在正文集合中检索
    """

    def __init__(self):
        self.milvus = get_milvus_store()
        self.embedder = get_embedder()
        self.collection_name = settings.collection_name
        self.summary_collection_name = settings.summary_collection_name

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """摘要引导检索

        Args:
            query: 用户查询文本
            top_k: 最终返回正文数量

        Returns:
            [{id, text, source, chapter, ..., score}, ...]
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

        # 提取相关章节
        chapters = list(set(h.get("chapter", "") for h in summary_hits if h.get("chapter")))
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
        # 过滤出相关章节
        filtered = [r for r in results if r.get("chapter") in chapters]
        if not filtered:
            filtered = results  # fallback：如果过滤后为空，返回全部

        for doc in filtered:
            doc["score"] = max(0, 1 - doc.get("distance", 0))

        logger.info(f"摘要引导检索完成: query='{query[:30]}...' hits={len(filtered)}")
        return filtered[:top_k]
