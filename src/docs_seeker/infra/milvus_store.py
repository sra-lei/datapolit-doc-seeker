"""
docs-seeker - Milvus 只读客户端
仅提供 search/query/count，不提供 create/insert/drop（入库由 doc-kit 负责）
"""
from typing import Any
from loguru import logger
from pymilvus import MilvusClient

from docs_seeker.config import settings


class MilvusStore:
    """Milvus 只读存储客户端

    与 doc-kit 的 VectorStore 共享同一 Milvus 实例和 collection，
    但只做读取（search/query/count），不做写入。
    """

    def __init__(self):
        self.client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
        )
        self.collection_name = settings.collection_name
        self.summary_collection_name = settings.summary_collection_name
        logger.info(f"MilvusStore(只读) 初始化完成 | collection={self.collection_name}")

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索

        Args:
            collection_name: 集合名
            query_vector: 查询向量
            top_k: 返回数
            filter_expr: Milvus filter 表达式（如 chapter == "第一章"）
            output_fields: 要返回的字段

        Returns:
            [{id, distance, text, ...}, ...]
        """
        if output_fields is None:
            output_fields = ["text", "source", "pages", "chapter", "chapter_title",
                             "section", "section_title", "article", "article_title"]

        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
                filter=filter_expr or "",
                output_fields=output_fields,
            )
            if not results:
                return []

            # MilvusClient.search 返回 [[{id, distance, entity:{...}}, ...]]
            hits = results[0] if isinstance(results[0], list) else results
            docs = []
            for hit in hits:
                entity = hit.get("entity", hit) if isinstance(hit, dict) else {}
                docs.append({
                    "id": hit.get("id", ""),
                    "distance": hit.get("distance", 0.0),
                    "text": entity.get("text", ""),
                    "source": entity.get("source", ""),
                    "pages": entity.get("pages", ""),
                    "chapter": entity.get("chapter", ""),
                    "chapter_title": entity.get("chapter_title", ""),
                    "section": entity.get("section", ""),
                    "section_title": entity.get("section_title", ""),
                    "article": entity.get("article", ""),
                    "article_title": entity.get("article_title", ""),
                })
            return docs
        except Exception as e:
            logger.error(f"[Milvus] search {collection_name} 失败: {e}")
            return []

    def query_by_chapter(
        self,
        collection_name: str,
        chapters: list[str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """按章节查询正文（摘要引导检索用）"""
        if not chapters:
            return []

        # Milvus filter: chapter == "第一章" or chapter == "第二章"
        chapter_list = ", ".join(f'"{ch}"' for ch in chapters)
        filter_expr = f"chapter in [{chapter_list}]"

        return self.search(
            collection_name=collection_name,
            query_vector=[],  # 不按向量检索，按 filter
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=["text", "source", "pages", "chapter", "chapter_title",
                           "section", "section_title", "article", "article_title"],
        )

    def get_all_documents(self, collection_name: str, limit: int = 10000) -> list[dict[str, Any]]:
        """全量查询文档（BM25 建索引用）"""
        try:
            results = self.client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["text", "source", "pages", "chapter", "chapter_title",
                               "section", "section_title", "article", "article_title"],
                limit=limit,
            )
            return results if results else []
        except Exception as e:
            logger.error(f"[Milvus] query all {collection_name} 失败: {e}")
            return []

    def has_collection(self, collection_name: str) -> bool:
        try:
            return self.client.has_collection(collection_name)
        except Exception:
            return False

    def count(self, collection_name: str) -> int:
        """统计记录数"""
        try:
            if not self.client.has_collection(collection_name):
                return -1
            result = self.client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["count(*)"],
                limit=1,
            )
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict) and "count(*)" in first:
                    return int(first["count(*)"])
                return len(result)
        except Exception as e:
            logger.warning(f"[Milvus] count {collection_name} 失败: {e}")
        return -1


_milvus_store: MilvusStore | None = None


def get_milvus_store() -> MilvusStore:
    global _milvus_store
    if _milvus_store is None:
        _milvus_store = MilvusStore()
    return _milvus_store
