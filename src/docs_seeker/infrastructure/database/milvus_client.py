"""
docs-seeker - Milvus 只读客户端
仅提供 search/query/count，不提供 create/insert/drop（入库由 doc-kit 负责）
"""

from typing import Any

from loguru import logger
from pymilvus import MilvusClient

from docs_seeker.core.config import settings


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
            output_fields = [
                "text",
                "source",
                "pages",
                "chapter",
                "chapter_title",
                "section",
                "section_title",
                "article",
                "article_title",
            ]

        try:
            return self._do_search(collection_name, query_vector, top_k, filter_expr, output_fields)
        except Exception as e:
            # 集合未加载（code=101 collection not loaded）：自动 load 后重试一次（自愈）
            if getattr(e, "code", None) == 101:
                logger.warning(f"[Milvus] 集合 {collection_name} 未加载，自动 load 后重试")
                try:
                    self.client.load_collection(collection_name)
                    return self._do_search(collection_name, query_vector, top_k, filter_expr, output_fields)
                except Exception as e2:
                    logger.error(f"[Milvus] load+search 重试失败 {collection_name}: {e2}")
                    return []
            logger.error(f"[Milvus] search {collection_name} 失败: {e}")
            return []

    def _do_search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
        filter_expr: str,
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        """实际执行 Milvus search（供 search 调用与未加载重试复用）"""
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
            docs.append(
                {
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
                }
            )
        return docs

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
            output_fields=[
                "text",
                "source",
                "pages",
                "chapter",
                "chapter_title",
                "section",
                "section_title",
                "article",
                "article_title",
            ],
        )

    def get_all_documents(self, collection_name: str, limit: int = 10000) -> list[dict[str, Any]]:
        """全量查询文档（BM25 建索引用）"""
        try:
            results = self.client.query(
                collection_name=collection_name,
                filter="",
                output_fields=[
                    "text",
                    "source",
                    "pages",
                    "chapter",
                    "chapter_title",
                    "section",
                    "section_title",
                    "article",
                    "article_title",
                ],
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

    def get_server_version(self) -> str:
        """Milvus 服务版本"""
        try:
            return str(self.client.get_server_version())
        except Exception as e:
            logger.warning(f"[Milvus] get_server_version 失败: {e}")
            return ""

    def describe_collection(self, collection_name: str) -> dict:
        """集合概要：字段列表（含向量字段维度）。

        Returns:
            {"name": ..., "fields": [{"name", "type", "dim"?, ...}, ...]}；集合不存在/失败返回 {}
        """
        try:
            if not self.client.has_collection(collection_name):
                return {}
            info = self.client.describe_collection(collection_name)
            fields = info.get("fields", []) if isinstance(info, dict) else getattr(info, "fields", [])
            result: dict = {"name": collection_name, "fields": []}
            for field in fields or []:
                fd = field if isinstance(field, dict) else getattr(field, "to_dict", lambda: {})()
                entry: dict = {
                    "name": fd.get("name") or "",
                    "type": str(fd.get("data_type", fd.get("type", ""))),
                }
                params = fd.get("params") or {}
                if isinstance(params, dict) and params.get("dim"):
                    entry["dim"] = int(params["dim"])
                result["fields"].append(entry)
            return result
        except Exception as e:
            logger.warning(f"[Milvus] describe_collection {collection_name} 失败: {e}")
            return {}

    def describe_index(self, collection_name: str) -> dict:
        """索引概要：索引名/字段/类型/度量。

        Returns:
            {"index_name", "field_name", "index_type", "metric_type"}；无索引/失败返回 {}
        """
        try:
            if not self.client.has_collection(collection_name):
                return {}
            # pymilvus 3.x describe_index 需要 index_name 参数，先尝试枚举索引名
            index_names: list[str] = []
            try:
                listed = self.client.list_indexes(collection_name)
                index_names = [str(n) for n in listed] if isinstance(listed, list) else []
            except Exception:
                index_names = []
            if not index_names:
                # fallback：用集合 schema 中向量字段名作为索引名（Milvus 默认索引名 = 字段名）
                desc = self.describe_collection(collection_name)
                index_names = [f.get("name") for f in desc.get("fields", []) if f.get("dim")]
            if not index_names:
                return {}

            info = self.client.describe_index(collection_name, index_names[0])
            items = info if isinstance(info, list) else [info]
            first = items[0] if items else {}
            d = first if isinstance(first, dict) else getattr(first, "to_dict", lambda: {})()
            return {
                "index_name": d.get("index_name") or d.get("indexName") or str(index_names[0]),
                "field_name": d.get("field_name") or d.get("fieldName") or "",
                "index_type": d.get("index_type") or d.get("indexType") or "",
                "metric_type": d.get("metric_type") or d.get("metricType") or "",
            }
        except Exception as e:
            logger.warning(f"[Milvus] describe_index {collection_name} 失败: {e}")
            return {}

    def count(self, collection_name: str) -> int:
        """统计记录数（优先 get_collection_stats，兼容 Milvus 2.4+）"""
        try:
            if not self.client.has_collection(collection_name):
                return -1
            stats = self.client.get_collection_stats(collection_name)
            if isinstance(stats, dict) and stats.get("row_count") is not None:
                return int(stats["row_count"])
        except Exception as e:
            logger.warning(f"[Milvus] get_collection_stats {collection_name} 失败: {e}")
        try:
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
