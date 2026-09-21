"""
docs-seeker - 多路检索融合编排（Reciprocal Rank Fusion）
"""

import copy
from typing import Any

from langfuse import get_client, observe
from loguru import logger

from docs_seeker.config.settings import settings
from docs_seeker.infra.retrieval.bm25_retriever import BM25Retriever
from docs_seeker.infra.retrieval.dense_retriever import DenseRetriever
from docs_seeker.infra.retrieval.metadata_filter import build_milvus_expr
from docs_seeker.infra.retrieval.summary_retriever import SummaryRetriever
from docs_seeker.interfaces.retriever import Retriever
from docs_seeker.models.chunk import Chunk


class CompositeRetriever(Retriever):
    """多路融合检索器：三路检索 → RRF 融合 → 去重排序"""

    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.summary = SummaryRetriever()

        # 融合参数走环境变量配置（settings；默认值与旧 retrieval.yaml 一致）
        self.rrf_k = settings.retrieval_rrf_k
        self.weights = {
            "dense": settings.retrieval_rrf_weight_dense,
            "bm25": settings.retrieval_rrf_weight_bm25,
            "summary": settings.retrieval_rrf_weight_summary,
        }
        self.fetch_factor = settings.retrieval_fetch_factor
        self.max_fetch = settings.retrieval_max_fetch

    @observe(name="retrieve-multi-route", as_type="retriever", capture_input=False, capture_output=False)
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_summary: bool = True,
        meta_filter: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> list[Chunk]:
        """三路检索 + RRF 融合。

        Args:
            meta_filter: 结构化元数据过滤（``{字段: [取值前缀]}``，见
                ``domain.services.metadata.parse_question_metadata``）。dense 路转成 Milvus
                过滤表达式、bm25 路转成进程内谓词；None/空 = 不过滤（旧行为）。
        """
        # Langfuse：检索观测只记录查询与结果规模，不捕获全量文档正文
        get_client().update_current_span(input={"query": query, "top_k": top_k, "meta_filter": meta_filter})
        filter_expr = build_milvus_expr(meta_filter)
        fetch_k = min(top_k * self.fetch_factor, self.max_fetch)
        dense_results = self.dense.search(query, top_k=fetch_k, filter_expr=filter_expr)
        if filter_expr and not dense_results:
            # 结构词解析出来的过滤条件在本语料没有命中（条号不存在/字段缺失）时，
            # 宁可退回全量检索，也不要因为一个猜测的过滤条件把答案变成空。
            logger.warning(f"元数据过滤后 dense 无命中（filter={filter_expr}），回退全量检索")
            dense_results = self.dense.search(query, top_k=fetch_k)
        bm25_results = self.bm25.search(query, top_k=fetch_k, meta_filter=meta_filter)
        if meta_filter and not bm25_results:
            logger.warning("元数据过滤后 bm25 无命中，回退全量检索")
            bm25_results = self.bm25.search(query, top_k=fetch_k)
        summary_results = self.summary.search(query, top_k=fetch_k) if use_summary else []
        logger.info(f"多路检索: dense={len(dense_results)} bm25={len(bm25_results)} summary={len(summary_results)}")

        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        def _merge(results: list[Chunk], weight: float) -> None:
            for rank, chunk in enumerate(results):
                # 无 id 的 chunk（如 BM25）以该路的 rank 兜底，与重构前行为一致
                chunk_id = chunk.id or str(rank)
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (self.rrf_k + rank + 1) * weight
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

        _merge(dense_results, self.weights.get("dense", 0.5))
        _merge(bm25_results, self.weights.get("bm25", 0.3))
        _merge(summary_results, self.weights.get("summary", 0.2))

        dense_ids = [c.id for c in dense_results]
        bm25_ids = [c.id for c in bm25_results]
        summary_ids = [c.id for c in summary_results]

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results: list[Chunk] = []
        for chunk_id in sorted_ids[:top_k]:
            chunk = copy.copy(chunk_map[chunk_id])
            chunk.score = scores[chunk_id]
            chunk.sources = []
            if chunk_id in dense_ids:
                chunk.sources.append("dense")
            if chunk_id in bm25_ids:
                chunk.sources.append("bm25")
            if chunk_id in summary_ids:
                chunk.sources.append("summary")
            results.append(chunk)

        logger.info(f"RRF 融合: query='{query[:30]}...' final={len(results)}")
        get_client().update_current_span(
            output={
                "final": len(results),
                "dense": len(dense_results),
                "bm25": len(bm25_results),
                "summary": len(summary_results),
            }
        )
        return results
