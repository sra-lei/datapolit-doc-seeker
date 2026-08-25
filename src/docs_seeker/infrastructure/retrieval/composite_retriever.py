"""
docs-seeker - 多路检索融合编排（Reciprocal Rank Fusion）
"""
import copy
from typing import Any

from loguru import logger

from docs_seeker.core.config import retrieval_config
from docs_seeker.domain.interfaces.retriever import Retriever
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.infrastructure.retrieval.bm25_retriever import BM25Retriever
from docs_seeker.infrastructure.retrieval.dense_retriever import DenseRetriever
from docs_seeker.infrastructure.retrieval.summary_retriever import SummaryRetriever

_DEFAULT_WEIGHTS = {"dense": 0.5, "bm25": 0.3, "summary": 0.2}


class CompositeRetriever(Retriever):
    """多路融合检索器：三路检索 → RRF 融合 → 去重排序"""

    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.summary = SummaryRetriever()

        # 从 retrieval.yaml 读取融合参数（缺失时回退默认值）
        rrf_cfg = retrieval_config.get("rrf", {})
        self.rrf_k = int(rrf_cfg.get("k", 60))
        self.weights = rrf_cfg.get("weights", _DEFAULT_WEIGHTS)
        comp_cfg = retrieval_config.get("composite", {})
        self.fetch_factor = int(comp_cfg.get("fetch_factor", 3))
        self.max_fetch = int(comp_cfg.get("max_fetch", 30))

    def search(self, query: str, top_k: int = 10, use_summary: bool = True, **kwargs: Any) -> list[Chunk]:
        fetch_k = min(top_k * self.fetch_factor, self.max_fetch)
        dense_results = self.dense.search(query, top_k=fetch_k)
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
        return results
