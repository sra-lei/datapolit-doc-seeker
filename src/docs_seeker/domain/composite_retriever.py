"""
docs-seeker - 多路检索融合编排（Reciprocal Rank Fusion）
"""
from loguru import logger

from docs_seeker.domain.dense_retriever import DenseRetriever
from docs_seeker.domain.bm25_retriever import BM25Retriever
from docs_seeker.domain.summary_retriever import SummaryRetriever


class CompositeRetriever:
    """多路融合检索器：三路检索 → RRF 融合 → 去重排序"""

    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.summary = SummaryRetriever()

    def search(self, query: str, top_k: int = 10, use_summary: bool = True) -> list[dict]:
        fetch_k = min(top_k * 3, 30)
        dense_results = self.dense.search(query, top_k=fetch_k)
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        summary_results = self.summary.search(query, top_k=fetch_k) if use_summary else []
        logger.info(f"多路检索: dense={len(dense_results)} bm25={len(bm25_results)} summary={len(summary_results)}")

        rrf_k = 60
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc.get("id", str(rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1) * 0.5
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("id", str(rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1) * 0.3
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        for rank, doc in enumerate(summary_results):
            doc_id = doc.get("id", str(rank))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1) * 0.2
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id].copy()
            doc["score"] = scores[doc_id]
            doc["sources"] = []
            if doc_id in [d.get("id") for d in dense_results]:
                doc["sources"].append("dense")
            if doc_id in [d.get("id") for d in bm25_results]:
                doc["sources"].append("bm25")
            if doc_id in [d.get("id") for d in summary_results]:
                doc["sources"].append("summary")
            results.append(doc)

        logger.info(f"RRF 融合: query='{query[:30]}...' final={len(results)}")
        return results
