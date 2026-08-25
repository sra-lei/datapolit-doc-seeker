"""CompositeRetriever（RRF 多路融合）单元测试

mock 三路检索器与检索配置，不依赖 Milvus / Redis / LLM。
"""
from unittest.mock import patch

from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.infrastructure.retrieval.composite_retriever import CompositeRetriever

_CFG = {
    "rrf": {"k": 60, "weights": {"dense": 0.5, "bm25": 0.3, "summary": 0.2}},
    "composite": {"fetch_factor": 3, "max_fetch": 30},
}


class FakeRetriever:
    """可配置返回结果的假检索器"""

    def __init__(self, results=None):
        self._results = results or []

    def search(self, query, top_k=10, **kwargs):
        return self._results[:top_k]


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, text=f"text-{cid}", source="s")


def _composite(dense=None, bm25=None, summary=None) -> CompositeRetriever:
    with patch("docs_seeker.infrastructure.retrieval.composite_retriever.DenseRetriever", FakeRetriever), \
         patch("docs_seeker.infrastructure.retrieval.composite_retriever.BM25Retriever", FakeRetriever), \
         patch("docs_seeker.infrastructure.retrieval.composite_retriever.SummaryRetriever", FakeRetriever), \
         patch("docs_seeker.infrastructure.retrieval.composite_retriever.retrieval_config", _CFG):
        comp = CompositeRetriever()
    comp.dense = FakeRetriever(dense or [])
    comp.bm25 = FakeRetriever(bm25 or [])
    comp.summary = FakeRetriever(summary or [])
    return comp


def test_rrf_fusion_merges_all_paths():
    comp = _composite(
        dense=[_chunk("a"), _chunk("b")],
        bm25=[_chunk("b"), _chunk("c")],
        summary=[_chunk("c")],
    )
    results = comp.search("测试查询", top_k=10)
    ids = [c.id for c in results]
    assert set(ids) == {"a", "b", "c"}
    # 权重 dense0.5/bm25 0.3/summary 0.2 下的 RRF 期望顺序：b(两路) > a(dense) > c(bm25+summary)
    assert ids == ["b", "a", "c"]


def test_sources_marked_per_path():
    comp = _composite(
        dense=[_chunk("a")],
        bm25=[_chunk("a")],
        summary=[_chunk("a")],
    )
    results = comp.search("测试查询", top_k=10)
    assert results[0].sources == ["dense", "bm25", "summary"]


def test_dedup_by_id():
    """同一 id 多路命中只保留一份"""
    comp = _composite(
        dense=[_chunk("a"), _chunk("a")],
        bm25=[_chunk("a")],
    )
    results = comp.search("测试查询", top_k=10)
    assert [c.id for c in results].count("a") == 1


def test_top_k_truncates():
    comp = _composite(
        dense=[_chunk(str(i)) for i in range(10)],
    )
    results = comp.search("测试查询", top_k=3)
    assert len(results) == 3


def test_use_summary_false_skips_summary():
    comp = _composite(
        dense=[_chunk("a")],
        summary=[_chunk("z")],
    )
    results = comp.search("测试查询", top_k=10, use_summary=False)
    assert [c.id for c in results] == ["a"]
