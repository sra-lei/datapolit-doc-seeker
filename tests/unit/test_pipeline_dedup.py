"""检索结果去重：无 id 的 chunk 不得被当成同一批（BM25 召回坍缩缺陷的回归测试）。

缺陷回顾：`MilvusStore.get_all_documents` 漏带主键 `id` → BM25 的 chunk 全是空
id → 管线按 `chunk.id or ""` 去重，把整条 BM25 召回静默归并成一条。修复 = 全量
查询带上 `id`（见 test_milvus_get_all_documents.py）+ 去重键的「来源+正文」指纹兜底。
"""

from docs_seeker.models.chunk import Chunk
from docs_seeker.models.query import Query
from docs_seeker.services.rag_pipeline import RAGPipeline, _dedup_key


class _Retriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, top_k=10, **kwargs):
        return list(self._chunks)


class _Decomposer:
    def decompose(self, question: str) -> Query:
        return Query(text=question, sub_queries=[question])


class _Generator:
    def generate(self, question, docs, conversation_history=None):
        return "答案", "high"


def _prepare(chunks: list[Chunk], top_k: int = 10):
    pipeline = RAGPipeline(retriever=_Retriever(chunks), decomposer=_Decomposer(), generator=_Generator())
    return pipeline.prepare("问题", top_k=top_k)[0]


# ---------------- 去重键 ----------------


def test_key_prefers_id():
    assert _dedup_key(Chunk(id="doc_1", text="任意")) == "doc_1"


def test_key_without_id_is_content_sensitive():
    assert _dedup_key(Chunk(text="第一段", source="x.pdf")) != _dedup_key(Chunk(text="第二段", source="x.pdf"))


def test_key_without_id_is_stable_for_same_content():
    assert _dedup_key(Chunk(text="同一段", source="x.pdf")) == _dedup_key(Chunk(text="同一段", source="x.pdf"))


def test_key_without_id_distinguishes_source():
    assert _dedup_key(Chunk(text="同一段", source="a.pdf")) != _dedup_key(Chunk(text="同一段", source="b.pdf"))


# ---------------- 管线去重 ----------------


def test_pipeline_keeps_distinct_idless_chunks():
    chunks = [Chunk(id="", text=f"片段{i}", source="x.pdf") for i in range(5)]
    assert len(_prepare(chunks)) == 5


def test_pipeline_merges_identical_idless_chunks():
    chunks = [Chunk(id="", text="同一段", source="x.pdf")] * 3
    assert len(_prepare(chunks)) == 1


def test_pipeline_dedups_by_id():
    chunks = [Chunk(id="doc_1", text="甲"), Chunk(id="doc_1", text="甲"), Chunk(id="doc_2", text="乙")]
    assert len(_prepare(chunks)) == 2


def test_pipeline_top_k_still_applies():
    chunks = [Chunk(id="", text=f"片段{i}", source="x.pdf") for i in range(8)]
    assert len(_prepare(chunks, top_k=3)) == 3
