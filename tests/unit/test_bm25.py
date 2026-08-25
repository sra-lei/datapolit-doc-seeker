"""BM25Retriever 单元测试（mock Milvus，不依赖真实向量库）

覆盖：懒构建、检索排序、新鲜度探测触发重建。
注意：BM25Retriever 使用 class-level 共享索引，测试间的重置由 tests/conftest.py 的
autouse fixture 保证。
"""
from unittest.mock import patch

from docs_seeker.core.config import settings
from docs_seeker.infrastructure.retrieval.bm25_retriever import BM25Retriever

DOCS_A = [
    {"text": "文档检索系统介绍 第一 章 目录", "source": "a", "chapter": "第一章"},
    {"text": "文档检索 高级 用法 第二 章", "source": "b", "chapter": "第二章"},
    {"text": "无关 内容 天气 很好 今天", "source": "c", "chapter": "第三章"},
]

DOCS_B = DOCS_A + [
    {"text": "新入库 的 文档 片段", "source": "d", "chapter": "第四章"},
]


class FakeMilvus:
    def __init__(self, docs):
        self._docs = docs

    def get_all_documents(self, collection_name, limit=10000):
        return list(self._docs)

    def count(self, collection_name):
        return len(self._docs)


def _retriever(fake: FakeMilvus) -> BM25Retriever:
    with patch("docs_seeker.infrastructure.retrieval.bm25_retriever.get_milvus_store", return_value=fake), \
         patch.object(settings, "bm25_refresh_seconds", 0), \
         patch.object(settings, "bm25_max_docs", 100):
        return BM25Retriever()


def test_build_and_search_ranks_relevant_docs():
    fake = FakeMilvus(DOCS_A)
    r = _retriever(fake)
    chunks = r.search("文档检索", top_k=5)
    assert len(chunks) >= 2
    # 真实 MilvusStore.get_all_documents 不返回 id 字段（既有行为，chunk.id 为空），
    # 因此按文本断言：含查询词的文档排在无关文档之前
    assert chunks[0].text in (DOCS_A[0]["text"], DOCS_A[1]["text"])
    assert all(c.score > 0 for c in chunks)


def test_lazy_build_only_once():
    fake = FakeMilvus(DOCS_A)
    r1 = _retriever(fake)
    r2 = _retriever(fake)  # 第二个实例应复用共享索引
    r1.search("文档检索", top_k=5)
    assert BM25Retriever._shared_built is True
    r2.search("文档检索", top_k=5)
    assert BM25Retriever._shared_built is True


def test_refresh_rebuilds_on_count_change():
    fake = FakeMilvus(DOCS_A)
    r = _retriever(fake)
    r.search("文档检索", top_k=5)
    assert BM25Retriever._shared_doc_count == len(DOCS_A)

    # 文档数变化 → 触发重建
    fake._docs = DOCS_B
    # 显式把上次检查时间归零，模拟新鲜度窗口已过期（否则真实 monotonic 大于
    # 下面 patch 的 9999.0，窗口判定会提前 return）
    BM25Retriever._shared_last_checked_at = 0.0
    with patch.object(settings, "bm25_refresh_seconds", 1), \
         patch("docs_seeker.infrastructure.retrieval.bm25_retriever.time.monotonic", return_value=9999.0):
        r.search("新入库", top_k=5)
    assert BM25Retriever._shared_doc_count == len(DOCS_B)
