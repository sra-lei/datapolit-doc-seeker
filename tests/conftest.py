"""pytest 公共 fixtures（tests/unit 与 tests/integration 共享）"""
import pytest

from docs_seeker.infrastructure.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture(autouse=True)
def _reset_bm25_shared_index():
    """每个用例前重置 BM25Retriever 的 class-level 共享索引，避免用例间污染"""
    BM25Retriever._shared_docs = []
    BM25Retriever._shared_index = {}
    BM25Retriever._shared_avgdl = 0.0
    BM25Retriever._shared_doc_count = 0
    BM25Retriever._shared_built = False
    BM25Retriever._shared_built_at = 0.0
    BM25Retriever._shared_last_checked_at = 0.0
    yield
