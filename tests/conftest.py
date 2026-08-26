"""pytest 公共 fixtures（tests/unit 与 tests/integration 共享）"""
import os

# 测试环境禁用 Langfuse 链路追踪：防止测试产生的观测被上报到真实项目
# （须在应用模块导入/客户端初始化之前设置）
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

import pytest  # noqa: E402

from docs_seeker.infrastructure.retrieval.bm25_retriever import BM25Retriever  # noqa: E402


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
