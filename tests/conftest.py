"""pytest 公共 fixtures（tests/unit 与 tests/integration 共享）"""

import os

# 测试环境禁用 Langfuse 链路追踪：防止测试产生的观测被上报到真实项目
# （须在应用模块导入/客户端初始化之前设置）
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

import pytest  # noqa: E402

from docs_seeker.core.config import settings  # noqa: E402
from docs_seeker.domain.services.guards import get_guard_chain  # noqa: E402
from docs_seeker.infra.llm.client import LLMClient, build_transport_middlewares  # noqa: E402
from docs_seeker.infra.llm.middleware import CircuitBreaker  # noqa: E402
from docs_seeker.infra.llm.middleware import retry as retry_module  # noqa: E402
from docs_seeker.infra.retrieval.bm25_retriever import BM25Retriever  # noqa: E402


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


@pytest.fixture
def make_llm_client(monkeypatch):
    """构造接假 SDK 的 ``LLMClient``（等价于生产组装，但 SDK 可替换）。

    生产路径的「读配置 + 初始化 client」在 ``api/deps.py::build_llm_client``；
    本 fixture 只做组装，让单测能注入假 SDK，不必碰 env / 真实 OpenAI 构造。

    调用形状：``make_llm_client(primary_sdk, fallback_client=..., breaker=...)``
    """
    # 退避等待在单测里必须跳过（否则重试用例会真的睡 1+2+4 秒）
    monkeypatch.setattr(retry_module.time, "sleep", lambda _s: None)

    def _make(
        primary_client,
        *,
        fallback_client=None,
        middlewares=None,
        breaker=None,
        primary_model: str = "deepseek-chat",
        fallback_model: str = "deepseek-chat",
        default_timeout: float | None = None,
        middleware_names: str | None = None,
    ) -> LLMClient:
        breaker = breaker if breaker is not None else CircuitBreaker()
        if middlewares is None:
            middlewares = build_transport_middlewares(
                guard_chain=get_guard_chain(),
                breaker=breaker,
                has_fallback=lambda: fallback_client is not None,
                names=settings.llm_transport_middlewares if middleware_names is None else middleware_names,
            )
        return LLMClient(
            primary_client=primary_client,
            primary_model=primary_model,
            fallback_client=fallback_client,
            fallback_model=fallback_model,
            circuit_breaker=breaker,
            default_timeout=settings.llm_timeout_seconds if default_timeout is None else default_timeout,
            middlewares=middlewares,
        )

    return _make
