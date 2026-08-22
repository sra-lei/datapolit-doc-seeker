"""docs-seeker - 依赖注入（单例管理）"""
from docs_seeker.domain.composite_retriever import CompositeRetriever
from docs_seeker.domain.generator import Generator
from docs_seeker.domain.query_decomposer import QueryDecomposer
from docs_seeker.domain.hybrid_router import HybridRouter
from docs_seeker.infra.semantic_cache import get_semantic_cache, SemanticCache
from docs_seeker.infra.guard import check_injection, sanitize_output


_composite_retriever: CompositeRetriever | None = None
_generator: Generator | None = None
_query_decomposer: QueryDecomposer | None = None
_hybrid_router: HybridRouter | None = None


def get_composite_retriever() -> CompositeRetriever:
    global _composite_retriever
    if _composite_retriever is None:
        _composite_retriever = CompositeRetriever()
    return _composite_retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def get_query_decomposer() -> QueryDecomposer:
    global _query_decomposer
    if _query_decomposer is None:
        _query_decomposer = QueryDecomposer()
    return _query_decomposer


def get_hybrid_router() -> HybridRouter:
    global _hybrid_router
    if _hybrid_router is None:
        _hybrid_router = HybridRouter()
    return _hybrid_router
