"""docs-seeker - 依赖注入（组装点，管理全应用单例）"""
from docs_seeker.domain.services.chat_service import ChatService
from docs_seeker.domain.services.generator import Generator
from docs_seeker.domain.services.search_service import SearchService
from docs_seeker.infrastructure.cache.semantic_cache import get_semantic_cache
from docs_seeker.infrastructure.llm.gateway import get_llm_gateway
from docs_seeker.infrastructure.retrieval.composite_retriever import CompositeRetriever
from docs_seeker.infrastructure.retrieval.hybrid_router import HybridRouter
from docs_seeker.infrastructure.retrieval.query_decomposer import QueryDecomposer
from docs_seeker.infrastructure.usage import get_usage_tracker

_composite_retriever: CompositeRetriever | None = None
_generator: Generator | None = None
_query_decomposer: QueryDecomposer | None = None
_hybrid_router: HybridRouter | None = None
_chat_service: ChatService | None = None
_search_service: SearchService | None = None


def get_composite_retriever() -> CompositeRetriever:
    global _composite_retriever
    if _composite_retriever is None:
        _composite_retriever = CompositeRetriever()
    return _composite_retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(llm=get_llm_gateway())
    return _generator


def get_query_decomposer() -> QueryDecomposer:
    global _query_decomposer
    if _query_decomposer is None:
        _query_decomposer = QueryDecomposer(llm=get_llm_gateway())
    return _query_decomposer


def get_hybrid_router() -> HybridRouter:
    global _hybrid_router
    if _hybrid_router is None:
        _hybrid_router = HybridRouter()
    return _hybrid_router


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        # 所有依赖显式组装：检索/预热/问答共用同一份 CompositeRetriever
        # （进而共用同一份 BM25 索引），LLM 网关与语义缓存亦为进程内单例
        _chat_service = ChatService(
            retriever=get_composite_retriever(),
            generator=get_generator(),
            decomposer=get_query_decomposer(),
            cache=get_semantic_cache(),
            usage_tracker=get_usage_tracker(),
        )
    return _chat_service


def get_search_service() -> SearchService:
    global _search_service
    if _search_service is None:
        _search_service = SearchService(retriever=get_composite_retriever())
    return _search_service
