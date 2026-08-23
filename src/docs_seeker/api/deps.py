"""docs-seeker - 依赖注入（单例管理）"""
from docs_seeker.application.services.chat_service import ChatService
from docs_seeker.application.services.generator import Generator
from docs_seeker.application.services.search_service import SearchService
from docs_seeker.retrieval.composite_retriever import CompositeRetriever
from docs_seeker.retrieval.hybrid_router import HybridRouter
from docs_seeker.retrieval.query_decomposer import QueryDecomposer

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


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


def get_search_service() -> SearchService:
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
