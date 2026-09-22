"""docs-seeker - 依赖注入（组装点，管理全应用单例）"""

from langfuse.openai import OpenAI

from docs_seeker.agent.runner import AgentRunner
from docs_seeker.config.settings import settings
from docs_seeker.services.semantic_cache import get_semantic_cache
from docs_seeker.infra.llm.middleware.guards import get_guard_chain
from docs_seeker.infra.llm.client import LLMClient, build_transport_middlewares
from docs_seeker.infra.llm.middleware import CircuitBreaker
from docs_seeker.infra.retrieval.composite_retriever import CompositeRetriever
from docs_seeker.infra.retrieval.hybrid_router import HybridRouter
from docs_seeker.infra.usage import get_usage_store
from docs_seeker.services.chat_service import ChatService
from docs_seeker.services.generator import Generator
from docs_seeker.services.query_decomposer import QueryDecomposer
from docs_seeker.services.top_warmup import TopQuestionWarmup
from docs_seeker.services.usage import UsageTracker

_composite_retriever: CompositeRetriever | None = None
_generator: Generator | None = None
_query_decomposer: QueryDecomposer | None = None
_hybrid_router: HybridRouter | None = None
_chat_service: ChatService | None = None
_agent_runner: AgentRunner | None = None
_usage_tracker: UsageTracker | None = None
_top_warmup: TopQuestionWarmup | None = None
_llm_client: LLMClient | None = None


def build_llm_client() -> LLMClient:
    """组装 LLM 客户端 —— **全应用唯一的「读配置 + 初始化 client」点**。

    ``LLMClient`` 自身不读 ``settings``、不构造 SDK；这里把配置读出来、把 SDK 客户端
    与 middleware 链建好，再显式注入（构造参数无默认值，漏配即报错）。
    """
    primary_client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    fallback_client = None
    if settings.fallback_api_key and settings.fallback_base_url:
        fallback_client = OpenAI(api_key=settings.fallback_api_key, base_url=settings.fallback_base_url)
    breaker = CircuitBreaker(
        failure_threshold=settings.llm_circuit_failure_threshold,
        recovery_timeout=settings.llm_circuit_recovery_seconds,
    )
    return LLMClient(
        primary_client=primary_client,
        primary_model=settings.llm_model,
        fallback_client=fallback_client,
        fallback_model=settings.fallback_model,
        circuit_breaker=breaker,
        default_timeout=settings.llm_timeout_seconds,
        middlewares=build_transport_middlewares(
            guard_chain=get_guard_chain(),
            breaker=breaker,
            has_fallback=lambda: fallback_client is not None,
            names=settings.llm_transport_middlewares,
        ),
    )


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = build_llm_client()
    return _llm_client


def get_composite_retriever() -> CompositeRetriever:
    global _composite_retriever
    if _composite_retriever is None:
        _composite_retriever = CompositeRetriever()
    return _composite_retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(llm=get_llm_client())
    return _generator


def get_query_decomposer() -> QueryDecomposer:
    global _query_decomposer
    if _query_decomposer is None:
        _query_decomposer = QueryDecomposer(llm=get_llm_client())
    return _query_decomposer


def get_hybrid_router() -> HybridRouter:
    global _hybrid_router
    if _hybrid_router is None:
        _hybrid_router = HybridRouter()
    return _hybrid_router


def get_agent_runner() -> AgentRunner:
    """Agentic M1：复用单例 LLM 客户端与混合检索器（含同一份 BM25 索引）。"""
    global _agent_runner
    if _agent_runner is None:
        _agent_runner = AgentRunner(
            llm=get_llm_client(),
            retriever=get_composite_retriever(),
        )
    return _agent_runner


def get_usage_tracker() -> UsageTracker:
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker(store=get_usage_store(), cache=get_semantic_cache())
    return _usage_tracker


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        # 所有依赖显式组装：检索/预热/问答共用同一份 CompositeRetriever
        # （进而共用同一份 BM25 索引），LLM 客户端与语义缓存亦为进程内单例
        _chat_service = ChatService(
            retriever=get_composite_retriever(),
            generator=get_generator(),
            decomposer=get_query_decomposer(),
            cache=get_semantic_cache(),
            usage_tracker=get_usage_tracker(),
            agent_runner=get_agent_runner(),
            # 护栏链：与 client 内那处共用同一份配置/实例
            guards=get_guard_chain(),
        )
    return _chat_service


def get_top_warmup() -> TopQuestionWarmup:
    """热门问题预热器：复用 API 路径的 ChatService 单例（同一份检索/缓存）"""
    global _top_warmup
    if _top_warmup is None:
        _top_warmup = TopQuestionWarmup(
            service=get_chat_service(),
            usage_tracker=get_usage_tracker(),
        )
    return _top_warmup
