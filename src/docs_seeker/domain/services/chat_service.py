"""docs-seeker - 问答用例服务"""

from dataclasses import dataclass, field

from loguru import logger

from docs_seeker.core.security import check_injection, sanitize_output
from docs_seeker.domain.services.generator import Generator
from docs_seeker.domain.services.rag_pipeline import RAGPipeline
from docs_seeker.infrastructure.cache.semantic_cache import SemanticCache, get_semantic_cache
from docs_seeker.infrastructure.retrieval.composite_retriever import CompositeRetriever
from docs_seeker.infrastructure.retrieval.query_decomposer import QueryDecomposer
from docs_seeker.infrastructure.usage import UsageTracker, get_usage_tracker

# 写入语义缓存 / 组装响应时保留的字段（与 SourceDoc 对齐）
CACHE_FIELDS = ("id", "text", "source", "chapter", "chapter_title", "section", "section_title", "score", "sources")


@dataclass
class ChatResult:
    answer: str
    confidence: str = "medium"
    sources: list[dict] = field(default_factory=list)
    cached: bool = False
    query_decomposed: list[str] | None = None


class ChatService:
    """问答用例：安全检测 → 语义缓存 → RAG 流程 → 输出脱敏 → 写缓存"""

    def __init__(
        self,
        retriever: CompositeRetriever | None = None,
        generator: Generator | None = None,
        decomposer: QueryDecomposer | None = None,
        cache: SemanticCache | None = None,
        usage_tracker: UsageTracker | None = None,
    ):
        # 允许注入共享依赖（deps 组装点传入）；缺省时自建/走全局单例（独立使用场景）
        self.pipeline = RAGPipeline(retriever=retriever, decomposer=decomposer, generator=generator)
        self.cache = cache or get_semantic_cache()
        self.usage_tracker = usage_tracker or get_usage_tracker()

    def chat(
        self, question: str, history: list[dict] | None = None, top_k: int = 10, use_cache: bool = True
    ) -> ChatResult:
        ok, reason = check_injection(question)
        if not ok:
            return ChatResult(answer=reason, confidence="low")

        # 热门问题计数（精确匹配归并；Redis 不可用时降级）
        self.usage_tracker.record_question(question)

        if use_cache:
            cached = self.cache.search(question)
            if cached:
                logger.info("语义缓存命中，直接返回缓存答案")
                return ChatResult(
                    answer=cached["answer"],
                    confidence=cached.get("confidence", "medium"),
                    sources=cached.get("sources", []),
                    cached=True,
                )

        answer, confidence, chunks, sub_questions = self.pipeline.run(
            question, top_k=top_k, conversation_history=history
        )
        answer = sanitize_output(answer)
        source_dicts = [{k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS} for chunk in chunks]

        if use_cache:
            self.cache.store(question, {"answer": answer, "confidence": confidence, "sources": source_dicts})

        return ChatResult(
            answer=answer,
            confidence=confidence,
            sources=source_dicts,
            query_decomposed=sub_questions if len(sub_questions) > 1 else None,
        )
