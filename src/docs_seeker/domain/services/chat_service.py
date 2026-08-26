"""docs-seeker - 问答用例服务"""

from dataclasses import dataclass, field

from langfuse import get_client, observe, propagate_attributes
from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.core.security import check_injection, sanitize_output
from docs_seeker.domain.services.generator import Generator, compute_confidence
from docs_seeker.domain.services.rag_pipeline import RAGPipeline
from docs_seeker.infrastructure.cache.semantic_cache import SemanticCache, get_semantic_cache
from docs_seeker.infrastructure.retrieval.composite_retriever import CompositeRetriever
from docs_seeker.infrastructure.retrieval.query_decomposer import QueryDecomposer
from docs_seeker.infrastructure.tracing import FEATURE_TAG, TRACE_NAME
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

    @observe(name=TRACE_NAME, capture_input=False, capture_output=False)
    def chat(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 10,
        use_cache: bool = True,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> ChatResult:
        # Langfuse：根观测只记录用户问题（而非全部函数参数），会话/用户归因通过
        # propagate_attributes 传播到该 trace 的所有子观测
        langfuse = get_client()
        langfuse.update_current_span(input={"question": question, "top_k": top_k})
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=[FEATURE_TAG],
            environment=settings.environment.lower(),
            metadata={"route": "/v1/chat"},
        ):
            ok, reason = check_injection(question)
            if not ok:
                langfuse.update_current_span(level="ERROR", status_message=reason, output={"answer": reason})
                return ChatResult(answer=reason, confidence="low")

            # 热门问题计数（精确匹配归并；Redis 不可用时降级）
            self.usage_tracker.record_question(question)

            if use_cache:
                cached = self.cache.search(question)
                if cached:
                    logger.info("语义缓存命中，直接返回缓存答案")
                    langfuse.update_current_span(
                        output={"answer": cached["answer"], "confidence": cached.get("confidence", "medium"), "cached": True}
                    )
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

            langfuse.update_current_span(
                output={"answer": answer, "confidence": confidence, "cached": False, "sources": len(source_dicts)}
            )
            return ChatResult(
                answer=answer,
                confidence=confidence,
                sources=source_dicts,
                query_decomposed=sub_questions if len(sub_questions) > 1 else None,
            )

    @observe(name=TRACE_NAME, capture_input=False, capture_output=False)
    def chat_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 10,
        use_cache: bool = True,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        """流式问答：逐事件产出 dict（由路由层序列化为 SSE）。

        事件类型：
          - {"type": "error", "message": str}            输入被拦截 / 生成失败
          - {"type": "meta", "cached": bool, "sources": [...], "query_decomposed": [...]|None, "confidence": str|None}
          - {"type": "delta", "content": str}            增量文本（可拼接为完整回答）
          - {"type": "done", "answer": str, "confidence": str, "sources": [...],
             "cached": bool, "query_decomposed": [...]|None}
        """
        # Langfuse：@observe 原生支持生成器（迭代结束/关闭时自动结束观测）
        langfuse = get_client()
        langfuse.update_current_span(input={"question": question, "top_k": top_k})
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=[FEATURE_TAG],
            environment=settings.environment.lower(),
            metadata={"route": "/v1/chat"},
        ):
            ok, reason = check_injection(question)
            if not ok:
                langfuse.update_current_span(level="ERROR", status_message=reason, output={"answer": reason})
                yield {"type": "error", "message": reason}
                return

            # 热门问题计数（精确匹配归并；Redis 不可用时降级）
            self.usage_tracker.record_question(question)

            if use_cache:
                cached = self.cache.search(question)
                if cached:
                    logger.info("语义缓存命中，流式返回缓存答案")
                    answer = cached["answer"]
                    confidence = cached.get("confidence", "medium")
                    sources = cached.get("sources", [])
                    langfuse.update_current_span(
                        output={"answer": answer, "confidence": confidence, "cached": True}
                    )
                    yield {
                        "type": "meta",
                        "cached": True,
                        "sources": sources,
                        "query_decomposed": None,
                        "confidence": confidence,
                    }
                    yield {"type": "delta", "content": answer}
                    yield {
                        "type": "done",
                        "answer": answer,
                        "confidence": confidence,
                        "sources": sources,
                        "cached": True,
                        "query_decomposed": None,
                    }
                    return

            chunks, sub_questions = self.pipeline.prepare(question, top_k=top_k, conversation_history=history)
            source_dicts = [{k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS} for chunk in chunks]
            query_decomposed = sub_questions if len(sub_questions) > 1 else None

            yield {
                "type": "meta",
                "cached": False,
                "sources": source_dicts,
                "query_decomposed": query_decomposed,
                "confidence": None,
            }

            parts: list[str] = []
            try:
                for delta in self.pipeline.generator.generate_stream(question, chunks, history):
                    parts.append(delta)
                    yield {"type": "delta", "content": delta}
            except Exception as e:
                logger.error(f"流式生成失败: {e}")
                langfuse.update_current_span(level="ERROR", status_message=f"答案生成失败: {e}")
                yield {"type": "error", "message": f"答案生成失败: {e}"}
                return

            answer = sanitize_output("".join(parts))
            confidence = compute_confidence(answer, chunks)

            if use_cache:
                self.cache.store(question, {"answer": answer, "confidence": confidence, "sources": source_dicts})

            langfuse.update_current_span(
                output={"answer": answer, "confidence": confidence, "cached": False, "sources": len(source_dicts)}
            )
            yield {
                "type": "done",
                "answer": answer,
                "confidence": confidence,
                "sources": source_dicts,
                "cached": False,
                "query_decomposed": query_decomposed,
            }
