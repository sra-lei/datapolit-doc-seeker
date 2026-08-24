"""docs-seeker - 问答用例服务"""
from dataclasses import dataclass, field

from loguru import logger

from docs_seeker.application.pipelines.rag_pipeline import RAGPipeline
from docs_seeker.infra.cache.semantic_cache import get_semantic_cache
from docs_seeker.infra.security.guard import check_injection, sanitize_output
from docs_seeker.infra.usage_tracker import get_usage_tracker

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

    def __init__(self):
        self.pipeline = RAGPipeline()
        self.cache = get_semantic_cache()

    def chat(self, question: str, history: list[dict] | None = None,
             top_k: int = 10, use_cache: bool = True) -> ChatResult:
        ok, reason = check_injection(question)
        if not ok:
            return ChatResult(answer=reason, confidence="low")

        # 热门问题计数（精确匹配归并；Redis 不可用时降级）
        get_usage_tracker().record_question(question)

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
