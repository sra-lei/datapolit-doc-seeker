"""docs-seeker - 问答用例服务"""

from dataclasses import asdict, dataclass, field

from langfuse import get_client, observe, propagate_attributes
from loguru import logger

from docs_seeker.config.settings import settings
from docs_seeker.infra.guards import ANSWER_CTX, DOCUMENT_CTX, USER_INPUT_CTX, GuardChain, get_guard_chain
from docs_seeker.infra.tracker import FEATURE_TAG, TRACE_NAME
from docs_seeker.interfaces.cache import SemanticCachePort
from docs_seeker.interfaces.retriever import Retriever
from docs_seeker.services.generator import Generator, compute_confidence
from docs_seeker.services.query_decomposer import QueryDecomposer
from docs_seeker.services.rag_pipeline import RAGPipeline
from docs_seeker.services.usage import UsageTracker

# 写入语义缓存 / 组装响应时保留的字段（与 SourceDoc 对齐）
CACHE_FIELDS = ("id", "text", "source", "chapter", "chapter_title", "section", "section_title", "score", "sources")


def _chunk_text(text: str, size: int = 80) -> list[str]:
    """把完整文本切成固定长度片段（agent 非流式产出的答案用于 SSE 增量吐流）。

    空文本返回 []（调用方不应产出空 delta）；片段在字符边界硬切，
    拼接后与原文完全一致（消费方按顺序拼接即可还原）。
    """
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


@dataclass
class ChatResult:
    answer: str
    confidence: str = "medium"
    sources: list[dict] = field(default_factory=list)
    cached: bool = False
    query_decomposed: list[str] | None = None
    agent_steps: list[dict] | None = None  # agent 路径的可审计 trace；旧管线为 None
    agent_sufficient: bool | None = None  # agent 路径：是否主动拒答（证据不足）


class ChatService:
    """问答用例：安全检测 → 语义缓存 → RAG 流程 → 输出脱敏 → 写缓存"""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        decomposer: QueryDecomposer,
        cache: SemanticCachePort,
        usage_tracker: UsageTracker,
        agent_runner=None,
        guards: GuardChain | None = None,
    ):
        # 依赖由组合根（api/deps）注入；不在这里兜底，保持 domain 不依赖 infra
        # 工厂。guards 是 domain 内部单例，保留缺省（独立使用场景）。
        self.pipeline = RAGPipeline(retriever=retriever, decomposer=decomposer, generator=generator)
        self.cache = cache
        self.usage_tracker = usage_tracker
        # Agentic M1：默认关闭（settings.agent_enabled）；开启后 agent 路径任何异常
        # 都回退下面的旧单轮管线，保证可灰度可回退
        self.agent_runner = agent_runner
        # 护栏链路：边界挂载（可拒答/可改写）；与 client 内那处共用同一份配置
        self.guards = guards or get_guard_chain()

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
            verdict = self.guards.inspect(question, USER_INPUT_CTX)
            if not verdict.allowed:
                langfuse.update_current_span(
                    level="ERROR", status_message=verdict.reason, output={"answer": verdict.reason}
                )
                return ChatResult(answer=verdict.reason, confidence="low")

            # 热门问题计数（精确匹配归并；Redis 不可用时降级）
            self.usage_tracker.record_question(question)

            if use_cache:
                cached = self.cache.search(question)
                if cached:
                    logger.info("语义缓存命中，直接返回缓存答案")
                    langfuse.update_current_span(
                        output={
                            "answer": cached["answer"],
                            "confidence": cached.get("confidence", "medium"),
                            "cached": True,
                        }
                    )
                    return ChatResult(
                        answer=cached["answer"],
                        confidence=cached.get("confidence", "medium"),
                        sources=cached.get("sources", []),
                        cached=True,
                    )

            sub_questions: list[str] = []
            agent_steps: list[dict] | None = None
            agent_sufficient: bool | None = None
            if settings.agent_enabled and self.agent_runner is not None:
                try:
                    ar = self.agent_runner.run(question, top_k=top_k)
                    answer, confidence, chunks = ar.answer, ar.confidence, ar.evidence
                    agent_steps = [asdict(s) for s in ar.steps]
                    agent_sufficient = ar.sufficient
                    logger.info(
                        f"Agent 路径完成: steps={len(ar.steps)} evidence={len(ar.evidence)} sufficient={ar.sufficient}"
                    )
                except Exception as e:  # noqa: BLE001 — 回退契约：编排/客户端任何异常都落回旧管线
                    if not settings.agent_fallback_to_pipeline:
                        # 开发期默认不回退：让 agent 的失败显式暴露，而不是被旧管线的成功掩盖
                        # （生产默认回退，可用性优先；开关见 AGENT_FALLBACK_ENABLED）
                        logger.error(
                            f"Agent 路径失败且未启用回退（environment={settings.environment}）: {type(e).__name__}: {e}"
                        )
                        raise
                    logger.warning(f"Agent 路径失败，回退旧单轮管线: {type(e).__name__}: {e}")
                    answer, confidence, chunks, sub_questions = self.pipeline.run(
                        question, top_k=top_k, conversation_history=history
                    )
            else:
                answer, confidence, chunks, sub_questions = self.pipeline.run(
                    question, top_k=top_k, conversation_history=history
                )
            self._inspect_documents(chunks)
            answer = self.guards.inspect(answer, ANSWER_CTX).text
            source_dicts = [{k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS} for chunk in chunks]

            if use_cache and answer.strip():
                # 空答案不写缓存：推理模型预算耗尽时会产出空正文，写进去会持续污染
                # 后续命中（且极难排查）
                self.cache.store(question, {"answer": answer, "confidence": confidence, "sources": source_dicts})

            langfuse.update_current_span(
                output={"answer": answer, "confidence": confidence, "cached": False, "sources": len(source_dicts)}
            )
            return ChatResult(
                answer=answer,
                confidence=confidence,
                sources=source_dicts,
                query_decomposed=sub_questions if len(sub_questions) > 1 else None,
                agent_steps=agent_steps,
                agent_sufficient=agent_sufficient,
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
          - {"type": "meta", "cached": bool, "sources": [...], "query_decomposed": [...]|None,
             "confidence": str|None, "agent_steps": [...]|None, "agent_sufficient": bool|None}
          - {"type": "delta", "content": str}            增量文本（可拼接为完整回答）
          - {"type": "done", "answer": str, "confidence": str, "sources": [...],
             "cached": bool, "query_decomposed": [...]|None,
             "agent_steps": [...]|None, "agent_sufficient": bool|None}

        Agentic 分流（与 chat() 对称）：``settings.agent_enabled`` 且已注入
        ``agent_runner`` 时走 agent 路径 —— 完整答案产出后按片段吐 delta，
        meta/done 携带 agent_steps（可审计 trace）与 agent_sufficient；
        任何异常按 ``agent_fallback_to_pipeline`` 决定回退旧管线或显式 error。
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
            verdict = self.guards.inspect(question, USER_INPUT_CTX)
            if not verdict.allowed:
                langfuse.update_current_span(
                    level="ERROR", status_message=verdict.reason, output={"answer": verdict.reason}
                )
                yield {"type": "error", "message": verdict.reason}
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
                    langfuse.update_current_span(output={"answer": answer, "confidence": confidence, "cached": True})
                    yield {
                        "type": "meta",
                        "cached": True,
                        "sources": sources,
                        "query_decomposed": None,
                        "confidence": confidence,
                        "agent_steps": None,
                        "agent_sufficient": None,
                    }
                    yield {"type": "delta", "content": answer}
                    yield {
                        "type": "done",
                        "answer": answer,
                        "confidence": confidence,
                        "sources": sources,
                        "cached": True,
                        "query_decomposed": None,
                        "agent_steps": None,
                        "agent_sufficient": None,
                    }
                    return

            # ---- Agentic 分流（与 chat() 对称）：agent 路径产出完整答案后按片段吐流 ----
            if settings.agent_enabled and self.agent_runner is not None:
                try:
                    ar = self.agent_runner.run(question, top_k=top_k)
                    answer = ar.answer
                    confidence = ar.confidence
                    source_dicts = [
                        {k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS} for chunk in ar.evidence
                    ]
                    agent_steps = [asdict(s) for s in ar.steps]
                    agent_sufficient = ar.sufficient
                    logger.info(
                        f"Agent 路径完成: steps={len(ar.steps)} evidence={len(ar.evidence)} sufficient={ar.sufficient}"
                    )
                except Exception as e:  # noqa: BLE001 — 回退契约：编排/客户端任何异常都落回旧管线
                    if not settings.agent_fallback_to_pipeline:
                        # 与 chat() 同口径：开发期默认不回退，让 agent 的失败显式暴露
                        logger.error(
                            f"Agent 路径失败且未启用回退（environment={settings.environment}）: {type(e).__name__}: {e}"
                        )
                        yield {"type": "error", "message": f"Agent 路径失败: {type(e).__name__}: {e}"}
                        return
                    logger.warning(f"Agent 路径失败，回退旧单轮管线: {type(e).__name__}: {e}")
                    chunks, sub_questions = self.pipeline.prepare(question, top_k=top_k, conversation_history=history)
                    self._inspect_documents(chunks)
                    source_dicts = [{k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS} for chunk in chunks]
                    query_decomposed = sub_questions if len(sub_questions) > 1 else None
                    agent_steps = None
                    agent_sufficient = None
                    yield {
                        "type": "meta",
                        "cached": False,
                        "sources": source_dicts,
                        "query_decomposed": query_decomposed,
                        "confidence": None,
                        "agent_steps": None,
                        "agent_sufficient": None,
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
                    answer = self.guards.inspect("".join(parts), ANSWER_CTX).text
                    confidence = compute_confidence(answer, chunks)
                else:
                    answer = self.guards.inspect(answer, ANSWER_CTX).text
                    query_decomposed = None
                    yield {
                        "type": "meta",
                        "cached": False,
                        "sources": source_dicts,
                        "query_decomposed": None,
                        "confidence": confidence,
                        "agent_steps": agent_steps,
                        "agent_sufficient": agent_sufficient,
                    }
                    # agent 是非流式产出：把完整答案切成片段吐流（保持 SSE 增量体验）
                    for delta in _chunk_text(answer, size=80):
                        yield {"type": "delta", "content": delta}
            else:
                chunks, sub_questions = self.pipeline.prepare(question, top_k=top_k, conversation_history=history)
                self._inspect_documents(chunks)
                source_dicts = [{k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS} for chunk in chunks]
                query_decomposed = sub_questions if len(sub_questions) > 1 else None
                agent_steps = None
                agent_sufficient = None

                yield {
                    "type": "meta",
                    "cached": False,
                    "sources": source_dicts,
                    "query_decomposed": query_decomposed,
                    "confidence": None,
                    "agent_steps": None,
                    "agent_sufficient": None,
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

                answer = self.guards.inspect("".join(parts), ANSWER_CTX).text
                confidence = compute_confidence(answer, chunks)

            if use_cache and answer.strip():
                # 同 chat()：空答案不写缓存，避免污染后续命中
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
                "query_decomposed": query_decomposed if len(query_decomposed or []) > 1 else None,
                "agent_steps": agent_steps,
                "agent_sufficient": agent_sufficient,
            }

    def _inspect_documents(self, chunks) -> None:
        """检索文档正文注入扫描（**仅告警，不干预答案** —— 评审已决 2026-09-17）。

        RAG 里真正的注入通道是检索回来的文档正文：模式命中只记 warning，
        答案照常生成（可用性优先）。与 client 内那处扫描同一个护栏配置。
        """
        texts = [getattr(chunk, "text", "") or "" for chunk in chunks or []]
        if any(texts):
            self.guards.inspect_many(texts, DOCUMENT_CTX)
