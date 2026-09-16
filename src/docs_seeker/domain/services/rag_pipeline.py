"""docs-seeker - RAG 完整流程编排"""

import hashlib

from langfuse import get_client, observe
from loguru import logger

from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.domain.models.query import Query
from docs_seeker.domain.services.generator import Generator
from docs_seeker.infrastructure.retrieval.composite_retriever import CompositeRetriever
from docs_seeker.infrastructure.retrieval.metadata_filter import parse_question_metadata
from docs_seeker.infrastructure.retrieval.query_decomposer import QueryDecomposer


class RAGPipeline:
    """问答主流程：查询分解 → 多路检索 → 去重 → 答案生成"""

    def __init__(
        self,
        retriever: CompositeRetriever | None = None,
        decomposer: QueryDecomposer | None = None,
        generator: Generator | None = None,
    ):
        self.retriever = retriever or CompositeRetriever()
        self.decomposer = decomposer or QueryDecomposer()
        self.generator = generator or Generator()

    @observe(name="retrieve-context", capture_input=False, capture_output=False)
    def prepare(
        self,
        question: str,
        top_k: int = 10,
        use_summary: bool = True,
        conversation_history: list[dict] | None = None,
    ) -> tuple[list[Chunk], list[str]]:
        """检索准备：查询分解 → 多路检索 → 去重。

        Returns:
            (deduped_chunks, sub_questions)
        """
        get_client().update_current_span(input={"question": question, "top_k": top_k, "use_summary": use_summary})
        q: Query = self.decomposer.decompose(question)
        sub_questions = q.sub_queries or [question]

        # 结构化元数据过滤：从**原问题**解析（子问题是改写，条号常被丢掉），
        # 再对所有子问题统一生效。解析不出结构词 → None → 检索走旧行为。
        meta_filter = parse_question_metadata(question) or None
        if meta_filter:
            logger.info(f"结构化元数据过滤: {meta_filter}")

        # 对每个子问题检索并合并
        all_chunks: list[Chunk] = []
        for sq in sub_questions:
            all_chunks.extend(self.retriever.search(sq, top_k=top_k, use_summary=use_summary, meta_filter=meta_filter))

        # 去重：优先按 id；无 id 的 chunk 退回「来源 + 正文」指纹（见 _dedup_key）。
        # 历史缺陷：BM25 的 chunk 曾因全量查询漏带主键 id 而全是空 id，
        # `chunk.id or ""` 把整条 BM25 召回归并成一条 —— 现在有指纹兜底。
        seen: set[str] = set()
        deduped: list[Chunk] = []
        for chunk in all_chunks:
            chunk_key = _dedup_key(chunk)
            if chunk_key not in seen:
                seen.add(chunk_key)
                deduped.append(chunk)
        deduped = deduped[:top_k]
        get_client().update_current_span(output={"chunks": len(deduped), "sub_questions": sub_questions})
        return deduped, sub_questions

    def run(
        self,
        question: str,
        top_k: int = 10,
        use_summary: bool = True,
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, str, list[Chunk], list[str]]:
        """执行完整 RAG 流程

        Returns:
            (answer, confidence, deduped_chunks, sub_questions)
        """
        deduped, sub_questions = self.prepare(
            question, top_k=top_k, use_summary=use_summary, conversation_history=conversation_history
        )

        answer, confidence = self.generator.generate(question, deduped, conversation_history)
        logger.info(f"RAG 流程完成: sub_questions={len(sub_questions)} deduped={len(deduped)} confidence={confidence}")
        return answer, confidence, deduped, sub_questions


def _dedup_key(chunk: Chunk) -> str:
    """去重键：优先 ``chunk.id``；无 id 时退回「来源 + 正文」指纹。

    为什么需要指纹兜底：BM25 的 chunk 来自 Milvus 全量查询，一旦该查询漏带主键
    `id`（历史缺陷，见 `MilvusStore.get_all_documents`），所有 BM25 结果都会落到
    空 id 上——旧的 `chunk.id or ""` 会把整条 BM25 召回静默归并成一条。指纹保证
    「内容不同的片段不会被当成同一个」，同时内容相同的重复片段仍会正常合并。
    """
    if chunk.id:
        return chunk.id
    digest = hashlib.md5(f"{chunk.source}|{chunk.text}".encode(), usedforsecurity=False).hexdigest()[:16]
    return f"noid:{digest}"
