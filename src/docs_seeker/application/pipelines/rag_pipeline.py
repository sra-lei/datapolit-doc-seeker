"""docs-seeker - RAG 完整流程编排"""
from loguru import logger

from docs_seeker.application.services.generator import Generator
from docs_seeker.domain.entities.chunk import Chunk
from docs_seeker.domain.entities.query import Query
from docs_seeker.retrieval.composite_retriever import CompositeRetriever
from docs_seeker.retrieval.query_decomposer import QueryDecomposer


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
        q: Query = self.decomposer.decompose(question)
        sub_questions = q.sub_queries or [question]

        # 对每个子问题检索并合并
        all_chunks: list[Chunk] = []
        for sq in sub_questions:
            all_chunks.extend(self.retriever.search(sq, top_k=top_k, use_summary=use_summary))

        # 按 id 去重（与重构前行为一致：无 id 的 chunk 视为同一批）
        seen: set[str] = set()
        deduped: list[Chunk] = []
        for chunk in all_chunks:
            chunk_id = chunk.id or ""
            if chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(chunk)
        deduped = deduped[:top_k]

        answer, confidence = self.generator.generate(question, deduped, conversation_history)
        logger.info(f"RAG 流程完成: sub_questions={len(sub_questions)} deduped={len(deduped)} confidence={confidence}")
        return answer, confidence, deduped, sub_questions
