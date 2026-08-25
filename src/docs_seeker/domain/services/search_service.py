"""docs-seeker - 纯检索用例服务"""
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.infrastructure.retrieval.composite_retriever import CompositeRetriever


class SearchService:
    """纯检索用例：不经过 LLM，返回融合后的文档列表"""

    def __init__(self, retriever: CompositeRetriever | None = None):
        # 允许注入共享 retriever（deps 单例）；缺省时自建（独立使用场景）
        self.retriever = retriever or CompositeRetriever()

    def search(self, query: str, top_k: int = 10, use_summary: bool = True) -> list[dict]:
        chunks: list[Chunk] = self.retriever.search(query, top_k=top_k, use_summary=use_summary)
        return [c.to_dict() for c in chunks]
