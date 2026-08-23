"""领域接口（依赖倒置：上层依赖抽象，infra/retrieval 提供实现）"""
from docs_seeker.domain.interfaces.embedder import EmbeddingProvider
from docs_seeker.domain.interfaces.llm import LLMProvider
from docs_seeker.domain.interfaces.retriever import Retriever

__all__ = ["EmbeddingProvider", "LLMProvider", "Retriever"]
