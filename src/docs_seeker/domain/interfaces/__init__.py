"""领域接口（依赖倒置：上层依赖抽象，infra 提供实现）"""

from docs_seeker.domain.interfaces.cache import SemanticCachePort
from docs_seeker.domain.interfaces.embedder import EmbeddingProvider
from docs_seeker.domain.interfaces.llm import LLMProvider
from docs_seeker.domain.interfaces.lock import DistributedLock
from docs_seeker.domain.interfaces.retriever import Retriever
from docs_seeker.domain.interfaces.usage import UsageStore

__all__ = [
    "DistributedLock",
    "EmbeddingProvider",
    "LLMProvider",
    "Retriever",
    "SemanticCachePort",
    "UsageStore",
]
