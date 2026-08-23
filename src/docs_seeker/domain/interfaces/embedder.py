"""docs-seeker - 向量化接口"""
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """向量化抽象接口（查询侧）"""

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """对单条文本做向量化"""
        raise NotImplementedError
