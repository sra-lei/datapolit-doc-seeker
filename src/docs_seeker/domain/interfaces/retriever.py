"""docs-seeker - 检索器抽象接口"""
from abc import ABC, abstractmethod

from docs_seeker.domain.entities.chunk import Chunk


class Retriever(ABC):
    """检索器统一接口：输入查询文本，输出文档块列表"""

    @abstractmethod
    def search(self, query: str, top_k: int = 10, **kwargs) -> list[Chunk]:
        """检索

        Args:
            query: 查询文本
            top_k: 返回数量
            **kwargs: 各实现特有参数（如 filter_expr / use_summary）

        Returns:
            按相关性降序的 Chunk 列表
        """
        raise NotImplementedError
