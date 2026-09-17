"""docs-seeker - 语义缓存接口

``SemanticCachePort`` 抽象「语义缓存」能力：命中查询 / 写入 / 清空 / 统计。
infra 的 ``SemanticCache``（向量化 + Milvus）实现本接口；domain 只依赖本接口，
不感知向量化与存储细节。
"""

from abc import ABC, abstractmethod


class SemanticCachePort(ABC):
    """语义缓存抽象：question → 缓存答案（含来源）"""

    @abstractmethod
    def search(self, question: str) -> dict | None:
        """按问题查询缓存；命中返回 ``{"answer", "confidence", "sources"}``，否则 None"""

    @abstractmethod
    def store(self, question: str, result: dict) -> None:
        """写入缓存（result 为 ``{"answer", "confidence", "sources"}``）"""

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...
