"""docs-seeker - 使用统计存储接口

``UsageStore`` 抽象「使用统计的持久化」：读写 Redis 原语由 infra 实现
（``RedisUsageStore``），业务口径（成功判定 / 问题归一化 / TopN 聚合）由
domain 的 ``UsageTracker`` 承担。domain 只依赖本接口，不感知 Redis。
"""

from abc import ABC, abstractmethod


class UsageStore(ABC):
    """使用统计的存储原语（键由实现方统一加 ``rag:usage`` 前缀）"""

    @abstractmethod
    def incr_total(self) -> None: ...

    @abstractmethod
    def incr_success(self) -> None: ...

    @abstractmethod
    def incr_user_total(self, uid: str) -> None: ...

    @abstractmethod
    def incr_user_success(self, uid: str) -> None: ...

    @abstractmethod
    def add_user(self, uid: str) -> None: ...

    @abstractmethod
    def get_total(self) -> int: ...

    @abstractmethod
    def get_success(self) -> int: ...

    @abstractmethod
    def get_users(self) -> set[str]: ...

    @abstractmethod
    def get_user_total(self, uid: str) -> int: ...

    @abstractmethod
    def get_user_success(self, uid: str) -> int: ...

    @abstractmethod
    def incr_top(self, question: str) -> None: ...

    @abstractmethod
    def top_questions(self, limit: int) -> list[tuple[str, float]]: ...
