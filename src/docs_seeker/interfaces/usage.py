"""docs-seeker - 使用统计存储接口

``UsageStore`` 抽象「使用统计的持久化」，方法为**业务操作**而非存储原语：
- 写：记录一次调用（成功与否由调用方判定后传入）、记录一次提问；
- 读：热门问题 TopN、调用聚合原始数。

存储实现（infra 的 ``RedisUsageStore``）内部处理键结构与命令；聚合的
**业务口径**（成功率格式 / 排序 / 截断）由 domain 的 ``UsageTracker`` 承担。
domain 只依赖本接口，不感知 Redis。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CallStats:
    """调用统计的原始数据（供 UsageTracker 做业务口径聚合）"""

    total: int
    success: int
    users: list["UserCalls"]


@dataclass
class UserCalls:
    """单个用户的调用计数"""

    user_id: str
    total: int
    success: int


class UsageStore(ABC):
    """使用统计存储：业务操作级接口"""

    @abstractmethod
    def record_call(self, uid: str, ok: bool) -> None:
        """记录一次 RAG 调用（uid 已归一化；ok 为是否成功，由调用方判定）"""

    @abstractmethod
    def record_question(self, question: str) -> None:
        """记录一次提问（question 已归一化）"""

    @abstractmethod
    def top_questions(self, limit: int) -> list[tuple[str, int]]:
        """热门问题 TopN：[(question, count), ...] 按次数降序"""

    @abstractmethod
    def call_stats(self) -> CallStats:
        """调用统计原始数（含用户明细，供上层聚合）"""
