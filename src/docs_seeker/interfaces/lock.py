"""docs-seeker - 分布式锁接口

``DistributedLock`` 抽象「跨实例互斥」能力：多副本部署时避免同一后台任务被重复执行
（如热门问题预热）。infra 提供 Redis 实现（``RedisDistributedLock``）。

约定：``acquire`` 拿不到锁返回 ``False``；**机制不可用（如 Redis 掉线）应抛异常**，
由调用方决定降级策略（可用性优先还是跳过）—— 那是业务口径，不在本接口内。
"""

from abc import ABC, abstractmethod


class DistributedLock(ABC):
    """分布式锁：按 key 互斥，带 TTL 兜底过期"""

    @abstractmethod
    def acquire(self, key: str, ttl_seconds: int) -> bool:
        """尝试获取锁：成功 True，已被占用 False（ttl_seconds 后自动过期）"""

    @abstractmethod
    def release(self, key: str) -> None:
        """释放锁（非持有者释放不应报错）"""
