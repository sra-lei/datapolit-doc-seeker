"""docs-seeker - LLM 熔断器

状态机：CLOSED --(连续失败到阈值)--> OPEN --(冷却超时)--> HALF_OPEN
        HALF_OPEN --(试探成功)--> CLOSED / --(试探失败)--> OPEN

⚠️ 历史缺陷修复（Phase 1）：旧实现里网关只在成功时把 ``failure_count`` 重置为 0，
**从未累加过**、``state`` 也从未置 OPEN —— 熔断器实际上永不打开（死代码）。
Phase 1 拆 middleware 时接上真正的状态机。
"""

import time
from enum import Enum
from threading import Lock

from loguru import logger


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self._lock = Lock()

    def before_call(self) -> None:
        """调用前检查：OPEN 且未到冷却时间 → 直接拒绝；冷却已过 → 转半开放行一次"""
        with self._lock:
            if self.state != CircuitState.OPEN:
                return
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.warning("熔断器进入半开状态")
                self.state = CircuitState.HALF_OPEN
                return
            raise CircuitBreakerOpenError("熔断器已打开，拒绝请求")

    def record_success(self) -> None:
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("熔断器恢复（半开→关闭）")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            # 半开状态下试探失败 → 立即回到 OPEN，重新计冷却
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    logger.error(f"熔断器打开！连续失败 {self.failure_count} 次")
                self.state = CircuitState.OPEN

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN
