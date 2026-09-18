"""熔断：状态机 + 挂载它的 middleware。

状态机：CLOSED --(连续失败到阈值)--> OPEN --(冷却超时)--> HALF_OPEN
        HALF_OPEN --(试探成功)--> CLOSED / --(试探失败)--> OPEN

⚠️ 历史缺陷修复（Phase 1）：旧实现里客户端只在成功时把 ``failure_count`` 重置为 0，
**从未累加过**、``state`` 也从未置 OPEN —— 熔断器实际上永不打开（死代码）。
Phase 1 拆 middleware 时接上真正的状态机。
"""

from __future__ import annotations

import time
from enum import Enum
from threading import Lock

from loguru import logger

from docs_seeker.infra.llm.middleware.base import CallNext, LLMCallContext
from docs_seeker.models.llm import LLMRequest, LLMResponse


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
            # 达到熔断条件（任一即可）：
            #  1) 半开状态下试探失败 → 下游未恢复，立即回到 OPEN 重新计冷却
            #     （HALF_OPEN 只放行一次试探，失败即证明还没恢复，与失败次数无关）
            #  2) 连续失败达到阈值 → CLOSED 路径的常规熔断
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                # 仅在首次转 OPEN 时打日志；已在 OPEN 时失败只刷新 last_failure_time
                # （冷却起点延后），不重复刷错误日志
                if self.state != CircuitState.OPEN:
                    logger.error(f"熔断器打开！连续失败 {self.failure_count} 次")
                self.state = CircuitState.OPEN

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


class CircuitBreakerMiddleware:
    """熔断 middleware：连续失败到阈值即快速失败，冷却后半开试探。

    放在重试**外层** —— 一个逻辑调用（可能内部重试 N 次）只记一次成败，
    不会因为内部重试把熔断器打得更快打开。
    """

    name = "circuit_breaker"

    def __init__(self, breaker: CircuitBreaker):
        self._breaker = breaker

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        self._breaker.before_call()
        try:
            resp = call_next(request)
        except Exception:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return resp
