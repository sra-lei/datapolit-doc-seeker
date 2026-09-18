"""CircuitBreakerMiddleware：熔断。"""

from __future__ import annotations

from docs_seeker.domain.models.llm import LLMRequest, LLMResponse
from docs_seeker.infra.llm.circuit_breaker import CircuitBreaker
from docs_seeker.infra.llm.middleware.base import CallNext, LLMCallContext


class CircuitBreakerMiddleware:
    """熔断：连续失败到阈值即快速失败，冷却后半开试探。

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
