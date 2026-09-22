"""LLM middleware：可插拔的调用策略

骨架（洋葱链 / Context / 协议）在 ``base.py``；每个 transport middleware 一个文件：
``observability`` / ``fallback`` / ``circuit_breaker``（含熔断状态机）/
``budget_guard`` / ``retry``；护栏（协议 / 模式表 / 内置实现 / client 适配器）
内聚在 ``guards/`` 子包。
"""

from docs_seeker.llm.middleware.base import CallNext, LLMCallContext, LLMMiddleware, MiddlewareChain
from docs_seeker.llm.middleware.budget_guard import BudgetGuardMiddleware
from docs_seeker.llm.middleware.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerMiddleware,
    CircuitBreakerOpenError,
    CircuitState,
)
from docs_seeker.llm.middleware.fallback import FallbackMiddleware
from docs_seeker.llm.middleware.guards import LLMGuardMiddleware
from docs_seeker.llm.middleware.observability import ObservabilityMiddleware
from docs_seeker.llm.middleware.retry import RetryMiddleware, is_retryable

__all__ = [
    "BudgetGuardMiddleware",
    "CallNext",
    "CircuitBreaker",
    "CircuitBreakerMiddleware",
    "CircuitBreakerOpenError",
    "CircuitState",
    "FallbackMiddleware",
    "LLMCallContext",
    "LLMGuardMiddleware",
    "LLMMiddleware",
    "MiddlewareChain",
    "ObservabilityMiddleware",
    "RetryMiddleware",
    "is_retryable",
]
