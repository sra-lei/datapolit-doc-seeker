"""LLM middleware：可插拔的调用策略

骨架（洋葱链 / Context / 协议）在 ``base.py``；每个 transport middleware 一个文件：
``observability`` / ``fallback`` / ``circuit_breaker`` / ``budget_guard`` / ``retry``；
client 内的 guard 插槽在 ``guards.py``。
"""

from docs_seeker.infra.llm.middleware.base import CallNext, LLMCallContext, LLMMiddleware, MiddlewareChain
from docs_seeker.infra.llm.middleware.budget_guard import BudgetGuardMiddleware
from docs_seeker.infra.llm.middleware.circuit_breaker import CircuitBreakerMiddleware
from docs_seeker.infra.llm.middleware.fallback import FallbackMiddleware
from docs_seeker.infra.llm.middleware.guards import LLMGuardMiddleware
from docs_seeker.infra.llm.middleware.observability import ObservabilityMiddleware
from docs_seeker.infra.llm.middleware.retry import RetryMiddleware, is_retryable

__all__ = [
    "BudgetGuardMiddleware",
    "CallNext",
    "CircuitBreakerMiddleware",
    "FallbackMiddleware",
    "LLMCallContext",
    "LLMGuardMiddleware",
    "LLMMiddleware",
    "MiddlewareChain",
    "ObservabilityMiddleware",
    "RetryMiddleware",
    "is_retryable",
]
