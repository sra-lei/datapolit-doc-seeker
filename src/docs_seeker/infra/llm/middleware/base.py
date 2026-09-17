"""LLM 调用 middleware 骨架（洋葱模型）。

一次逻辑调用 = 一串 middleware 依次包裹终端调用：

    observability → fallback → circuit_breaker → retry → [terminal: 组 payload + 调 SDK + 包信封]

每个 middleware 拿到 ``call_next``，**自行决定**调用几次、要不要换 provider、要不要
短路返回 —— 重试 / 熔断 / 降级 / 观测因此都可以插拔替换，网关只剩「组 payload →
调 SDK → 包信封」这一件事。

约定：
- 列表顺序 = 外层到内层（越靠前越先进入、越晚返回）；
- middleware 之间通过 :class:`LLMCallContext` 协作（``provider`` / ``attempts`` /
  ``fallback_used`` 等由 middleware 维护，终端据此盖章到 ``LLMResponse``）；
- middleware 可以改写 ``request``（如观测参数注入 ``extra``）后传给 ``call_next``，
  但**不得**丢弃 ``call_next`` 的返回值。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from docs_seeker.domain.interfaces.llm import LLMRequest, LLMResponse

# 链路里「下一层」的调用形状；终端（终端 = 真正调 SDK 的那一层）也满足它
CallNext = Callable[[LLMRequest], LLMResponse]


@dataclass
class LLMCallContext:
    """一次逻辑调用的共享状态。

    ``provider`` / ``attempts`` / ``fallback_used`` 由 middleware 维护、终端读取；
    ``applied_middlewares`` 记录**实际执行过**的 middleware，便于审计归因；
    ``provider_errors`` 收集各 provider 的原始异常，供失败时组装可归因的错误对象。
    """

    provider: str = "primary"
    attempts: int = 0
    fallback_used: bool = False
    applied_middlewares: list[str] = field(default_factory=list)
    provider_errors: list[tuple[str, Exception]] = field(default_factory=list)
    started_at: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMMiddleware(Protocol):
    """middleware 协议：``__call__(request, ctx, call_next) -> LLMResponse``"""

    name: str

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse: ...


class MiddlewareChain:
    """把 middleware 列表折成单个可调用对象。

    包装顺序与列表相反：列表第一个在最外层（最先执行、最后拿到结果）。
    """

    def __init__(self, middlewares: list[LLMMiddleware]):
        self.middlewares = list(middlewares)

    def run(self, request: LLMRequest, ctx: LLMCallContext, terminal: CallNext) -> LLMResponse:
        handler = terminal
        for middleware in reversed(self.middlewares):
            handler = _bind(middleware, ctx, handler)
        return handler(request)


def _bind(middleware: LLMMiddleware, ctx: LLMCallContext, call_next: CallNext) -> CallNext:
    def _handler(request: LLMRequest) -> LLMResponse:
        # 只记第一次（fallback 会重复进入内层，避免重复登记）
        if middleware.name not in ctx.applied_middlewares:
            ctx.applied_middlewares.append(middleware.name)
        return middleware(request, ctx, call_next)

    return _handler
