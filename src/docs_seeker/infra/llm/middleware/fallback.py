"""FallbackMiddleware：主备模型降级。"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from docs_seeker.domain.models.llm import LLMRequest, LLMResponse
from docs_seeker.infra.llm.errors import AllModelsFailedError
from docs_seeker.infra.llm.middleware.base import CallNext, LLMCallContext
from docs_seeker.infra.llm.middleware.circuit_breaker import CircuitBreakerOpenError
from docs_seeker.infra.llm.middleware.retry import is_retryable


class FallbackMiddleware:
    """主备降级：主模型链路失败（含熔断打开）→ 换备用 provider 再走一次内层链路。

    降级不再静默：``ctx.fallback_used`` / ``ctx.provider`` 会被终端盖章到信封上。
    """

    name = "fallback"

    def __init__(self, has_fallback: Callable[[], bool]):
        self._has_fallback = has_fallback

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        try:
            return call_next(request)
        except CircuitBreakerOpenError as e:
            if not self._has_fallback():
                raise AllModelsFailedError(
                    "熔断器已打开，且无备用模型",
                    attempts=ctx.attempts,
                    provider_errors=ctx.provider_errors,
                    retryable=False,
                ) from e
            return self._switch_to_fallback(request, ctx, call_next)
        except Exception as e:
            logger.error(f"主模型调用失败: {e}")
            if not self._has_fallback():
                raise AllModelsFailedError(
                    f"主模型失败且无备用: {e}",
                    attempts=ctx.attempts,
                    provider_errors=ctx.provider_errors,
                    retryable=is_retryable(e),
                ) from e
            return self._switch_to_fallback(request, ctx, call_next)

    def _switch_to_fallback(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        ctx.provider = "fallback"
        ctx.fallback_used = True
        try:
            return call_next(request)
        except Exception as fb_e:
            logger.error(f"备用模型也失败: {fb_e}")
            ctx.provider_errors.append(("fallback", fb_e))
            raise AllModelsFailedError(
                "主模型和备用模型均失败",
                attempts=ctx.attempts,
                provider_errors=ctx.provider_errors,
                fallback_attempted=True,
                retryable=False,
            ) from fb_e
