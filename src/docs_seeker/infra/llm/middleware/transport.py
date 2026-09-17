"""Transport 级 middleware：观测 / 降级 / 熔断 / 重试。

这些策略原本全部内联在 ``LLMGateway.generate`` 里（重试循环、熔断计数、主备切换
写死在一个函数中），Phase 1 起拆成可插拔插件：网关只保留「组 payload → 调 SDK →
包信封」，策略由链路组合决定（见 ``LLM_TRANSPORT_MIDDLEWARES``）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.llm import LLMRequest, LLMResponse
from docs_seeker.infra.llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from docs_seeker.infra.llm.errors import AllModelsFailedError
from docs_seeker.infra.llm.middleware.base import CallNext, LLMCallContext

# 不可重试的错误类型名（4xx 里除了 429 速率限制，重试多少次都一样）
_NON_RETRYABLE_NAMES = frozenset(
    {"AuthenticationError", "BadRequestError", "NotFoundError", "PermissionDeniedError", "UnprocessableEntityError"}
)


def is_retryable(error: Exception) -> bool:
    """错误分类：鉴权 / 参数类错误不可重试，其余（超时、连接、限流、5xx）可重试。

    取不到状态码时**保持旧行为**（可重试），避免分类器误伤未知错误类型。
    """
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status != 429:
        return False
    return type(error).__name__ not in _NON_RETRYABLE_NAMES


class ObservabilityMiddleware:
    """观测参数注入：流式时开启 usage 上报（Langfuse 才能记账 token 与成本）。

    只处理**真实的 provider 参数** ``stream_options``（走 ``request.extra``）；
    langfuse drop-in 专属的 ``name`` kwarg 由网关终端注入（只有构造 client 的那层
    知道它是不是 langfuse 包装）。
    """

    name = "observability"

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        if request.stream:
            # OpenAI 会在最后一个 chunk（choices 为空）返回 usage
            request.extra.setdefault("stream_options", {"include_usage": True})
        return call_next(request)


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
                raise AllModelsFailedError("熔断器已打开，且无备用模型") from e
            return self._switch_to_fallback(request, ctx, call_next)
        except Exception as e:
            logger.error(f"主模型调用失败: {e}")
            if not self._has_fallback():
                raise AllModelsFailedError(f"主模型失败且无备用: {e}") from e
            return self._switch_to_fallback(request, ctx, call_next)

    def _switch_to_fallback(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        ctx.provider = "fallback"
        ctx.fallback_used = True
        try:
            return call_next(request)
        except Exception as fb_e:
            logger.error(f"备用模型也失败: {fb_e}")
            raise AllModelsFailedError("主模型和备用模型均失败") from fb_e


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


class BudgetGuardMiddleware:
    """预算兜底：推理模型把预算吃在 reasoning 上时，正文为空且 ``finish_reason='length'``。

    **按请求启用**（``request.meta["budget_guard"]=True``）—— 由调用方决定这次调用的
    预算语义：在「正文为空 + 截断」时用 ``LLM_RETRY_MAX_TOKENS`` 放大预算重试**一次**；
    放大预算不大于原预算时视为关闭该重试。

    生成与查询改写共用这一份实现（原先两处各抄了一遍，约 15 行 ×2）。
    兜底失败就如实返回空正文 —— 绝不把 ``reasoning_content`` 当正文、也不静默编造。
    """

    name = "budget_guard"

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        if not request.meta.get("budget_guard"):
            return call_next(request)
        resp = call_next(request)
        boosted = settings.llm_retry_max_tokens
        if resp.text or resp.finish_reason != "length" or boosted <= (request.max_tokens or 0):
            return resp
        logger.warning(
            f"正文为空且被截断（max_tokens={request.max_tokens}, finish=length）"
            f"——疑似 reasoning 吃满预算，用 max_tokens={boosted} 重试一次（name={request.name}）"
        )
        # 不污染调用方传入的 request：换一份放大预算的副本再走一次内层链路
        return call_next(replace(request, max_tokens=boosted, name=f"{request.name}-budget-retry"))


class RetryMiddleware:
    """退避重试：只重试可重试错误，指数退避（1s / 2s / 4s ...）。

    不可重试错误（鉴权、参数错误）立即抛出，不再白等退避时间。
    """

    name = "retry"

    def __init__(self, max_retries: int = 3):
        self._max_retries = max_retries

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        for attempt in range(self._max_retries + 1):
            ctx.attempts = attempt + 1
            try:
                return call_next(request)
            except Exception as e:
                if attempt >= self._max_retries or not is_retryable(e):
                    raise
                wait = 2**attempt
                logger.warning(f"重试 {attempt + 1}/{self._max_retries}，等待 {wait}s: {e}")
                time.sleep(wait)
        raise RuntimeError("重试链路异常结束")  # pragma: no cover - 循环必以 return/raise 结束
