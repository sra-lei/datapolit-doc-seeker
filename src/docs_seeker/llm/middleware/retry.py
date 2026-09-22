"""错误重试分类 + RetryMiddleware（退避重试）。

``is_retryable`` 也被 ``fallback`` middleware 复用（主备切换时要给失败打上
``retryable`` 标记），所以分类函数与重试 middleware 收在同一模块。
"""

from __future__ import annotations

import time

from loguru import logger

from docs_seeker.llm.middleware.base import CallNext, LLMCallContext
from docs_seeker.models.llm import LLMRequest, LLMResponse

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
                    # 记录该 provider 的原始异常：上层组装失败对象时要能归因，
                    # 不能只留一句「都失败了」。异常链由 bare raise 保留。
                    ctx.provider_errors.append((ctx.provider, e))
                    raise
                wait = 2**attempt
                logger.warning(f"重试 {attempt + 1}/{self._max_retries}，等待 {wait}s: {e}")
                time.sleep(wait)
        raise RuntimeError("重试链路异常结束")  # pragma: no cover - 循环必以 return/raise 结束
