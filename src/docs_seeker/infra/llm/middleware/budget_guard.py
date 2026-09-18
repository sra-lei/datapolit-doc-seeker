"""BudgetGuardMiddleware：推理模型空正文时的预算兜底。"""

from __future__ import annotations

from dataclasses import replace

from loguru import logger

from docs_seeker.config.settings import settings
from docs_seeker.infra.llm.middleware.base import CallNext, LLMCallContext
from docs_seeker.models.llm import LLMRequest, LLMResponse


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
