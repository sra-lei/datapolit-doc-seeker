"""ObservabilityMiddleware：观测参数注入。"""

from __future__ import annotations

from docs_seeker.llm.middleware.base import CallNext, LLMCallContext
from docs_seeker.models.llm import LLMRequest, LLMResponse


class ObservabilityMiddleware:
    """观测参数注入：流式时开启 usage 上报（Langfuse 才能记账 token 与成本）。

    只处理**真实的 provider 参数** ``stream_options``（走 ``request.extra``）；
    langfuse drop-in 专属的 ``name`` kwarg 由客户端终端注入（只有构造 client 的那层
    知道它是不是 langfuse 包装）。
    """

    name = "observability"

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        if request.stream:
            # OpenAI 会在最后一个 chunk（choices 为空）返回 usage
            request.extra.setdefault("stream_options", {"include_usage": True})
        return call_next(request)
