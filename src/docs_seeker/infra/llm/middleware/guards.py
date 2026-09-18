"""client 内的 guard 插槽：扫描送 provider 的 messages（**仅告警**）。

本类是**守卫的客户端适配器**：把 domain 的 ``GuardChain``（领域安全规则）挂到
LLM 调用链上（middleware 机制，属 infra）。守卫规则在
``domain/services/guards``（GuardChain / builtin 模式），这里只负责
「从 ``LLMRequest`` 提取待扫文本 → 调链 → 放行」。

与 pipeline 边界那处**共用同一份护栏配置**（``get_guard_chain()``），但策略不同：

- 边界面用户输入：可拒答；
- 这里：**只检测 / 只告警，不短路** —— messages 里混着系统提示与检索证据，
  误判的代价高于收益（评审已决 2026-09-17）。

存在意义：让 agent 循环内的每一次 LLM 调用（决策 / 成文 / 判断）也经过护栏，
而不是只有最外层那次用户问答。
"""

from __future__ import annotations

from docs_seeker.domain.services.guards.base import LLM_MESSAGES_CTX, GuardChain
from docs_seeker.infra.llm.middleware.base import CallNext, LLMCallContext
from docs_seeker.models.llm import LLMRequest, LLMResponse


class LLMGuardMiddleware:
    name = "guard"

    def __init__(self, chain: GuardChain):
        self._chain = chain

    def __call__(self, request: LLMRequest, ctx: LLMCallContext, call_next: CallNext) -> LLMResponse:
        self._chain.inspect_many(self._message_texts(request), LLM_MESSAGES_CTX)
        return call_next(request)

    @staticmethod
    def _message_texts(request: LLMRequest) -> list[str]:
        """只扫非 system 消息。

        system prompt 是我们自己写的（已实测对现有模式表全部 clean），扫它只会制造
        误报；注入风险来自用户输入与**检索回来的证据正文**。
        """
        texts: list[str] = []
        for message in request.messages or []:
            if not isinstance(message, dict) or message.get("role") == "system":
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                texts.append(content)
        return texts
