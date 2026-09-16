"""把 LLMGateway 返回的 OpenAI 风格响应映射为中立的 LLMMessage。

隔离 vendor 字段（DeepSeek reasoning_content、未来 native tool_calls），
runner 只依赖 LLMMessage。
"""

from __future__ import annotations

from typing import Any

from docs_seeker.agent.models import LLMMessage


def parse_llm_response(response: Any) -> LLMMessage:
    choice = response.choices[0]
    message = choice.message
    content = (getattr(message, "content", None) or "").strip()
    reasoning = getattr(message, "reasoning_content", None)
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    tool_calls: list[dict] = []
    for tc in raw_tool_calls:
        fn = getattr(tc, "function", None)
        tool_calls.append(
            {
                "id": getattr(tc, "id", None),
                "name": getattr(fn, "name", None) if fn else None,
                "arguments": getattr(fn, "arguments", None) if fn else None,
            }
        )
    return LLMMessage(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=getattr(choice, "finish_reason", None),
    )
