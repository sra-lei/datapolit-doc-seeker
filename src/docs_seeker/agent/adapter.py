"""把客户端返回的 ``LLMResponse`` 映射为中立的 LLMMessage。

vendor 字段（DeepSeek reasoning_content、native tool_calls）在
``LLMResponse.from_raw`` 已隔离，这里只做「信封 → LLMMessage」的形状转换，
runner 不直接碰 provider 原始结构。
"""

from __future__ import annotations

from docs_seeker.agent.models import LLMMessage
from docs_seeker.models.llm import LLMResponse


def parse_llm_response(response: LLMResponse) -> LLMMessage:
    return LLMMessage(
        content=response.text,
        reasoning=response.reasoning,
        tool_calls=_normalize_tool_calls(response.tool_calls),
        finish_reason=response.finish_reason,
    )


def _normalize_tool_calls(raw_tool_calls) -> list[dict]:
    """归一化 tool_calls（兼容对象式与 dict 式两种 provider 返回）"""
    tool_calls: list[dict] = []
    for tc in raw_tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            tool_calls.append(
                {
                    "id": tc.get("id"),
                    "name": fn.get("name") if isinstance(fn, dict) else None,
                    "arguments": fn.get("arguments") if isinstance(fn, dict) else None,
                }
            )
            continue
        fn = getattr(tc, "function", None)
        tool_calls.append(
            {
                "id": getattr(tc, "id", None),
                "name": getattr(fn, "name", None) if fn else None,
                "arguments": getattr(fn, "arguments", None) if fn else None,
            }
        )
    return tool_calls
