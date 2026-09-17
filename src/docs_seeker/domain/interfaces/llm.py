"""docs-seeker - LLM 接口

LLM 调用的输入 / 输出契约。上层只依赖 ``LLMRequest`` / ``LLMResponse`` 两个实体，
不感知具体 provider、SDK 结构或客户端策略（重试 / 熔断 / 降级 / 护栏都藏在客户端与
middleware 之后）。

设计约定（2026-09 重构 Phase 0，方案见 ``docs/llm-gateway-guard-refactor.md``）：
- 参数**不做白名单**：未知 provider 参数走 ``extra`` 原样透传；
- 返回**不降维**：``raw`` 保留原始响应，便捷字段只是视图；
- 调用**可归因**：降级 / 重试 / 生效的 middleware 在信封上显式可见。
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    """LLM 调用请求实体。

    三层字段，职责分明：

    - **常用 provider 参数**（``model`` / ``max_tokens`` / ``temperature`` /
      ``stream`` / ``timeout``）：保留强类型，便于发现与校验；
    - **``extra`` 逃生舱**：任意 provider 参数**原样透传**，客户端不做白名单
      （``top_p`` / ``seed`` / ``stop`` / ``response_format`` / ``tools`` /
      ``logprobs`` / ``reasoning_effort`` ...），与常用字段同名时以 ``extra``
      为准（显式覆盖）；
    - **``meta`` 框架参数**：观测标签、护栏策略等，**绝不进入 provider payload**。

    字段为 ``None`` = 不下发该参数，交给 provider 用服务端默认值。
    """

    messages: list
    # —— 常用 provider 参数 ——
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    model: str | None = None
    timeout: float | None = None
    # —— 框架参数（不进 payload）——
    name: str = "llm-call"  # Langfuse generation 观测名
    # —— 逃生舱 / 扩展位 ——
    extra: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM 调用结果信封：便捷字段 + 原始响应保真。

    - ``text`` / ``finish_reason`` / ``usage`` 等便捷字段供绝大多数调用方直接使用；
    - ``raw`` **保留 provider 原始响应对象**，需要 provider 特有字段时从它取；
    - ``provider`` / ``fallback_used`` / ``attempts`` / ``latency_ms`` /
      ``applied_middlewares`` 让「这次调用实际发生了什么」可见，便于评估归因与审计；
    - 流式调用 ``.text`` 置空（只做透传），正文用 :meth:`iter_text` 逐个产出。
    """

    text: str = ""
    raw: Any = None
    finish_reason: str | None = None
    usage: dict | None = None
    model: str | None = None
    reasoning: str | None = None
    tool_calls: list | None = None
    provider: str = ""
    fallback_used: bool = False
    attempts: int = 1
    latency_ms: int = 0
    applied_middlewares: list[str] = field(default_factory=list)
    trace_id: str | None = None
    stream: bool = False

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        stream: bool = False,
        provider: str = "",
        fallback_used: bool = False,
        attempts: int = 1,
        latency_ms: int = 0,
        trace_id: str | None = None,
    ) -> "LLMResponse":
        """包装 provider 原始响应 —— **唯一**解析 vendor 结构的地方。

        流式响应只挂 ``raw``（``.text`` 置空，聚合交给调用方）。
        结构异常不抛错：``text`` 留空、``finish_reason`` 为 ``None``，由上层按
        「空正文」的统一口径处理。
        """
        if stream:
            return cls(
                raw=raw,
                stream=True,
                provider=provider,
                fallback_used=fallback_used,
                attempts=attempts,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
        try:
            choice = raw.choices[0]
        except (AttributeError, IndexError, TypeError):
            choice = None
        text, finish_reason, reasoning, tool_calls = "", None, None, None
        if choice is not None:
            message = getattr(choice, "message", None)
            # reasoning 模型的思考内容在 message.reasoning_content，不能当正文
            text = (getattr(message, "content", None) or "").strip()
            reasoning = getattr(message, "reasoning_content", None) or None
            tool_calls = getattr(message, "tool_calls", None) or None
            finish_reason = getattr(choice, "finish_reason", None)
        return cls(
            text=text,
            raw=raw,
            finish_reason=finish_reason,
            usage=_usage_to_dict(getattr(raw, "usage", None)),
            model=getattr(raw, "model", None),
            reasoning=reasoning,
            tool_calls=tool_calls,
            provider=provider,
            fallback_used=fallback_used,
            attempts=attempts,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )

    def iter_text(self) -> Iterator[str]:
        """产出增量正文（str）。

        - 流式（``stream=True``）：从 ``raw`` chunk 流提取增量，兼容 reasoning 模型
          先出 ``reasoning_content``、以及 ``include_usage`` 的末包 choices 为空；
        - 非流式：直接产出 ``text``（为空则不产出）。
        """
        if not self.stream:
            if self.text:
                yield self.text
            return
        for chunk in self.raw or ():
            delta = _delta_text(chunk)
            if delta:
                yield delta


def _usage_to_dict(usage: Any) -> dict | None:
    """把 provider 的 usage 对象归一成 dict（取不到就返回 None）"""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    normalized = {k: getattr(usage, k, None) for k in keys}
    return normalized if any(v is not None for v in normalized.values()) else None


def _delta_text(chunk: Any) -> str:
    """从 OpenAI 风格流式 chunk 中提取增量正文（结构异常返回空串）"""
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


class LLMProvider(ABC):
    """LLM 抽象接口：答案生成 / 查询分解 / agent 推理的统一调用入口"""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """调用 LLM 生成

        Args:
            request: LLM 调用请求实体（消息、预算、温度、超时、透传参数等）

        Returns:
            LLMResponse：便捷字段 + ``raw`` 原始响应 + 调用元信息
        """
        raise NotImplementedError
