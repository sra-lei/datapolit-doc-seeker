"""docs-seeker - LLM 层错误类型

设计要点（Phase 4）：**失败要可归因** —— 上层拿到的异常必须带够排障信息，
并且保留底层 provider 的原始异常链（不吞、不抹）：

- ``provider_errors``：每个 provider 的原始异常（含类型名），主备都失败时两条都在；
- ``fallback_attempted``：是否已经试过备用模型（区分"主挂了"和"主备都挂了"）；
- ``attempts``：主链路实际尝试次数（含重试）；
- ``retryable``：这次失败值不值得重试（4xx 鉴权/参数类为 False）。
"""

from __future__ import annotations


class LLMError(Exception):
    """LLM 调用失败的统一基类（保留 provider 侧原始错误与调用元信息）"""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        provider_errors: list[tuple[str, Exception]] | None = None,
        fallback_attempted: bool = False,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.provider_errors: list[tuple[str, Exception]] = list(provider_errors or [])
        self.fallback_attempted = fallback_attempted
        self.retryable = retryable

    def __str__(self) -> str:
        base = super().__str__()
        if not self.provider_errors:
            return base
        detail = "; ".join(f"{provider}: {type(err).__name__}: {err}" for provider, err in self.provider_errors)
        return f"{base}（provider 错误：{detail}）"


class AllModelsFailedError(LLMError):
    """主模型与备用模型都失败（客户端对上层暴露的统一失败类型）"""
