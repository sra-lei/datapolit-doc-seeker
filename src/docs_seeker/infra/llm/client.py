"""
docs-seeker - LLM 客户端
统一调用入口：客户端本身只做三件事 —— **组 payload → 调 SDK → 包信封**；
重试 / 熔断 / 降级 / 观测等策略全部下沉为可插拔 middleware。

⚠️ **本模块不读配置、不构造 SDK 客户端**：SDK 客户端、模型名、熔断器、默认超时、
middleware 链**全部由组装点显式传入**（``docs_seeker/api/deps.py::build_llm_client``），
构造参数一律**无默认值** —— 缺参数直接 TypeError，不让「默认值悄悄生效」把
「配置没接上」变成静默失败。

设计约定（方案见 ``docs/llm-gateway-guard-refactor.md``）：
- **参数透明**：常用字段过滤 ``None`` 后下发，``request.extra`` 原样合并 —— 不做
  provider 参数白名单，新增参数无需改客户端；
- **框架参数隔离**：langfuse 专属的 ``name`` 由 ``_framework_params`` 注入，与
  provider 参数不同源；``request.meta`` 永不下发；
- **返回保真**：统一返回 ``LLMResponse`` 信封，``raw`` 保留原始响应，降级与重试以
  ``fallback_used`` / ``attempts`` / ``applied_middlewares`` 显式暴露，不静默。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from docs_seeker.domain.interfaces.llm import LLMProvider
from docs_seeker.domain.models.llm import LLMRequest, LLMResponse
from docs_seeker.domain.services.guards.base import GuardChain
from docs_seeker.infra.llm.errors import AllModelsFailedError, LLMError
from docs_seeker.infra.llm.middleware import (
    BudgetGuardMiddleware,
    CircuitBreaker,
    CircuitBreakerMiddleware,
    CircuitBreakerOpenError,
    CircuitState,
    FallbackMiddleware,
    LLMCallContext,
    LLMGuardMiddleware,
    LLMMiddleware,
    MiddlewareChain,
    ObservabilityMiddleware,
    RetryMiddleware,
)

if TYPE_CHECKING:  # 仅用于类型标注：真实构造在组装点（api/deps.py）
    from langfuse.openai import OpenAI

__all__ = [
    "AllModelsFailedError",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "LLMError",
    "LLMClient",
    "build_transport_middlewares",
]


class LLMClient(LLMProvider):
    """LLM 客户端：透明传输 + 可插拔策略链（依赖全部由组装点注入）。"""

    def __init__(
        self,
        *,
        primary_client: OpenAI,
        primary_model: str,
        fallback_client: OpenAI | None,
        fallback_model: str,
        circuit_breaker: CircuitBreaker,
        default_timeout: float,
        middlewares: list[LLMMiddleware],
    ) -> None:
        self.primary_client = primary_client
        self.primary_model = primary_model
        self.primary_provider = "primary"
        self.fallback_client = fallback_client
        self.fallback_model = fallback_model
        self.circuit_breaker = circuit_breaker
        self.default_timeout = default_timeout
        self.total_calls = 0
        self.success_calls = 0
        self.fallback_calls = 0
        self.middlewares = list(middlewares)
        self._chain = MiddlewareChain(self.middlewares)

    # ---- 对外入口 ----
    def generate(self, request: LLMRequest) -> LLMResponse:
        """调用 LLM，返回 ``LLMResponse`` 信封。

        策略由 middleware 链决定（观测 / 降级 / 熔断 / 重试），客户端不再内联。

        ``request.model`` 非空时覆盖主模型（分层模型路由，见 LLM_GENERATE_MODEL）：
        覆盖只影响本次调用 —— 生成层走非推理模型降延迟/成本，而「判断/改写」仍用
        `LLM_MODEL` 指定的推理模型；降级时沿用本次覆盖的模型名。

        ``request.timeout`` 非空时覆盖默认超时（判断类调用传 LLM_JUDGE_TIMEOUT_SECONDS
        快速失败走回退，避免 120s × 重试卡住调用方循环）。
        """
        self.total_calls += 1
        ctx = LLMCallContext(started_at=time.time())
        resp = self._chain.run(request, ctx, lambda req: self._invoke(req, ctx))
        # 策略执行结果盖章到信封（降级不再静默）
        resp.provider = ctx.provider
        resp.fallback_used = ctx.fallback_used
        resp.applied_middlewares = list(ctx.applied_middlewares)
        if ctx.fallback_used:
            self.fallback_calls += 1
        else:
            self.success_calls += 1
        return resp

    # ---- 链路最内层：真正发起 SDK 调用 ----
    def _invoke(self, request: LLMRequest, ctx: LLMCallContext) -> LLMResponse:
        if ctx.provider == "fallback":
            client, default_model = self.fallback_client, self.fallback_model
        else:
            client, default_model = self.primary_client, self.primary_model
        if client is None:  # 理论不可达：fallback middleware 只在备用客户端存在时才切 provider
            raise AllModelsFailedError("备用模型未配置")
        payload = self._build_payload(request, request.model or default_model)
        raw = client.chat.completions.create(**payload)
        return LLMResponse.from_raw(
            raw,
            stream=request.stream,
            provider=ctx.provider,
            fallback_used=ctx.fallback_used,
            attempts=ctx.attempts or 1,
            latency_ms=int((time.time() - ctx.started_at) * 1000),
        )

    # ---- 参数组装：唯一产出 provider payload 的地方 ----
    def _build_payload(self, request: LLMRequest, model: str) -> dict:
        """把 ``LLMRequest`` 组装成 provider payload。

        规则：
        1. 常用字段过滤 ``None`` —— ``None`` = 不下发，用服务端默认值；
        2. ``request.extra`` 原样合并 —— 任意 provider 参数可透传（含 middleware 注入的
           ``stream_options``），客户端不做白名单；
        3. ``request.meta`` **永不进入 payload**（框架参数不污染 provider 参数）。
        """
        payload: dict = {
            "model": model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
            # 推理模型响应时间波动大（实测 10~60s），默认超时由组装点从配置注入；
            # 判断类调用可传短超时覆盖（LLM_JUDGE_TIMEOUT_SECONDS）
            "timeout": request.timeout if request.timeout is not None else self.default_timeout,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        payload.update(request.extra or {})
        payload.update(self._framework_params(request))
        return payload

    def _framework_params(self, request: LLMRequest) -> dict:
        """langfuse.openai drop-in 专属参数（**非** OpenAI 参数）。

        ``name`` = generation 观测名。只有构造 client 的这一层知道它是不是 langfuse
        包装，所以在这里注入；provider 参数走 ``extra``，两者分源。
        """
        return {"name": request.name} if request.name else {}

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "success_calls": self.success_calls,
            "fallback_calls": self.fallback_calls,
            "circuit_state": self.circuit_breaker.state.value,
            "circuit_failures": self.circuit_breaker.failure_count,
            "middlewares": [m.name for m in self.middlewares],
        }


def build_transport_middlewares(
    *,
    guard_chain: GuardChain,
    breaker: CircuitBreaker,
    has_fallback: Callable[[], bool],
    names: str,
) -> list[LLMMiddleware]:
    """按配置名组装 transport 链（列表顺序 = 外层到内层）。

    这是**纯组装函数**：依赖全部由调用方传入（组装点读配置后在这里落地），
    本身不读 ``settings``。``names`` 为逗号分隔的 middleware 名，空 = 默认全链。

    ``guard``（扫 messages，仅告警）→ ``observability``（观测参数）→ ``fallback``
    （主备降级）→ ``circuit_breaker``（每个逻辑调用记一次成败）→ ``budget_guard``
    （截断空正文放大预算重试一次）→ ``retry``（同 provider 内退避重试）。

    - ``guard`` 在最外层：agent 循环内的每次 LLM 调用都先过护栏（只告警不短路）；
    - ``budget_guard`` 排在 ``retry`` **外层**：放大预算那次调用仍享受错误重试。

    可用 ``LLM_TRANSPORT_MIDDLEWARES`` 覆盖（逗号分隔的名字，空 = 默认链）：
    例 ``LLM_TRANSPORT_MIDDLEWARES=observability,retry`` 即关掉护栏/降级/熔断/预算兜底。
    """
    registry: dict[str, Callable[[], LLMMiddleware]] = {
        "guard": lambda: LLMGuardMiddleware(guard_chain),
        "observability": ObservabilityMiddleware,
        "fallback": lambda: FallbackMiddleware(has_fallback),
        "circuit_breaker": lambda: CircuitBreakerMiddleware(breaker),
        "budget_guard": BudgetGuardMiddleware,
        "retry": RetryMiddleware,
    }
    wanted = [n.strip() for n in (names or "").split(",") if n.strip()]
    if not wanted:
        wanted = list(registry)
    unknown = [n for n in wanted if n not in registry]
    if unknown:
        logger.warning(f"未知的 LLM transport middleware {unknown}，已忽略")
    return [registry[n]() for n in wanted if n in registry]
