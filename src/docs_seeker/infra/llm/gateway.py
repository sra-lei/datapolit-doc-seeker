"""
docs-seeker - LLM 网关
统一调用入口：网关本身只做三件事 —— **组 payload → 调 SDK → 包信封**；
重试 / 熔断 / 降级 / 观测等策略全部下沉为可插拔 middleware（Phase 1）。

已接入 Langfuse 链路追踪：OpenAI 客户端使用 langfuse.openai 的 drop-in 包装，
自动把每次模型调用记录为 generation 观测（模型名、token 用量、耗时、错误）。

设计约定（方案见 ``docs/llm-gateway-guard-refactor.md``）：
- **参数透明**：常用字段过滤 ``None`` 后下发，``request.extra`` 原样合并 —— 不做
  provider 参数白名单，新增参数无需改网关；
- **框架参数隔离**：langfuse 专属的 ``name`` 由 ``_framework_params`` 注入，与
  provider 参数不同源；``request.meta`` 永不下发；
- **返回保真**：统一返回 ``LLMResponse`` 信封，``raw`` 保留原始响应，降级与重试以
  ``fallback_used`` / ``attempts`` / ``applied_middlewares`` 显式暴露，不静默。
"""

import os
import time
from collections.abc import Callable

from dotenv import load_dotenv
from langfuse.openai import OpenAI
from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.llm import LLMProvider, LLMRequest, LLMResponse
from docs_seeker.infra.llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState
from docs_seeker.infra.llm.errors import AllModelsFailedError
from docs_seeker.infra.llm.middleware import (
    CircuitBreakerMiddleware,
    FallbackMiddleware,
    LLMCallContext,
    LLMMiddleware,
    MiddlewareChain,
    ObservabilityMiddleware,
    RetryMiddleware,
)

load_dotenv()

__all__ = [
    "AllModelsFailedError",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "LLMGateway",
    "default_transport_middlewares",
    "get_llm_gateway",
]


class LLMGateway(LLMProvider):
    """LLM 网关：透明传输 + 可插拔策略链。"""

    def __init__(self, middlewares: list[LLMMiddleware] | None = None):
        self.primary_client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
        self.primary_model = settings.llm_model
        self.primary_provider = "primary"
        self.fallback_client = None
        fk = os.getenv("FALLBACK_API_KEY", "")
        fu = os.getenv("FALLBACK_BASE_URL", "")
        if fk and fu:
            self.fallback_client = OpenAI(api_key=fk, base_url=fu)
        self.fallback_model = os.getenv("FALLBACK_MODEL", "deepseek-chat")
        self.circuit_breaker = CircuitBreaker()
        self.total_calls = 0
        self.success_calls = 0
        self.fallback_calls = 0
        # 传入 middlewares 可完全自定义链路（测试 / 特殊部署）；缺省走默认链
        self.middlewares = list(middlewares) if middlewares is not None else default_transport_middlewares(self)
        self._chain = MiddlewareChain(self.middlewares)

    # ---- 对外入口 ----
    def generate(self, request: LLMRequest) -> LLMResponse:
        """调用 LLM，返回 ``LLMResponse`` 信封。

        策略由 middleware 链决定（观测 / 降级 / 熔断 / 重试），网关不再内联。

        ``request.model`` 非空时覆盖主模型（分层模型路由，见 LLM_GENERATE_MODEL）：
        覆盖只影响本次调用 —— 生成层走非推理模型降延迟/成本，而「判断/改写」仍用
        `LLM_MODEL` 指定的推理模型；降级时沿用本次覆盖的模型名。

        ``request.timeout`` 非空时覆盖全局超时（判断类调用传 LLM_JUDGE_TIMEOUT_SECONDS
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
           ``stream_options``），网关不做白名单；
        3. ``request.meta`` **永不进入 payload**（框架参数不污染 provider 参数）。
        """
        payload: dict = {
            "model": model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
            # 推理模型响应时间波动大（实测 10~60s），超时配置化，默认 120s；
            # 判断类调用可传短超时覆盖（LLM_JUDGE_TIMEOUT_SECONDS）
            "timeout": request.timeout if request.timeout is not None else settings.llm_timeout_seconds,
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


def default_transport_middlewares(gateway: LLMGateway) -> list[LLMMiddleware]:
    """默认 transport 链（列表顺序 = 外层到内层）。

    ``observability``（观测参数）→ ``fallback``（主备降级）→ ``circuit_breaker``
    （每个逻辑调用记一次成败）→ ``retry``（同 provider 内退避重试）。

    可用 ``LLM_TRANSPORT_MIDDLEWARES`` 覆盖（逗号分隔的名字，空 = 默认链）：
    例 ``LLM_TRANSPORT_MIDDLEWARES=observability,retry`` 即关掉降级与熔断。
    """
    registry: dict[str, Callable[[], LLMMiddleware]] = {
        "observability": ObservabilityMiddleware,
        "fallback": lambda: FallbackMiddleware(lambda: gateway.fallback_client is not None),
        "circuit_breaker": lambda: CircuitBreakerMiddleware(gateway.circuit_breaker),
        "retry": RetryMiddleware,
    }
    names = [n.strip() for n in (settings.llm_transport_middlewares or "").split(",") if n.strip()]
    if not names:
        names = list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        logger.warning(f"未知的 LLM transport middleware {unknown}，已忽略")
    return [registry[n]() for n in names if n in registry]


_llm_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    global _llm_gateway
    if _llm_gateway is None:
        _llm_gateway = LLMGateway()
    return _llm_gateway
