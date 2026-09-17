"""
docs-seeker - LLM 网关
提供：重试、超时、熔断、降级、统一调用入口
已接入 Langfuse 链路追踪：OpenAI 客户端使用 langfuse.openai 的 drop-in 包装，
自动把每次模型调用记录为 generation 观测（模型名、token 用量、耗时、错误）。

设计约定（2026-09 重构 Phase 0，方案见 ``docs/llm-gateway-guard-refactor.md``）：
- **参数透明**：常用字段过滤 ``None`` 后下发，``request.extra`` 原样合并 —— 网关不做
  provider 参数白名单，新增参数无需改网关；
- **框架参数隔离**：``name`` / ``stream_options`` 由 ``_provider_extras`` 注入，与
  provider 参数不同源（Phase 1 起由 observability middleware 接管）；
- **返回保真**：统一返回 ``LLMResponse`` 信封，``raw`` 保留原始响应，降级与重试以
  ``fallback_used`` / ``attempts`` 显式暴露，不静默。
"""

import os
import time
from enum import Enum
from threading import Lock

from dotenv import load_dotenv
from langfuse.openai import OpenAI
from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.llm import LLMProvider, LLMRequest, LLMResponse

load_dotenv()


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self._lock = Lock()

    def call(self, func, *args, **kwargs):
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.warning("熔断器进入半开状态")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError("熔断器已打开，拒绝请求")
        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info("熔断器恢复（半开→关闭）")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(f"熔断器打开！连续失败 {self.failure_count} 次")
            raise e


class CircuitBreakerOpenError(Exception):
    pass


class LLMGateway(LLMProvider):
    def __init__(self):
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

    # ---- 参数组装：唯一产出 provider payload 的地方 ----
    def _build_payload(self, request: LLMRequest, model: str) -> dict:
        """把 ``LLMRequest`` 组装成 provider payload。

        规则：
        1. 常用字段过滤 ``None`` —— ``None`` = 不下发，用服务端默认值；
        2. ``request.extra`` 原样合并 —— 任意 provider 参数可透传，网关不做白名单；
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
        payload.update(self._provider_extras(request))
        return payload

    def _provider_extras(self, request: LLMRequest) -> dict:
        """框架 / 观测参数（与 provider 参数不同源）。

        - ``name``：Langfuse generation 观测名（``langfuse.openai`` drop-in 参数）；
        - ``stream_options``：流式开启 usage 上报，否则 Langfuse 记不到 token 用量与成本。
          OpenAI 会在最后一个 chunk（choices 为空）返回 usage。

        Phase 1 起这两个键由 observability middleware 注入，网关不再关心。
        """
        extras: dict = {}
        if request.name:
            extras["name"] = request.name
        if request.stream:
            extras["stream_options"] = {"include_usage": True}
        return extras

    def generate(self, request: LLMRequest) -> LLMResponse:
        """调用 LLM，返回 ``LLMResponse`` 信封。

        ``request.model`` 非空时覆盖主模型（分层模型路由，见 LLM_GENERATE_MODEL）：
        覆盖只影响本次调用 —— 生成层走非推理模型降延迟/成本，而「判断/改写」仍用
        `LLM_MODEL` 指定的推理模型。熔断降级时同样沿用本次覆盖的模型名。

        ``request.timeout`` 非空时覆盖全局超时（判断类调用传 LLM_JUDGE_TIMEOUT_SECONDS
        快速失败走回退，避免 120s × 重试卡住调用方循环）。

        降级不再静默：走备用模型时 ``fallback_used=True``、``provider='fallback'``。
        """
        self.total_calls += 1
        if self.circuit_breaker.state == CircuitState.OPEN:
            if self.fallback_client:
                return self._try_fallback(request)
            raise AllModelsFailedError("熔断器已打开，且无备用模型")
        try:
            resp = self._call_with_retry(
                self.primary_client,
                request.model or self.primary_model,
                request,
                provider=self.primary_provider,
            )
            self.success_calls += 1
            self.circuit_breaker.failure_count = 0
            return resp
        except Exception as e:
            logger.error(f"主模型调用失败: {e}")
            if self.fallback_client:
                try:
                    resp = self._try_fallback(request)
                    self.fallback_calls += 1
                    return resp
                except Exception as fb_e:
                    logger.error(f"备用模型也失败: {fb_e}")
                    raise AllModelsFailedError("主模型和备用模型均失败") from fb_e
            raise AllModelsFailedError(f"主模型失败且无备用: {e}") from e

    def _try_fallback(self, request: LLMRequest) -> LLMResponse:
        return self._call_with_retry(
            self.fallback_client,
            request.model or self.fallback_model,
            request,
            provider="fallback",
            fallback_used=True,
        )

    def _call_with_retry(
        self,
        client,
        model,
        request: LLMRequest,
        *,
        provider: str,
        fallback_used: bool = False,
        max_retries: int = 3,
    ) -> LLMResponse:
        payload = self._build_payload(request, model)
        last_error = None
        started = time.time()
        for attempt in range(max_retries + 1):
            try:
                raw = client.chat.completions.create(**payload)
                return LLMResponse.from_raw(
                    raw,
                    stream=request.stream,
                    provider=provider,
                    fallback_used=fallback_used,
                    attempts=attempt + 1,
                    latency_ms=int((time.time() - started) * 1000),
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(f"重试 {attempt + 1}/{max_retries}，等待 {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise last_error from None
        # 循环必然以 return / raise 结束；此处兜底只为静态类型收窄
        raise last_error if last_error is not None else RuntimeError("LLM 调用未执行")

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "success_calls": self.success_calls,
            "fallback_calls": self.fallback_calls,
            "circuit_state": self.circuit_breaker.state.value,
            "circuit_failures": self.circuit_breaker.failure_count,
        }


class AllModelsFailedError(Exception):
    pass


_llm_gateway: LLMGateway | None = None


def get_llm_gateway() -> LLMGateway:
    global _llm_gateway
    if _llm_gateway is None:
        _llm_gateway = LLMGateway()
    return _llm_gateway
