"""
docs-seeker - LLM 网关
提供：重试、超时、熔断、降级、统一调用入口
已接入 Langfuse 链路追踪：OpenAI 客户端使用 langfuse.openai 的 drop-in 包装，
自动把每次模型调用记录为 generation 观测（模型名、token 用量、耗时、错误）。
"""

import os
import time
from enum import Enum
from threading import Lock

from dotenv import load_dotenv
from langfuse.openai import OpenAI
from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.llm import LLMProvider, LLMRequest

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

    def generate(self, request: LLMRequest):
        """调用 LLM；``request.model`` 非空时覆盖主模型（分层模型路由，见 LLM_GENERATE_MODEL）。

        覆盖只影响本次调用：生成层走非推理模型降延迟/成本，而「判断/改写」仍用
        `LLM_MODEL` 指定的推理模型。熔断降级时同样沿用本次覆盖的模型名。
        ``request.timeout`` 非空时覆盖全局超时（判断类调用传 LLM_JUDGE_TIMEOUT_SECONDS
        快速失败走回退，避免 120s × 重试卡住调用方循环）。
        """
        self.total_calls += 1
        if self.circuit_breaker.state == CircuitState.OPEN:
            if self.fallback_client:
                return self._try_fallback(request)
            raise AllModelsFailedError("熔断器已打开，且无备用模型")
        try:
            result = self._call_with_retry(
                self.primary_client,
                request.model or self.primary_model,
                request,
            )
            self.success_calls += 1
            self.circuit_breaker.failure_count = 0
            return result
        except Exception as e:
            logger.error(f"主模型调用失败: {e}")
            if self.fallback_client:
                try:
                    result = self._try_fallback(request)
                    self.fallback_calls += 1
                    return result
                except Exception as fb_e:
                    logger.error(f"备用模型也失败: {fb_e}")
                    raise AllModelsFailedError("主模型和备用模型均失败") from fb_e
            raise AllModelsFailedError(f"主模型失败且无备用: {e}") from e

    def _try_fallback(self, request: LLMRequest):
        return self._call_with_retry(
            self.fallback_client,
            request.model or self.fallback_model,
            request,
        )

    def _call_with_retry(
        self,
        client,
        model,
        request: LLMRequest,
        max_retries=3,
    ):
        last_error = None
        call_kwargs: dict = {
            "model": model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
            # 推理模型响应时间波动大（实测 10~60s），超时配置化，默认 120s；
            # 判断类调用可传短超时覆盖（LLM_JUDGE_TIMEOUT_SECONDS）
            "timeout": request.timeout if request.timeout is not None else settings.llm_timeout_seconds,
            # Langfuse：为本次生成指定稳定名称（generation 观测名）
            "name": request.name,
        }
        if request.stream:
            # 流式场景开启 usage 上报，Langfuse 才能记录 token 用量与成本；
            # OpenAI 会在最后一个 chunk（choices 为空）返回 usage，_extract_delta 已兼容空 choices。
            call_kwargs["stream_options"] = {"include_usage": True}
        for attempt in range(max_retries + 1):
            try:
                return client.chat.completions.create(**call_kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(f"重试 {attempt + 1}/{max_retries}，等待 {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise last_error from None

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
