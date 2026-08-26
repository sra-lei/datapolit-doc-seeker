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
from docs_seeker.domain.interfaces.llm import LLMProvider

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

    def generate(self, messages: list, max_tokens: int = 600, temperature: float = 0.3, stream: bool = False, name: str = "llm-call"):
        self.total_calls += 1
        if self.circuit_breaker.state == CircuitState.OPEN:
            if self.fallback_client:
                return self._try_fallback(messages, max_tokens, temperature, stream, name)
            raise AllModelsFailedError("熔断器已打开，且无备用模型")
        try:
            result = self._call_with_retry(
                self.primary_client, self.primary_model, messages, max_tokens, temperature, stream, name
            )
            self.success_calls += 1
            self.circuit_breaker.failure_count = 0
            return result
        except Exception as e:
            logger.error(f"主模型调用失败: {e}")
            if self.fallback_client:
                try:
                    result = self._try_fallback(messages, max_tokens, temperature, stream, name)
                    self.fallback_calls += 1
                    return result
                except Exception as fb_e:
                    logger.error(f"备用模型也失败: {fb_e}")
                    raise AllModelsFailedError("主模型和备用模型均失败") from fb_e
            raise AllModelsFailedError(f"主模型失败且无备用: {e}") from e

    def _try_fallback(self, messages, max_tokens, temperature, stream, name="llm-call"):
        return self._call_with_retry(
            self.fallback_client, self.fallback_model, messages, max_tokens, temperature, stream, name
        )

    def _call_with_retry(self, client, model, messages, max_tokens, temperature, stream, name="llm-call", max_retries=3):
        last_error = None
        call_kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "timeout": 15,
            # Langfuse：为本次生成指定稳定名称（generation 观测名）
            "name": name,
        }
        if stream:
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
