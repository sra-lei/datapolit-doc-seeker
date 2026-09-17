"""LLM 失败对象可归因（Phase 4）。

失败路径最容易退化成「都失败了」一句空话 —— 本文件锁定：

- ``provider_errors`` 列出每个 provider 的**原始异常**（主备都失败时两条都在）；
- ``fallback_attempted`` 区分「主挂了」与「主备都挂了」；
- ``attempts`` 是主链路实际尝试次数（含重试）；
- ``retryable`` 反映错误性质（4xx 鉴权/参数类为 False）；
- ``__cause__`` 保留底层原始异常（历史实现用 ``from None`` 抹掉过异常链）。

不依赖真实 API。
"""
# pyright: reportArgumentType=false

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docs_seeker.domain.interfaces.llm import LLMRequest
from docs_seeker.infra.llm import gateway as gateway_module
from docs_seeker.infra.llm.errors import AllModelsFailedError, LLMError

PROMPT = LLMRequest(messages=[{"role": "user", "content": "x"}])


class _AuthError(Exception):
    """不可重试错误（模拟 401）"""

    def __init__(self):
        super().__init__("invalid api key")
        self.status_code = 401


class _FakeOpenAI:
    def __init__(self, sink: list[dict], fail_times: int = 0, error_factory=None):
        outer = self
        self.sink = sink
        self.fail_times = fail_times
        self._error_factory = error_factory

        class _Completions:
            def create(self, **kwargs):
                outer.sink.append(kwargs)
                if len(outer.sink) <= outer.fail_times:
                    raise outer._error_factory() if outer._error_factory else TimeoutError("simulated timeout")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
                    usage=None,
                    model="fake-model",
                )

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def no_fallback(monkeypatch):
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("FALLBACK_BASE_URL", raising=False)
    monkeypatch.setattr(gateway_module.time, "sleep", lambda _s: None)


def _primary_only_gateway(monkeypatch, error_factory=None) -> gateway_module.LLMGateway:
    sink: list[dict] = []
    monkeypatch.setattr(gateway_module, "OpenAI", lambda **kwargs: _FakeOpenAI(sink, fail_times=99, error_factory=error_factory))
    return gateway_module.LLMGateway()


# ------------------------------------------------------------------ #
#  主模型失败（无备用）
# ------------------------------------------------------------------ #
def test_primary_only_failure_is_attributable(no_fallback, monkeypatch) -> None:
    gw = _primary_only_gateway(monkeypatch)
    with pytest.raises(AllModelsFailedError) as excinfo:
        gw.generate(PROMPT)

    err = excinfo.value
    assert isinstance(err, LLMError)
    assert err.attempts == 4  # 首次 + 3 次重试
    assert err.fallback_attempted is False
    assert err.provider_errors and err.provider_errors[0][0] == "primary"
    assert isinstance(err.provider_errors[0][1], TimeoutError)
    assert err.retryable is True


def test_error_chain_is_preserved(no_fallback, monkeypatch) -> None:
    """底层原始异常必须能被追踪到（历史实现曾用 `raise ... from None` 抹掉）"""
    gw = _primary_only_gateway(monkeypatch)
    with pytest.raises(AllModelsFailedError) as excinfo:
        gw.generate(PROMPT)
    assert isinstance(excinfo.value.__cause__, TimeoutError)


def test_message_carries_provider_detail(no_fallback, monkeypatch) -> None:
    gw = _primary_only_gateway(monkeypatch)
    with pytest.raises(AllModelsFailedError) as excinfo:
        gw.generate(PROMPT)
    text = str(excinfo.value)
    assert "primary" in text and "TimeoutError" in text


# ------------------------------------------------------------------ #
#  不可重试错误
# ------------------------------------------------------------------ #
def test_non_retryable_failure_is_marked(no_fallback, monkeypatch) -> None:
    gw = _primary_only_gateway(monkeypatch, error_factory=_AuthError)
    with pytest.raises(AllModelsFailedError) as excinfo:
        gw.generate(PROMPT)
    err = excinfo.value
    assert err.retryable is False
    assert err.attempts == 1  # 鉴权错误不重试


# ------------------------------------------------------------------ #
#  主备都失败
# ------------------------------------------------------------------ #
def test_both_providers_failure_lists_both_errors(monkeypatch) -> None:
    monkeypatch.setenv("FALLBACK_API_KEY", "fk")
    monkeypatch.setenv("FALLBACK_BASE_URL", "https://fallback.example/v1")
    monkeypatch.setattr(gateway_module.time, "sleep", lambda _s: None)

    sink: list[dict] = []

    def factory(**kwargs):
        return _FakeOpenAI(sink, fail_times=99)  # 主、备都永远失败

    monkeypatch.setattr(gateway_module, "OpenAI", factory)
    gw = gateway_module.LLMGateway()
    with pytest.raises(AllModelsFailedError) as excinfo:
        gw.generate(PROMPT)

    err = excinfo.value
    providers = [name for name, _ in err.provider_errors]
    assert "primary" in providers and "fallback" in providers
    assert err.fallback_attempted is True
    assert err.retryable is False
    assert isinstance(err.__cause__, TimeoutError)  # 备用模型的失败是直接起因


def test_open_circuit_without_fallback_is_marked(no_fallback, monkeypatch) -> None:
    gw = _primary_only_gateway(monkeypatch)
    gw.circuit_breaker.failure_threshold = 1
    with pytest.raises(AllModelsFailedError):
        gw.generate(PROMPT)  # 先把熔断器打开

    with pytest.raises(AllModelsFailedError) as excinfo:
        gw.generate(PROMPT)
    err = excinfo.value
    assert err.retryable is False
    assert "熔断" in str(err)
    assert isinstance(err.__cause__, gateway_module.CircuitBreakerOpenError)
