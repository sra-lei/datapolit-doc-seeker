"""Transport middleware 单测：重试 / 熔断 / 降级 / 观测 + 链路可插拔（Phase 1）。

这些策略原本内联在 ``LLMClient.generate`` 里，Phase 1 拆成 middleware，要求
行为等价且可插拔：

- **重试**：可重试错误退避重试，不可重试错误（4xx 非 429）立即失败；
- **熔断**：连续失败到阈值 → 快速失败（不再打 SDK）；冷却后半开试探、成功即恢复；
  （历史缺陷：旧实现 failure_count 只重置不累加，熔断器永不打开）
- **降级**：主模型失败 → 备用 provider，``fallback_used`` 在信封上可见；
- **可插拔**：``LLM_TRANSPORT_MIDDLEWARES`` 可裁剪链路。

不依赖真实 API：假 OpenAI 客户端只记录调用参数。
"""
# pyright: reportArgumentType=false

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docs_seeker.core.config import settings
from docs_seeker.domain.models.llm import LLMRequest
from docs_seeker.infra.llm import client as client_module
from docs_seeker.infra.llm.circuit_breaker import CircuitState
from docs_seeker.infra.llm.errors import AllModelsFailedError

PROMPT = LLMRequest(messages=[{"role": "user", "content": "x"}])


class _AuthError(Exception):
    """不可重试错误（模拟 401 鉴权失败）"""

    def __init__(self):
        super().__init__("invalid api key")
        self.status_code = 401


def _raw_response(content: str = "ok", finish: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=None,
        model="fake-model",
    )


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
                return _raw_response()

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def install(monkeypatch):
    """安装假客户端；返回 (sink, 构造客户端的工厂)"""
    sink: list[dict] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)  # 跳过退避等待
    monkeypatch.setattr(client_module, "OpenAI", lambda **kwargs: _FakeOpenAI(sink))
    return sink


@pytest.fixture
def install_multi(monkeypatch):
    """按构造顺序返回不同客户端（用于主/备双客户端场景）"""
    sink: list[dict] = []
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    clients: list[_FakeOpenAI] = []

    def factory(**kwargs):
        client = _FakeOpenAI(sink)
        clients.append(client)
        clients[0].fail_times = 99  # 第一个构造 = 主客户端：永远失败
        return client

    monkeypatch.setattr(client_module, "OpenAI", factory)
    return sink, clients


# ------------------------------------------------------------------ #
#  重试
# ------------------------------------------------------------------ #
def test_retry_succeeds_after_transient_failures(install) -> None:
    client = client_module.LLMClient()
    client.primary_client.fail_times = 2
    resp = client.generate(PROMPT)
    assert resp.text == "ok"
    assert resp.attempts == 3
    assert len(install) == 3


def test_non_retryable_error_fails_fast(install, monkeypatch) -> None:
    """鉴权 / 参数错误不该退避重试（旧实现一律重试 3 次，白等 7 秒）"""
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("FALLBACK_BASE_URL", raising=False)
    sink = install
    sink.clear()
    client = client_module.LLMClient()
    client.primary_client.fail_times = 99
    client.primary_client._error_factory = _AuthError
    with pytest.raises(AllModelsFailedError):
        client.generate(PROMPT)
    assert len(sink) == 1  # 没有重试


# ------------------------------------------------------------------ #
#  熔断
# ------------------------------------------------------------------ #
def test_circuit_breaker_opens_and_fails_fast(install, monkeypatch) -> None:
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("FALLBACK_BASE_URL", raising=False)
    sink = install
    sink.clear()
    client = client_module.LLMClient()
    client.circuit_breaker.failure_threshold = 2
    client.primary_client.fail_times = 99

    for _ in range(2):  # 两次逻辑调用（内部各重试 4 次）
        with pytest.raises(AllModelsFailedError):
            client.generate(PROMPT)
    assert client.circuit_breaker.state is CircuitState.OPEN
    calls_before = len(sink)

    with pytest.raises(AllModelsFailedError):
        client.generate(PROMPT)
    assert len(sink) == calls_before  # 熔断打开：直接拒绝，没有再打 SDK


def test_circuit_breaker_half_open_recovers(install, monkeypatch) -> None:
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("FALLBACK_BASE_URL", raising=False)
    sink = install
    sink.clear()
    client = client_module.LLMClient()
    client.circuit_breaker.failure_threshold = 1
    client.primary_client.fail_times = 99
    with pytest.raises(AllModelsFailedError):
        client.generate(PROMPT)
    assert client.circuit_breaker.state is CircuitState.OPEN

    # 冷却期已过 + 服务恢复 → 半开试探成功 → 关闭
    client.circuit_breaker.last_failure_time = 0
    client.primary_client.fail_times = 0
    resp = client.generate(PROMPT)
    assert resp.text == "ok"
    assert client.circuit_breaker.state is CircuitState.CLOSED
    assert client.circuit_breaker.failure_count == 0


# ------------------------------------------------------------------ #
#  降级
# ------------------------------------------------------------------ #
def test_fallback_switches_provider_and_marks_envelope(install_multi, monkeypatch) -> None:
    monkeypatch.setenv("FALLBACK_API_KEY", "fk")
    monkeypatch.setenv("FALLBACK_BASE_URL", "https://fallback.example/v1")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model")
    sink, clients = install_multi
    client = client_module.LLMClient()

    resp = client.generate(PROMPT)

    assert resp.provider == "fallback"
    assert resp.fallback_used is True
    assert sink[-1]["model"] == "fallback-model"  # 降级用备用模型
    assert client.stats["fallback_calls"] == 1
    assert client.stats["success_calls"] == 0  # 主模型没成功过


# ------------------------------------------------------------------ #
#  熔断参数配置化
# ------------------------------------------------------------------ #
def test_circuit_breaker_params_come_from_settings(install, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_circuit_failure_threshold", 9)
    monkeypatch.setattr(settings, "llm_circuit_recovery_seconds", 120)
    client = client_module.LLMClient()
    assert client.circuit_breaker.failure_threshold == 9
    assert client.circuit_breaker.recovery_timeout == 120


# ------------------------------------------------------------------ #
#  可插拔
# ------------------------------------------------------------------ #
def test_default_chain_order(install) -> None:
    client = client_module.LLMClient()
    expected = ["guard", "observability", "fallback", "circuit_breaker", "budget_guard", "retry"]
    assert [m.name for m in client.middlewares] == expected
    resp = client.generate(PROMPT)
    assert resp.applied_middlewares == expected


def test_chain_can_be_trimmed_by_settings(install, monkeypatch) -> None:
    """LLM_TRANSPORT_MIDDLEWARES 可裁剪策略链（真可插拔）"""
    monkeypatch.setattr(settings, "llm_transport_middlewares", "observability")
    sink = install
    sink.clear()
    client = client_module.LLMClient()
    assert [m.name for m in client.middlewares] == ["observability"]

    client.primary_client.fail_times = 99
    with pytest.raises(TimeoutError):  # 无 retry / fallback：原始错误直接冒泡
        client.generate(PROMPT)
    assert len(sink) == 1


def test_middlewares_can_be_injected_explicitly(install) -> None:
    from docs_seeker.infra.llm.middleware import ObservabilityMiddleware

    client = client_module.LLMClient(middlewares=[ObservabilityMiddleware()])
    assert [m.name for m in client.middlewares] == ["observability"]


def test_observability_injects_stream_options_only_when_streaming(install) -> None:
    client = client_module.LLMClient()
    client.generate(PROMPT)
    assert "stream_options" not in install[-1]

    client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], stream=True))
    assert install[-1]["stream_options"] == {"include_usage": True}
