"""Transport middleware 单测：重试 / 熔断 / 降级 / 观测 + 链路可插拔（Phase 1）。

这些策略原本内联在 ``LLMClient.generate`` 里，Phase 1 拆成 middleware，要求
行为等价且可插拔：

- **重试**：可重试错误退避重试，不可重试错误（4xx 非 429）立即失败；
- **熔断**：连续失败到阈值 → 快速失败（不再打 SDK）；冷却后半开试探、成功即恢复；
  （历史缺陷：旧实现 failure_count 只重置不累加，熔断器永不打开）
- **降级**：主模型失败 → 备用 provider，``fallback_used`` 在信封上可见；
- **可插拔**：``LLM_TRANSPORT_MIDDLEWARES`` 可裁剪链路。

重构（2026-09-18）后客户端不再自读配置：SDK 与 middleware 链由组装点注入，
本文件用 ``make_llm_client`` 夹具直接注入假 SDK —— 不再 monkeypatch ``OpenAI``。

不依赖真实 API。
"""
# pyright: reportArgumentType=false

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docs_seeker.config.settings import settings
from docs_seeker.infra.llm.client import build_transport_middlewares
from docs_seeker.infra.llm.errors import AllModelsFailedError
from docs_seeker.infra.llm.middleware import CircuitState
from docs_seeker.models.llm import LLMRequest

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


# ------------------------------------------------------------------ #
#  重试
# ------------------------------------------------------------------ #
def test_retry_succeeds_after_transient_failures(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink, fail_times=2))
    resp = client.generate(PROMPT)
    assert resp.text == "ok"
    assert resp.attempts == 3
    assert len(sink) == 3


def test_non_retryable_error_fails_fast(make_llm_client) -> None:
    """鉴权 / 参数错误不该退避重试（旧实现一律重试 3 次，白等 7 秒）"""
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink, fail_times=99, error_factory=_AuthError))
    with pytest.raises(AllModelsFailedError):
        client.generate(PROMPT)
    assert len(sink) == 1  # 没有重试


# ------------------------------------------------------------------ #
#  熔断
# ------------------------------------------------------------------ #
def test_circuit_breaker_opens_and_fails_fast(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink, fail_times=99))
    client.circuit_breaker.failure_threshold = 2

    for _ in range(2):  # 两次逻辑调用（内部各重试 4 次）
        with pytest.raises(AllModelsFailedError):
            client.generate(PROMPT)
    assert client.circuit_breaker.state is CircuitState.OPEN
    calls_before = len(sink)

    with pytest.raises(AllModelsFailedError):
        client.generate(PROMPT)
    assert len(sink) == calls_before  # 熔断打开：直接拒绝，没有再打 SDK


def test_circuit_breaker_half_open_recovers(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink, fail_times=99))
    client.circuit_breaker.failure_threshold = 1
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
def test_fallback_switches_provider_and_marks_envelope(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(
        _FakeOpenAI(sink, fail_times=99),
        fallback_client=_FakeOpenAI(sink),
        fallback_model="fallback-model",
    )

    resp = client.generate(PROMPT)

    assert resp.provider == "fallback"
    assert resp.fallback_used is True
    assert sink[-1]["model"] == "fallback-model"  # 降级用备用模型
    assert client.stats["fallback_calls"] == 1
    assert client.stats["success_calls"] == 0  # 主模型没成功过


# ------------------------------------------------------------------ #
#  可插拔
# ------------------------------------------------------------------ #
def test_default_chain_order(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink))
    expected = ["guard", "observability", "fallback", "circuit_breaker", "budget_guard", "retry"]
    assert [m.name for m in client.middlewares] == expected
    resp = client.generate(PROMPT)
    assert resp.applied_middlewares == expected


def test_chain_can_be_trimmed_by_settings(make_llm_client) -> None:
    """``LLM_TRANSPORT_MIDDLEWARES`` 可裁剪策略链（组装点把配置名传给组装函数）"""
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink, fail_times=99), middleware_names="observability")
    assert [m.name for m in client.middlewares] == ["observability"]

    with pytest.raises(TimeoutError):  # 无 retry / fallback：原始错误直接冒泡
        client.generate(PROMPT)
    assert len(sink) == 1


def test_middlewares_can_be_injected_explicitly(make_llm_client) -> None:
    from docs_seeker.infra.llm.middleware import ObservabilityMiddleware

    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink), middlewares=[ObservabilityMiddleware()])
    assert [m.name for m in client.middlewares] == ["observability"]


def test_unknown_middleware_name_is_ignored(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink), middleware_names="observability,nope")
    assert [m.name for m in client.middlewares] == ["observability"]


def test_build_transport_middlewares_is_pure(make_llm_client) -> None:
    """组装函数不读 settings：全部依赖由入参决定（配置读取在组装点）"""
    from docs_seeker.infra.llm.middleware import CircuitBreaker
    from docs_seeker.services.guards import get_guard_chain

    breaker = CircuitBreaker(failure_threshold=3)
    chain = build_transport_middlewares(
        guard_chain=get_guard_chain(),
        breaker=breaker,
        has_fallback=lambda: True,
        names="circuit_breaker,fallback",
    )
    assert [m.name for m in chain] == ["circuit_breaker", "fallback"]


def test_observability_injects_stream_options_only_when_streaming(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink))
    client.generate(PROMPT)
    assert "stream_options" not in sink[-1]

    client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], stream=True))
    assert sink[-1]["stream_options"] == {"include_usage": True}


def test_empty_middleware_names_means_default_chain(make_llm_client) -> None:
    """空配置 = 默认全链（与生产 settings 默认口径一致）"""
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink), middleware_names="")
    assert [m.name for m in client.middlewares] == [
        "guard",
        "observability",
        "fallback",
        "circuit_breaker",
        "budget_guard",
        "retry",
    ]
    assert settings.llm_transport_middlewares == ""  # 生产默认口径
