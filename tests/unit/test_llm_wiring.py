"""组装点接线测试（2026-09-18 重构）。

重构前 ``LLMClient`` 自读 ``settings`` / 环境变量：默认值一旦生效，「配置没接上」
会变成静默失败。现在**配置读取与客户端初始化全部收在 ``api/deps.py``**，客户端构造
参数无默认值 —— 本文件锁定的就是这条接线：

- 熔断阈值 / 冷却时长 / 默认超时 / 主模型 来自 settings；
- 备用 provider **api_key 与 base_url 都配才启用**（只配一个等于没配）；
- middleware 链按 ``LLM_TRANSPORT_MIDDLEWARES`` 组装；
- ``get_llm_client`` 是进程内单例。

不依赖真实 API（``deps.OpenAI`` 被替换为假构造器）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docs_seeker.api import deps as deps_module
from docs_seeker.core.config import settings


class _FakeSdk:
    """假 SDK 客户端：只需要有 chat.completions.create"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
                    usage=None,
                    model="fake-model",
                )
            )
        )


@pytest.fixture
def fake_sdk(monkeypatch):
    """替换 deps 里的 OpenAI 构造器，记录每次构造的 kwargs"""
    built: list[dict] = []

    def factory(**kwargs):
        built.append(kwargs)
        return _FakeSdk(**kwargs)

    monkeypatch.setattr(deps_module, "OpenAI", factory)
    return built


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    """本文件只验证「settings → 组装」这条线，显式固定相关配置，
    避免本机 .env（如恰好配了备用 provider）影响断言。"""
    monkeypatch.setattr(settings, "fallback_api_key", "")
    monkeypatch.setattr(settings, "fallback_base_url", "")
    monkeypatch.setattr(settings, "fallback_model", "deepseek-chat")
    monkeypatch.setattr(settings, "llm_transport_middlewares", "")
    yield


def test_circuit_breaker_params_come_from_settings(fake_sdk, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_circuit_failure_threshold", 9)
    monkeypatch.setattr(settings, "llm_circuit_recovery_seconds", 120)
    client = deps_module.build_llm_client()
    assert client.circuit_breaker.failure_threshold == 9
    assert client.circuit_breaker.recovery_timeout == 120


def test_default_timeout_comes_from_settings(fake_sdk, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_timeout_seconds", 33.0)
    client = deps_module.build_llm_client()
    assert client.default_timeout == 33.0


def test_primary_model_and_sdk_config_come_from_settings(fake_sdk, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "deepseek_api_key", "k-primary")
    monkeypatch.setattr(settings, "deepseek_base_url", "https://primary.example/v1")

    client = deps_module.build_llm_client()

    assert client.primary_model == "deepseek-v4-flash"
    assert fake_sdk[0] == {"api_key": "k-primary", "base_url": "https://primary.example/v1"}


def test_fallback_disabled_when_only_one_of_key_and_url_set(fake_sdk, monkeypatch) -> None:
    """api_key 与 base_url 都配才启用备用（只配一个 = 没配）"""
    monkeypatch.setattr(settings, "fallback_api_key", "fk")
    monkeypatch.setattr(settings, "fallback_base_url", "")

    client = deps_module.build_llm_client()

    assert client.fallback_client is None
    assert len(fake_sdk) == 1  # 只构造了主客户端


def test_fallback_client_is_built_and_wired(fake_sdk, monkeypatch) -> None:
    monkeypatch.setattr(settings, "fallback_api_key", "fk")
    monkeypatch.setattr(settings, "fallback_base_url", "https://fallback.example/v1")
    monkeypatch.setattr(settings, "fallback_model", "fallback-model")

    client = deps_module.build_llm_client()

    assert client.fallback_client is not None
    assert fake_sdk[1] == {"api_key": "fk", "base_url": "https://fallback.example/v1"}
    assert client.fallback_model == "fallback-model"


def test_middleware_chain_comes_from_settings(fake_sdk, monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_transport_middlewares", "observability,retry")
    client = deps_module.build_llm_client()
    assert [m.name for m in client.middlewares] == ["observability", "retry"]


def test_empty_setting_means_default_chain(fake_sdk) -> None:
    client = deps_module.build_llm_client()
    assert [m.name for m in client.middlewares] == [
        "guard",
        "observability",
        "fallback",
        "circuit_breaker",
        "budget_guard",
        "retry",
    ]


def test_get_llm_client_is_singleton(fake_sdk, monkeypatch) -> None:
    monkeypatch.setattr(deps_module, "_llm_client", None)
    first = deps_module.get_llm_client()
    second = deps_module.get_llm_client()
    assert first is second
    assert len(fake_sdk) == 1  # 只构造一次
    monkeypatch.setattr(deps_module, "_llm_client", None)  # 不污染其他用例


def test_llm_client_requires_all_dependencies() -> None:
    """构造参数无默认值：漏传必须直接报错，而不是悄悄用默认值跑起来"""
    from docs_seeker.infra.llm.client import LLMClient

    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        LLMClient(  # type: ignore[call-arg]
            primary_client=_FakeSdk(),
            primary_model="m",
            fallback_client=None,
            fallback_model="m",
            circuit_breaker=deps_module.CircuitBreaker(),
            # 故意漏 default_timeout 与 middlewares
        )
