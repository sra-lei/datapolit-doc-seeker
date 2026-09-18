"""LLM 客户端单元测试（不依赖真实 API）：超时与重试口径。

背景：客户端原先硬编码 `timeout=15`（普通模型口径），推理模型的响应时间随题目
波动（实测 10~60s），长答案会被直接打成超时失败 → 超时必须配置化且默认够大。

重构后客户端不再自读配置：默认超时由组装点（``api/deps.py``）从 settings 注入，
本文件用 ``make_llm_client`` 夹具注入假 SDK 验证口径。
"""

from __future__ import annotations

from types import SimpleNamespace

from docs_seeker.config.settings import settings
from docs_seeker.models.llm import LLMRequest


def _response(content: str, finish: str = "stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])


class _FakeOpenAI:
    """记录 chat.completions.create 的调用参数；可指定前 N 次抛错。"""

    def __init__(self, sink: list[dict], fail_times: int = 0):
        self.sink = sink
        self.fail_times = fail_times
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.sink.append(kwargs)
                if len(outer.sink) <= outer.fail_times:
                    raise TimeoutError("simulated timeout")
                return _response("ok")

        self.chat = SimpleNamespace(completions=_Completions())


def test_client_uses_configured_timeout(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink))
    client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=100))
    assert sink[0]["timeout"] == settings.llm_timeout_seconds
    assert settings.llm_timeout_seconds >= 60  # 推理模型兜底：不得回落到 15s 这类短超时


def test_client_timeout_override_wins(make_llm_client) -> None:
    """判断类调用传短超时：覆盖全局默认，避免 agent 循环被卡死"""
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink))
    client.generate(
        LLMRequest(
            messages=[{"role": "user", "content": "x"}], max_tokens=100, timeout=settings.llm_judge_timeout_seconds
        )
    )
    assert sink[0]["timeout"] == settings.llm_judge_timeout_seconds


def test_client_retries_then_succeeds(make_llm_client) -> None:
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink, fail_times=1))
    client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=100))
    assert len(sink) == 2  # 首次超时 → 重试一次成功
    assert all(call["timeout"] == settings.llm_timeout_seconds for call in sink)


def test_explicit_timeout_is_injected_by_composition_root(make_llm_client) -> None:
    """默认超时是构造参数（不是客户端读 settings）：显式注入即生效"""
    sink: list[dict] = []
    client = make_llm_client(_FakeOpenAI(sink), default_timeout=7.5)
    client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=100))
    assert sink[0]["timeout"] == 7.5
