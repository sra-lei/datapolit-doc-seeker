"""LLM 网关单元测试（不依赖真实 API）：超时与重试口径。

背景：网关原先硬编码 `timeout=15`（普通模型口径），推理模型的响应时间随题目
波动（实测 10~60s），长答案会被直接打成超时失败 → 超时必须配置化且默认够大。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docs_seeker.core.config import settings
from docs_seeker.infra import llm as llm_pkg

gateway_module = llm_pkg.gateway


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


@pytest.fixture
def patch_client(monkeypatch):
    sink: list[dict] = []

    def _install(fail_times: int = 0):
        monkeypatch.setattr(gateway_module, "OpenAI", lambda **kwargs: _FakeOpenAI(sink, fail_times))
        return sink

    return _install


def test_gateway_uses_configured_timeout(patch_client) -> None:
    sink = patch_client()
    gw = gateway_module.LLMGateway()
    gw.generate(messages=[{"role": "user", "content": "x"}], max_tokens=100)
    assert sink[0]["timeout"] == settings.llm_timeout_seconds
    assert settings.llm_timeout_seconds >= 60  # 推理模型兜底：不得回落到 15s 这类短超时


def test_gateway_retries_then_succeeds(patch_client, monkeypatch) -> None:
    sink = patch_client(fail_times=1)
    monkeypatch.setattr(gateway_module.time, "sleep", lambda _s: None)  # 跳过退避等待
    gw = gateway_module.LLMGateway()
    gw.generate(messages=[{"role": "user", "content": "x"}], max_tokens=100)
    assert len(sink) == 2  # 首次超时 → 重试一次成功
    assert all(call["timeout"] == settings.llm_timeout_seconds for call in sink)
