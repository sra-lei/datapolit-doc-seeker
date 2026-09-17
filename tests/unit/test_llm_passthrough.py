"""LLM 网关参数透传与返回保真（重构 Phase 0）。

方案：``docs/llm-gateway-guard-refactor.md``。本文件锁定三条契约：

1. **参数不设白名单** —— ``extra`` 里的任意 provider 参数原样到达 SDK，同名时
   显式覆盖常用字段；``None`` 字段不下发（用服务端默认值）；
2. **框架参数隔离** —— ``meta`` / ``name`` 绝不进入 provider payload；
3. **返回不降维** —— ``raw`` 就是原始响应对象，``usage`` / ``finish_reason``
   提取到位，降级与重试以 ``fallback_used`` / ``attempts`` 显式可见。

不依赖真实 API：假 OpenAI 客户端只记录调用参数并回显原始响应。
"""
# pyright: reportArgumentType=false

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docs_seeker.domain.interfaces.llm import LLMRequest, LLMResponse
from docs_seeker.infra.llm import client as client_module


def _raw_response(content: str = "ok", finish: str = "stop", usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=usage,
        model="fake-model",
    )


def _default_raw_factory(_kwargs):
    return _raw_response()


class _FakeOpenAI:
    """记录 chat.completions.create 的调用参数；可指定前 N 次抛错。"""

    def __init__(self, sink: list[dict], fail_times: int = 0, raw_factory=None):
        outer = self
        self.sink = sink
        self.fail_times = fail_times
        self._raw_factory = raw_factory or _default_raw_factory

        class _Completions:
            def create(self, **kwargs):
                outer.sink.append(kwargs)
                if len(outer.sink) <= outer.fail_times:
                    raise TimeoutError("simulated timeout")
                return outer._raw_factory(kwargs)

        self.chat = SimpleNamespace(completions=_Completions())


def _gateway(monkeypatch, *, fail_times: int = 0, raw_factory=None) -> tuple[list[dict], client_module.LLMClient]:
    """构造接了假 OpenAI 客户端的网关；返回 (调用参数 sink, 网关)"""
    sink: list[dict] = []
    monkeypatch.setattr(client_module, "OpenAI", lambda **kwargs: _FakeOpenAI(sink, fail_times, raw_factory))
    return sink, client_module.LLMClient()


# ------------------------------------------------------------------ #
#  1. 参数透传：不做白名单
# ------------------------------------------------------------------ #
def test_extra_passthrough_reaches_sdk(monkeypatch) -> None:
    sink, gw = _gateway(monkeypatch)
    gw.generate(
        LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            max_tokens=100,
            extra={"top_p": 0.9, "seed": 42, "response_format": {"type": "json_object"}},
        )
    )
    assert sink[0]["top_p"] == 0.9
    assert sink[0]["seed"] == 42
    assert sink[0]["response_format"] == {"type": "json_object"}


def test_extra_overrides_common_field(monkeypatch) -> None:
    """同名时 extra 显式覆盖（逃生舱不该被常用字段静默吃掉）"""
    sink, gw = _gateway(monkeypatch)
    gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], temperature=0.1, extra={"temperature": 0.7}))
    assert sink[0]["temperature"] == 0.7


def test_none_fields_are_not_sent(monkeypatch) -> None:
    """None = 不下发，交给 provider 用服务端默认值"""
    sink, gw = _gateway(monkeypatch)
    gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert "max_tokens" not in sink[0]
    assert "temperature" not in sink[0]


def test_meta_never_reaches_payload(monkeypatch) -> None:
    """框架参数与 provider 参数彻底分离"""
    sink, gw = _gateway(monkeypatch)
    gw.generate(
        LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            meta={"guard_policy": "strict", "tags": ["eval"]},
        )
    )
    assert "meta" not in sink[0]
    assert "guard_policy" not in sink[0]
    assert "tags" not in sink[0]


def test_framework_name_is_observation_only(monkeypatch) -> None:
    """name 是观测名（框架参数），不是 messages 的一部分"""
    sink, gw = _gateway(monkeypatch)
    gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], name="query-decompose"))
    assert sink[0]["name"] == "query-decompose"


# ------------------------------------------------------------------ #
#  2. 返回保真
# ------------------------------------------------------------------ #
def test_response_envelope_is_returned(monkeypatch) -> None:
    _sink, gw = _gateway(monkeypatch)
    resp = gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert isinstance(resp, LLMResponse)
    assert resp.attempts == 1
    assert resp.provider == "primary"
    assert resp.fallback_used is False


def test_raw_response_is_preserved(monkeypatch) -> None:
    """raw 必须是原始响应对象本身，便捷字段只是视图"""
    holder: dict = {}

    def factory(_kwargs):
        raw = _raw_response("正文")
        holder["raw"] = raw
        return raw

    _sink, gw = _gateway(monkeypatch, raw_factory=factory)
    resp = gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert resp.raw is holder["raw"]
    assert resp.text == "正文"


def test_usage_and_finish_reason_are_extracted(monkeypatch) -> None:
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=22, total_tokens=33)
    _sink, gw = _gateway(monkeypatch, raw_factory=lambda _kw: _raw_response("  带空格的正文  ", "length", usage))
    resp = gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert resp.text == "带空格的正文"
    assert resp.finish_reason == "length"
    assert resp.usage == {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
    assert resp.model == "fake-model"


def test_empty_truncated_response_stays_empty(monkeypatch) -> None:
    """截断空正文如实返回空串 —— 不把 reasoning 当正文、不编造"""
    _sink, gw = _gateway(monkeypatch, raw_factory=lambda _kw: _raw_response("", "length"))
    resp = gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert resp.text == ""
    assert resp.finish_reason == "length"


# ------------------------------------------------------------------ #
#  3. 流式：只透传，不聚合
# ------------------------------------------------------------------ #
def test_streaming_response_keeps_raw_stream_and_empty_text(monkeypatch) -> None:
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="甲"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="乙"))]),
        # include_usage 末包：choices 为空，不得抛错
        SimpleNamespace(choices=[]),
    ]
    _sink, gw = _gateway(monkeypatch, raw_factory=lambda _kw: iter(chunks))
    resp = gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], stream=True))
    assert resp.stream is True
    assert resp.text == ""  # 评审已决：流式 .text 只做透传
    assert list(resp.iter_text()) == ["甲", "乙"]
    assert resp.raw is not None


# ------------------------------------------------------------------ #
#  4. 降级可见
# ------------------------------------------------------------------ #
def test_fallback_is_visible_in_envelope(monkeypatch) -> None:
    monkeypatch.setenv("FALLBACK_API_KEY", "fk")
    monkeypatch.setenv("FALLBACK_BASE_URL", "https://fallback.example/v1")
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)  # 跳过退避等待

    sink: list[dict] = []
    calls = {"n": 0}

    def factory(**kwargs):
        calls["n"] += 1
        # 第一次构造 = 主客户端（永远失败）；第二次 = 备用客户端（成功）
        return _FakeOpenAI(sink, fail_times=99 if calls["n"] == 1 else 0)

    monkeypatch.setattr(client_module, "OpenAI", factory)
    gw = client_module.LLMClient()
    resp = gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))

    assert resp.fallback_used is True
    assert resp.provider == "fallback"
    assert gw.stats["fallback_calls"] == 1
    assert gw.stats["total_calls"] == 1


def test_primary_failure_without_fallback_raises(monkeypatch) -> None:
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("FALLBACK_BASE_URL", raising=False)
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    _sink, gw = _gateway(monkeypatch, fail_times=99)
    with pytest.raises(client_module.AllModelsFailedError):
        gw.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
