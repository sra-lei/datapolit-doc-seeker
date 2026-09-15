"""采样温度可配置（评估 / A-B 必须能把温度固定为 0）。

背景：同代码同语料的单题分数摆幅实测达 0.75（T022 三轮 0.5/1.0/0.25），
噪声大于待测改动的量级；根因之一是生成侧硬编码 `temperature=0.3` + 推理模型采样。
温度配置化后，评估跑批用 `LLM_TEMPERATURE=0 / LLM_DECOMPOSE_TEMPERATURE=0`
固定口径，而线上默认仍是 0.3（历史口径，改动可逆）。

不依赖真实 LLM / Milvus / Redis。断言写成「等于 settings 里的值」而不是「等于 0.3」，
避免本地 `.env` 设置该变量后用例变红（沿用 test_llm_budget_guard 的鸭子类型假实现风格）。
"""
# pyright: reportArgumentType=false

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from docs_seeker.core.config import Settings, settings
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.domain.services.generator import Generator
from docs_seeker.infrastructure.retrieval.query_decomposer import QueryDecomposer


class RecordingLLM:
    """记录每次调用的 temperature，其余按固定响应返回。"""

    def __init__(self, content: str = "答案"):
        self.content = content
        self.calls: list[dict] = []

    def generate(self, messages, max_tokens=600, temperature=0.3, stream=False, name="llm-call", model=None):
        self.calls.append({"temperature": temperature, "max_tokens": max_tokens, "name": name, "model": model})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content), finish_reason="stop")]
        )


def test_class_defaults_keep_legacy_temperature():
    """默认值 = 历史口径（0.3 / 0.1）：不设环境变量时线上行为完全不变"""
    assert Settings.model_fields["llm_temperature"].default == 0.3
    assert Settings.model_fields["llm_decompose_temperature"].default == 0.1


def test_generator_temperature_comes_from_settings():
    llm = RecordingLLM()
    Generator(llm=llm).generate("问题", [Chunk(id="c1", text="正文")])
    assert llm.calls[0]["temperature"] == settings.llm_temperature


def test_generator_temperature_zero_reaches_gateway():
    llm = RecordingLLM()
    with patch.object(settings, "llm_temperature", 0.0):
        Generator(llm=llm).generate("问题", [Chunk(id="c1", text="正文")])
    assert llm.calls[0]["temperature"] == 0.0


def test_decomposer_temperature_zero_reaches_gateway():
    llm = RecordingLLM(content="子问题一\n子问题二")
    with patch.object(settings, "llm_decompose_temperature", 0.0):
        QueryDecomposer(llm=llm).decompose("问题")
    assert llm.calls[0]["temperature"] == 0.0
