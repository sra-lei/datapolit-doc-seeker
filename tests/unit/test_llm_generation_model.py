"""生成层模型覆盖（分层路由：生成用非推理模型，判断/改写用推理模型）。

背景：`deepseek-v4-flash` 这类推理模型在生成环节的增益尚未证明（本语料是抽取式的），
却带来 51.8s/题 的平均延迟与 reasoning token 成本。`LLM_GENERATE_MODEL` 允许只把
**生成**改走非推理模型（如 `deepseek-chat`），查询改写仍用 `LLM_MODEL`。
留空 = 旧行为（生成也用 LLM_MODEL），改动可逆。

不依赖真实 API：假 OpenAI 客户端只记录调用参数。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from docs_seeker.core.config import Settings, settings
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.domain.models.llm import LLMRequest
from docs_seeker.domain.services.generator import Generator
from docs_seeker.infra.llm import client as client_module


class _FakeOpenAI:
    def __init__(self, sink: list[dict]):
        outer = self
        self.sink = sink

        class _Completions:
            def create(self, **kwargs):
                outer.sink.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")]
                )

        self.chat = SimpleNamespace(completions=_Completions())


def _install_fake_client(monkeypatch) -> list[dict]:
    sink: list[dict] = []
    monkeypatch.setattr(client_module, "OpenAI", lambda **kwargs: _FakeOpenAI(sink))
    return sink


def test_class_default_is_empty_so_behaviour_is_unchanged():
    assert Settings.model_fields["llm_generate_model"].default == ""


def test_client_model_override_wins(monkeypatch):
    sink = _install_fake_client(monkeypatch)
    client = client_module.LLMClient()
    client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=100, model="deepseek-chat"))
    assert sink[0]["model"] == "deepseek-chat"


def test_client_without_override_uses_primary_model(monkeypatch):
    sink = _install_fake_client(monkeypatch)
    client = client_module.LLMClient()
    client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=100))
    assert sink[0]["model"] == client.primary_model


def test_generator_passes_configured_generation_model(monkeypatch):
    sink = _install_fake_client(monkeypatch)
    client = client_module.LLMClient()
    with patch.object(settings, "llm_generate_model", "deepseek-chat"):
        Generator(llm=client).generate("问题", [Chunk(id="c1", text="正文")])
    assert sink[0]["model"] == "deepseek-chat"


def test_generator_without_override_keeps_primary_model(monkeypatch):
    sink = _install_fake_client(monkeypatch)
    client = client_module.LLMClient()
    with patch.object(settings, "llm_generate_model", ""):
        Generator(llm=client).generate("问题", [Chunk(id="c1", text="正文")])
    assert sink[0]["model"] == client.primary_model
