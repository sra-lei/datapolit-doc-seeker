"""agent 路径失败时的回退开关（AGENT_FALLBACK_ENABLED）。

契约：

- **生产**（``environment=production``）默认**回退**旧单轮管线 —— 可用性优先；
- **开发 / 其他环境**默认**不回退** —— 让 agent 的失败显式暴露，不被旧管线的成功掩盖
  （否则 agentic 的 A/B 无法归因：agent 失败了但接口照样有答案）；
- 显式 ``AGENT_FALLBACK_ENABLED=true/false`` 覆盖上述推导。

不依赖真实 LLM / Milvus / Redis：agent 用必定抛错的替身，旧管线用固定答案的替身。
"""

from __future__ import annotations

import pytest

from docs_seeker.config.settings import Settings, settings
from docs_seeker.domain.services.chat_service import ChatService
from docs_seeker.models.chunk import Chunk
from docs_seeker.models.query import Query


class _BoomRunner:
    """agent 路径必定失败"""

    def run(self, question, top_k=10):
        raise RuntimeError("agent 挂了")


class _Retriever:
    def search(self, query, top_k=10, use_summary=True, **kwargs):
        return [Chunk(id="c1", text="正文", source="a.pdf", score=0.5)]


class _Decomposer:
    def decompose(self, question):
        return Query(text=question, sub_queries=[question])


class _Generator:
    def generate(self, question, docs, conversation_history=None, **kwargs):
        return "旧管线的答案", "medium"


class _Cache:
    def search(self, question):
        return None

    def store(self, question, payload):
        return None


class _Usage:
    def record_question(self, question):
        return None


def _service() -> ChatService:
    return ChatService(
        retriever=_Retriever(),
        generator=_Generator(),
        decomposer=_Decomposer(),
        cache=_Cache(),
        usage_tracker=_Usage(),
        agent_runner=_BoomRunner(),
    )


@pytest.fixture
def agent_enabled(monkeypatch):
    monkeypatch.setattr(settings, "agent_enabled", True)
    monkeypatch.setattr(settings, "agent_fallback_enabled", None)  # 回到「按 environment 推导」
    yield


def test_development_default_does_not_fall_back(agent_enabled, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    with pytest.raises(RuntimeError):
        _service().chat("问题")


def test_production_default_falls_back(agent_enabled, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    assert _service().chat("问题").answer == "旧管线的答案"


def test_explicit_true_overrides_development(agent_enabled, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "agent_fallback_enabled", True)
    assert _service().chat("问题").answer == "旧管线的答案"


def test_explicit_false_overrides_production(agent_enabled, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "agent_fallback_enabled", False)
    with pytest.raises(RuntimeError):
        _service().chat("问题")


# ------------------------------------------------------------------ #
#  推导规则本身
# ------------------------------------------------------------------ #
def test_derivation_follows_environment() -> None:
    assert Settings(environment="production").agent_fallback_to_pipeline is True
    assert Settings(environment="development").agent_fallback_to_pipeline is False
    assert Settings(environment="staging").agent_fallback_to_pipeline is False


def test_derivation_is_overridable() -> None:
    assert Settings(environment="development", agent_fallback_enabled=True).agent_fallback_to_pipeline is True
    assert Settings(environment="production", agent_fallback_enabled=False).agent_fallback_to_pipeline is False
