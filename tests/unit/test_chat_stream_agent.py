"""chat_stream 的 Agentic 分流（选项 A）单元测试。

契约（与 chat() 对称）：
- agent_enabled 且注入 agent_runner → agent 路径：完整答案按片段吐 delta，
  meta/done 携带 agent_steps 与 agent_sufficient；
- agent 失败：回退旧管线（meta 带 None agent 字段 + 旧管线流式答案）或显式 error 事件；
- agent 关闭 → 纯旧管线流式（事件不含 agent 字段，值为 None）。

不依赖真实 LLM / Milvus / Redis：agent 与旧管线全用替身。
"""

from __future__ import annotations

import pytest

from docs_seeker.config.settings import settings
from docs_seeker.models.chunk import Chunk
from docs_seeker.models.query import Query
from docs_seeker.services.chat_service import ChatService, _chunk_text


class _OkRunner:
    """agent 路径成功"""

    def run(self, question, top_k=10):
        from docs_seeker.agent.models import AgentResult, AgentStep

        return AgentResult(
            answer="agent 的完整答案，用来测试分片吐流。",
            confidence="high",
            steps=[
                AgentStep(
                    idx=1,
                    thought="先检索",
                    action="retrieve",
                    action_input={"query": "问题"},
                    observation="找到 1 条证据",
                    elapsed_ms=12,
                )
            ],
            evidence=[Chunk(id="c1", text="证据正文", source="a.pdf", score=0.6)],
            sufficient=True,
        )


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


class _StreamGenerator:
    """旧管线流式生成（逐段吐）"""

    def generate_stream(self, question, docs, conversation_history=None):
        yield "旧管线第一段"
        yield "旧管线第二段"


class _Cache:
    def search(self, question):
        return None

    def store(self, question, payload):
        return None


class _Usage:
    def record_question(self, question):
        return None


def _service(agent_runner=None) -> ChatService:
    return ChatService(
        retriever=_Retriever(),
        generator=_StreamGenerator(),  # type: ignore[arg-type]
        decomposer=_Decomposer(),
        cache=_Cache(),  # type: ignore[arg-type]
        usage_tracker=_Usage(),  # type: ignore[arg-type]
        agent_runner=agent_runner,
    )


def _events(service: ChatService) -> list[dict]:
    return list(service.chat_stream("问题", use_cache=False))


@pytest.fixture
def agent_on(monkeypatch):
    monkeypatch.setattr(settings, "agent_enabled", True)
    monkeypatch.setattr(settings, "agent_fallback_enabled", None)
    yield


# ------------------------------------------------------------------ #
#  agent 成功路径
# ------------------------------------------------------------------ #
def test_stream_agent_success_yields_meta_deltas_done(agent_on) -> None:
    events = _events(_service(agent_runner=_OkRunner()))

    types = [e["type"] for e in events]
    # 答案约 20 字符 < 80 分片大小 → 1 段 delta；meta + done 收尾
    assert types == ["meta", "delta", "done"]


def test_stream_agent_meta_carries_agent_fields(agent_on) -> None:
    events = _events(_service(agent_runner=_OkRunner()))
    meta = events[0]

    assert meta["cached"] is False
    assert meta["agent_sufficient"] is True
    assert meta["agent_steps"][0]["action"] == "retrieve"
    assert meta["agent_steps"][0]["thought"] == "先检索"
    assert meta["sources"]  # 来自 agent 证据


def test_stream_agent_deltas_reassemble_to_full_answer(agent_on) -> None:
    events = _events(_service(agent_runner=_OkRunner()))
    deltas = [e["content"] for e in events if e["type"] == "delta"]

    assert "".join(deltas) == "agent 的完整答案，用来测试分片吐流。"


def test_stream_agent_done_carries_agent_fields(agent_on) -> None:
    events = _events(_service(agent_runner=_OkRunner()))
    done = events[-1]

    assert done["answer"] == "agent 的完整答案，用来测试分片吐流。"
    assert done["confidence"] == "high"
    assert done["agent_sufficient"] is True
    assert done["agent_steps"] and done["agent_steps"][0]["idx"] == 1


# ------------------------------------------------------------------ #
#  agent 失败路径
# ------------------------------------------------------------------ #
def test_stream_agent_failure_falls_back_in_production(agent_on, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")  # 默认回退
    events = _events(_service(agent_runner=_BoomRunner()))

    types = [e["type"] for e in events]
    assert types == ["meta", "delta", "delta", "done"]
    meta = events[0]
    assert meta["agent_steps"] is None
    assert meta["agent_sufficient"] is None
    assert events[-1]["answer"] == "旧管线第一段旧管线第二段"
    assert events[-1]["agent_steps"] is None
    assert events[-1]["agent_sufficient"] is None


def test_stream_agent_failure_errors_out_in_development(agent_on, monkeypatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")  # 默认不回退
    events = _events(_service(agent_runner=_BoomRunner()))

    assert events == [{"type": "error", "message": "Agent 路径失败: RuntimeError: agent 挂了"}]


# ------------------------------------------------------------------ #
#  agent 关闭 → 纯旧管线
# ------------------------------------------------------------------ #
def test_stream_without_agent_is_plain_pipeline() -> None:
    events = _events(_service(agent_runner=None))

    types = [e["type"] for e in events]
    assert types == ["meta", "delta", "delta", "done"]
    assert events[0]["agent_steps"] is None
    assert events[0]["agent_sufficient"] is None
    assert events[-1]["answer"] == "旧管线第一段旧管线第二段"


# ------------------------------------------------------------------ #
#  _chunk_text 辅助
# ------------------------------------------------------------------ #
def test_chunk_text_splits_and_reassembles() -> None:
    text = "一二三四五六七八九十"
    chunks = _chunk_text(text, size=3)
    assert chunks == ["一二三", "四五六", "七八九", "十"]
    assert "".join(chunks) == text


def test_chunk_text_empty() -> None:
    assert _chunk_text("") == []
    assert _chunk_text("   ") == ["   "]
