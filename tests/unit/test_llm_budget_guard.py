"""LLM 输出预算兜底单元测试（推理模型 reasoning 吃满预算 → 正文为空）。

背景：`deepseek-v4-flash` 这类推理模型的 reasoning token 与正文共用
`max_tokens` 预算，预算偏小会出现 `finish_reason=length` 且 `content` 为空——
表现为「答案为空」与「查询分解静默失效」。本文件锁定三处防线：

1. 生成：截断空返回 → 放大预算重试一次；仍为空则如实返回空串 + confidence=low；
2. 查询改写：同上，且空结果必须回退成「原问题单路检索」（不得抛异常）；
3. 缓存：空答案不得写入语义缓存。

Phase 2 变更：放大预算重试的机制从 Generator / QueryDecomposer 各自的副本收编到
transport 的 ``BudgetGuardMiddleware``（按请求 `meta["budget_guard"]` 启用）。因此
本文件里凡涉及兜底的用例都走**真实客户端 + 假 OpenAI 客户端**（而不是裸的假 LLM），
否则测不到 middleware；断言口径（调用次数与预算序列）与收编前完全一致。

不依赖真实 LLM / Milvus / Redis。
"""
# pyright: reportArgumentType=false

from __future__ import annotations

from types import SimpleNamespace

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.llm import LLMRequest
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.domain.models.query import Query
from docs_seeker.domain.services.chat_service import ChatService
from docs_seeker.domain.services.generator import Generator, compute_confidence
from docs_seeker.domain.services.query_decomposer import QueryDecomposer
from docs_seeker.infra.llm import client as client_module

_EMPTY_TRUNCATED = ("", "length")  # 推理吃满预算：正文空 + 截断
_EMPTY_STOPPED = ("", "stop")  # 真的没内容（非截断）


def _scripted_client(monkeypatch, script: list[tuple[str, str]]):
    """构造带完整 middleware 链的客户端；假客户端按 script 依次返回 (正文, finish_reason)。

    返回 (客户端, 调用参数 sink)。``script`` 用完后重复最后一项。
    """
    sink: list[dict] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            class _Completions:
                def create(self, **kw):
                    sink.append(kw)
                    content, finish = script[min(len(sink) - 1, len(script) - 1)]
                    if kw.get("stream"):
                        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])]
                        return iter(chunks)
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)]
                    )

            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(client_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    monkeypatch.delenv("FALLBACK_BASE_URL", raising=False)
    return client_module.LLMClient(), sink


def _budgets(sink: list[dict]) -> list[int]:
    return [call["max_tokens"] for call in sink]


def _docs(n: int = 3) -> list[Chunk]:
    return [Chunk(id=f"c{i}", text="正文", source="a.pdf", score=0.5) for i in range(n)]


# ------------------------------------------------------------------ #
#  生成：预算兜底 + 置信度
# ------------------------------------------------------------------ #
class TestGeneratorBudgetGuard:
    def test_truncated_empty_retries_with_boosted_budget(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [_EMPTY_TRUNCATED, ("这是答案", "stop")])
        answer, confidence = Generator(llm=client).generate("问题", _docs())
        assert answer == "这是答案"
        assert _budgets(sink) == [settings.llm_generate_max_tokens, settings.llm_retry_max_tokens]
        assert confidence != "low"

    def test_persistent_empty_returns_empty_low_confidence(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [_EMPTY_TRUNCATED])
        answer, confidence = Generator(llm=client).generate("问题", _docs())
        assert answer == ""
        assert confidence == "low"
        assert len(sink) == 2  # 只重试一次，不做无界重试

    def test_empty_without_truncation_does_not_retry(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [_EMPTY_STOPPED])
        answer, confidence = Generator(llm=client).generate("问题", _docs())
        assert answer == "" and confidence == "low"
        assert _budgets(sink) == [settings.llm_generate_max_tokens]

    def test_explicit_max_tokens_keeps_single_call(self, monkeypatch) -> None:
        """显式预算沿用旧口径（不放大重试），供测试/特殊调用方使用。"""
        client, sink = _scripted_client(monkeypatch, [("答案", "stop")])
        Generator(llm=client).generate("问题", _docs(), max_tokens=123)
        assert _budgets(sink) == [123]

    def test_stream_uses_configured_budget(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [("片段", "stop")])
        assert list(Generator(llm=client).generate_stream("问题", _docs())) == ["片段"]
        assert _budgets(sink) == [settings.llm_generate_max_tokens]
        assert sink[0]["stream"] is True

    def test_empty_answer_confidence_is_low_not_medium(self) -> None:
        assert compute_confidence("", _docs(10)) == "low"
        assert compute_confidence("   ", _docs(10)) == "low"


# ------------------------------------------------------------------ #
#  查询改写：预算兜底 + 空结果回退
# ------------------------------------------------------------------ #
class TestQueryDecomposerBudgetGuard:
    def test_empty_then_retry_success(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [_EMPTY_TRUNCATED, ("子问题A\n子问题B", "stop")])
        query = QueryDecomposer(llm=client).decompose("原问题")
        assert query.sub_queries == ["原问题", "子问题A", "子问题B"]
        assert _budgets(sink) == [settings.llm_decompose_max_tokens, settings.llm_retry_max_tokens]

    def test_persistent_empty_falls_back_to_original_question(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [_EMPTY_TRUNCATED])
        query = QueryDecomposer(llm=client).decompose("原问题")
        assert query.sub_queries == ["原问题"]
        assert len(sink) == 2

    def test_simple_question_returns_single_sub_query(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [("原问题", "stop")])
        query = QueryDecomposer(llm=client).decompose("原问题")
        assert query.sub_queries == ["原问题"]
        assert _budgets(sink) == [settings.llm_decompose_max_tokens]


# ------------------------------------------------------------------ #
#  兜底是按请求启用的（不会被偷偷放大预算）
# ------------------------------------------------------------------ #
class TestBudgetGuardOptIn:
    def test_without_flag_no_boost(self, monkeypatch) -> None:
        client, sink = _scripted_client(monkeypatch, [_EMPTY_TRUNCATED, ("不该被调用", "stop")])
        resp = client.generate(LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=100))
        assert resp.text == ""
        assert _budgets(sink) == [100]

    def test_boost_skipped_when_retry_budget_not_larger(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "llm_retry_max_tokens", 100)
        client, sink = _scripted_client(monkeypatch, [_EMPTY_TRUNCATED, ("不该被调用", "stop")])
        resp = client.generate(
            LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=100, meta={"budget_guard": True})
        )
        assert resp.text == ""
        assert _budgets(sink) == [100]


# ------------------------------------------------------------------ #
#  ChatService：空答案不写缓存
# ------------------------------------------------------------------ #
class RecordingCache:
    def __init__(self) -> None:
        self.stored: list[dict] = []

    def search(self, question):
        return None

    def store(self, question, payload):
        self.stored.append(payload)


class _Retriever:
    def search(self, query, top_k=10, use_summary=True, **kwargs):
        return [Chunk(id="c1", text="正文", source="a.pdf", score=0.5)]


class _Decomposer:
    def decompose(self, question):
        return Query(text=question, sub_queries=[question])


class _ScriptedGenerator:
    def __init__(self, answer: str):
        self.answer = answer

    def generate(self, question, docs, conversation_history=None, **kwargs):
        return self.answer, ("low" if not self.answer.strip() else "medium")


class _Usage:
    def record_question(self, question):
        return None


def _service(answer: str, cache: RecordingCache) -> ChatService:
    return ChatService(
        retriever=_Retriever(),
        generator=_ScriptedGenerator(answer),
        decomposer=_Decomposer(),
        cache=cache,
        usage_tracker=_Usage(),
    )


class TestChatServiceCacheGuard:
    def test_empty_answer_is_not_cached(self) -> None:
        cache = RecordingCache()
        result = _service("", cache).chat("问题", use_cache=True)
        assert result.answer == ""
        assert cache.stored == []

    def test_non_empty_answer_is_cached(self) -> None:
        cache = RecordingCache()
        _service("正常回答", cache).chat("问题", use_cache=True)
        assert [p["answer"] for p in cache.stored] == ["正常回答"]
