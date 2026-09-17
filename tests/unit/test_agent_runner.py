"""AgentRunner M1 单测：动作协议、自纠、预算、拒答、证据去重、短超时透传。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docs_seeker.agent.runner import AgentError, AgentRunner
from docs_seeker.agent.tools import default_tools
from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.llm import LLMRequest, LLMResponse
from docs_seeker.domain.models.chunk import Chunk


def _resp(content: str):
    msg = SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


class FakeLLM:
    """按队列返回 decide 响应；compose 统一返回固定答案，并记录全部调用参数。"""

    def __init__(self, decide_contents: list[str], compose_answer: str = "最终答案"):
        self._queue = list(decide_contents)
        self._compose = compose_answer
        self.calls: list[dict] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(
            {
                "name": request.name,
                "model": request.model,
                "timeout": request.timeout,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "messages": [dict(m) for m in request.messages],
                "meta": dict(request.meta),
            }
        )
        if request.name == "agent-compose":
            return LLMResponse.from_raw(_resp(self._compose))
        return LLMResponse.from_raw(_resp(self._queue.pop(0)))


class FakeRetriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.calls: list[dict] = []
        self._chunks = chunks if chunks is not None else []

    def search(self, query, top_k=10, meta_filter=None, **kwargs):
        self.calls.append({"query": query, "top_k": top_k, "meta_filter": meta_filter})
        return list(self._chunks)


def _chunks():
    return [
        Chunk(id="c1", text="库龄超过6个月收取超龄库存附加费", chapter="第三章"),
        Chunk(id="c2", text="加拿大站为9个月，日本站365天", chapter="第三章"),
    ]


def _compose_call(llm: FakeLLM) -> dict:
    return next(c for c in llm.calls if c["name"] == "agent-compose")


def test_happy_path_retrieve_then_final(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 4)
    monkeypatch.setattr(settings, "llm_judge_timeout_seconds", 30.0)
    monkeypatch.setattr(settings, "agent_judge_max_tokens", 500)
    monkeypatch.setattr(settings, "llm_generate_model", "deepseek-chat")
    retriever = FakeRetriever(_chunks())
    llm = FakeLLM(
        [
            '{"thought": "先语义检索", "action": "retrieve", "action_input": {"query": "超龄库存多久收费"}}',
            '{"thought": "证据齐了", "action": "final", "action_input": {"sufficient": true, "reason": ""}}',
        ]
    )
    runner = AgentRunner(llm=llm, retriever=retriever)

    result = runner.run("美国站库龄多久收附加费？", top_k=10)

    assert result.sufficient is True
    assert result.answer == "最终答案"
    assert len(result.evidence) == 2
    assert [s.action for s in result.steps] == ["retrieve", "final"]
    # decide 走推理模型（model=None）+ 短超时；compose 走生成模型覆盖 + 默认超时
    decide_call = next(c for c in llm.calls if c["name"] == "agent-decide")
    compose_call = _compose_call(llm)
    assert decide_call["model"] is None
    assert decide_call["timeout"] == 30.0
    assert compose_call["model"] == "deepseek-chat"
    assert compose_call["timeout"] is None
    # 决策 / 成文两处都挂预算兜底：推理模型 + 小预算是「思考吃满预算 → 空正文」的高发组合，
    # 成文与普通生成路径同源，同样不能静默交空答案
    assert decide_call["meta"].get("budget_guard") is True
    assert compose_call["meta"].get("budget_guard") is True
    assert retriever.calls[0]["meta_filter"] is None


def test_parse_error_self_corrects(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 4)
    llm = FakeLLM(
        [
            "我想想啊……",  # 不是 JSON
            '{"thought": "改检索", "action": "retrieve", "action_input": {"query": "库龄附加费"}}',
            '{"thought": "够了", "action": "final", "action_input": {"sufficient": true, "reason": ""}}',
        ]
    )
    result = AgentRunner(llm=llm, retriever=FakeRetriever(_chunks())).run("库龄附加费", top_k=5)

    actions = [s.action for s in result.steps]
    assert actions == ["parse_error", "retrieve", "final"]
    assert result.steps[0].error
    # 自纠反馈作为 user 消息回灌
    messages = llm.calls[1]["messages"]
    assert "无法解析" in messages[-1]["content"]


def test_consecutive_parse_errors_raise(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 4)
    llm = FakeLLM(["不是JSON", "也不是JSON"])
    with pytest.raises(AgentError):
        AgentRunner(llm=llm, retriever=FakeRetriever()).run("某问题")


def test_budget_exhausted_composes_from_evidence(monkeypatch):
    """步数耗尽不回退、不丢证据：有证据→强制成文，trace 标明 budget_forced。"""
    monkeypatch.setattr(settings, "agent_max_steps", 2)
    llm = FakeLLM(
        [
            '{"thought": "查", "action": "retrieve", "action_input": {"query": "q1"}}',
            '{"thought": "再查", "action": "retrieve", "action_input": {"query": "q2"}}',
        ]
    )
    result = AgentRunner(llm=llm, retriever=FakeRetriever(_chunks())).run("枚举全部站点规则")
    assert result.sufficient is True  # 有证据 → 成文（非回退旧管线）
    last = result.steps[-1]
    assert last.action == "final" and last.action_input.get("budget_forced")
    compose_call = next(c for c in llm.calls if c["name"] == "agent-compose")
    assert compose_call is not None


def test_budget_exhausted_without_evidence_abstains(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 1)
    llm = FakeLLM(
        [
            '{"thought": "查", "action": "retrieve", "action_input": {"query": "q1"}}',
        ],
        compose_answer="ABSTAIN: 现有资料中没有找到该内容，无法回答。",
    )
    result = AgentRunner(llm=llm, retriever=FakeRetriever([])).run("库里没有的问题")
    assert result.sufficient is False
    assert result.confidence == "low"
    assert result.answer == "现有资料中没有找到该内容，无法回答。"
    # 决策判断用短超时，成文调用不强制短超时
    assert llm.calls[0]["timeout"] == settings.llm_judge_timeout_seconds


def test_abstain_when_insufficient(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 3)
    llm = FakeLLM(
        [
            '{"thought": "先查", "action": "retrieve", "action_input": {"query": "虫鼠咬坏赔偿"}}',
            '{"thought": "资料没有", "action": "final", "action_input": {"sufficient": false, "reason": "无赔偿规则"}}',
        ],
        compose_answer="ABSTAIN: 现有资料中未提及虫鼠咬坏的赔偿规则，无法回答。",
    )
    result = AgentRunner(llm=llm, retriever=FakeRetriever([])).run("库存被虫鼠咬坏怎么赔偿？")

    assert result.sufficient is False
    assert "未提及" in result.answer
    assert result.confidence == "low"
    prompt = _compose_call(llm)["messages"][-1]["content"]
    # 走「先核实再决定」提示，决策顾虑原样带入
    assert "核实" in prompt and "无赔偿规则" in prompt


def test_lookup_article_passes_meta_filter():
    retriever = FakeRetriever([Chunk(id="a59", text="锂电池 TIC 认证新规", chapter="第一章", article="第59条")])
    tools = default_tools(retriever)
    out = tools["lookup_article"].run("第59条是什么内容？", top_k=10)
    assert retriever.calls[0]["meta_filter"]  # 解析出 article 过滤条件
    assert out.chunks and out.chunks[0].id == "a59"

    out2 = tools["lookup_article"].run("什么是泛欧计划？", top_k=10)
    assert not out2.chunks
    assert "章/节/条" in out2.observation


def test_lookup_article_rejects_hallucinated_number():
    """结构过滤无命中时混合检索会回退全量；lookup 必须剔除不匹配编号的近似结果。"""
    unrelated = [Chunk(id="x", text="完全不相关的佣金通知", chapter="第六章", article="第3条")]
    retriever = FakeRetriever(unrelated)
    tools = default_tools(retriever)
    out = tools["lookup_article"].run("第59条讲了什么？", top_k=10)
    assert not out.chunks
    assert "不计入证据" in out.observation


def test_evidence_dedup_across_steps(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 4)
    llm = FakeLLM(
        [
            '{"thought": "查1", "action": "retrieve", "action_input": {"query": "q1"}}',
            '{"thought": "查2", "action": "retrieve", "action_input": {"query": "q2"}}',
            '{"thought": "齐", "action": "final", "action_input": {"sufficient": true, "reason": ""}}',
        ]
    )
    result = AgentRunner(llm=llm, retriever=FakeRetriever(_chunks())).run("多跳问题")
    assert len(result.evidence) == 2  # 两次检索同一批 id，不翻倍


def test_unknown_action_feeds_back(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 3)
    llm = FakeLLM(
        [
            '{"thought": "x", "action": "browse_web", "action_input": {}}',
            '{"thought": "好", "action": "final", "action_input": {"sufficient": false, "reason": "无工具"}}',
        ],
        compose_answer="资料中没有相关内容。",
    )
    result = AgentRunner(llm=llm, retriever=FakeRetriever(_chunks())).run("q")
    assert result.steps[0].action == "parse_error"
    assert result.steps[0].error and "未知 action" in result.steps[0].error


def test_false_abstain_corrected_by_verification(monkeypatch):
    """决策代理误判不足，但证据其实够；成文核实阶段正常作答 → sufficient 被纠正为 True。

    回归：真服务冒烟 T26（美/加/日库龄门槛）曾因此误拒答。
    """
    monkeypatch.setattr(settings, "agent_max_steps", 3)
    llm = FakeLLM(
        [
            '{"thought": "查", "action": "retrieve", "action_input": {"query": "库龄门槛"}}',
            '{"thought": "我觉得不全", "action": "final", "action_input": {"sufficient": false, "reason": "怕缺日本站"}}',
        ],
        # 核实后发现资料里有答案，不输出 ABSTAIN: 前缀
        compose_answer="美国站超过6个月、加拿大站超过9个月收超龄库存附加费。来源：[1]",
    )
    result = AgentRunner(llm=llm, retriever=FakeRetriever(_chunks())).run("三国门槛分别是多久？")
    assert result.sufficient is True
    assert result.answer.startswith("美国站超过6个月")
    # 走的是核实提示而非普通成文提示
    assert "ABSTAIN" in _compose_call(llm)["messages"][-1]["content"]
