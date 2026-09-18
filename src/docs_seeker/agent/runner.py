"""Agent 循环编排（M1）。

数据流：思考(LLM) → 动作 JSON → 本地执行工具 → observation 回灌 → … → final → 成文。
本层只做编排与预算治理；重试/熔断/超时/降级在 LLMClient，不在这里重复。

回退契约：任何编排层异常（AgentError、client 的 AllModelsFailedError 等）都向上抛，
由 chat_service 捕获后回退旧单轮管线。步数耗尽不回退——基于已收集证据成文/拒答。
"""

from __future__ import annotations

import json
import re
import time

from loguru import logger

from docs_seeker.agent.adapter import parse_llm_response
from docs_seeker.agent.models import AgentResult, AgentStep
from docs_seeker.agent.prompts import COMPOSE_PROMPT, SYSTEM_PROMPT, VERIFY_PROMPT
from docs_seeker.agent.tools import default_tools
from docs_seeker.config.settings import settings
from docs_seeker.domain.interfaces.llm import LLMProvider
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.domain.models.llm import LLMRequest
from docs_seeker.domain.services.generator import compute_confidence

VALID_ACTIONS = ("retrieve", "lookup_article", "final")
MAX_PARSE_ERRORS = 2


class AgentError(Exception):
    """语义级失败（动作连续无法解析等），调用方应回退旧管线。"""


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    """从模型输出提取动作 JSON；容忍代码块包裹与前后多余文字。"""
    cleaned = _strip_fence(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("输出中没有 JSON 对象")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON 不是对象")
    return obj


def _chunk_key(c: Chunk) -> str:
    return c.id or f"text:{hash(c.text)}"


class AgentRunner:
    def __init__(self, llm: LLMProvider, tools: dict | None = None, retriever=None):
        self.llm = llm
        self.tools = tools or default_tools(retriever)
        if retriever is None and tools is None:
            raise ValueError("必须提供 retriever 或 tools")

    # ---- 对外入口 ----
    def run(self, question: str, top_k: int = 10) -> AgentResult:
        max_steps = settings.agent_max_steps
        messages: list[dict] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(question=question, remaining=max_steps),
            },
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []
        evidence: dict[str, Chunk] = {}
        parse_errors = 0

        for idx in range(1, max_steps + 1):
            t0 = time.time()
            resp = self.llm.generate(
                LLMRequest(
                    messages=messages,
                    max_tokens=settings.agent_judge_max_tokens,
                    temperature=settings.llm_decompose_temperature,
                    name="agent-decide",
                    model=None,  # 判断始终走 LLM_MODEL（推理模型）
                    timeout=settings.llm_judge_timeout_seconds,
                    # 决策走推理模型 + 小预算（AGENT_JUDGE_MAX_TOKENS），是「思考吃满预算 →
                    # 空正文」的高发组合，与生成 / 改写两处对齐挂预算兜底
                    meta={"budget_guard": True},
                )
            )
            raw = parse_llm_response(resp).content
            messages.append({"role": "assistant", "content": raw})

            try:
                action_obj = _extract_json(raw)
                action = str(action_obj.get("action") or "").strip()
                action_input = action_obj.get("action_input") or {}
                thought = str(action_obj.get("thought") or "").strip()
                if action not in VALID_ACTIONS:
                    raise ValueError(f"未知 action: {action!r}（允许 {VALID_ACTIONS}）")
                if not isinstance(action_input, dict):
                    raise ValueError("action_input 必须是对象")
                parse_errors = 0
            except (ValueError, json.JSONDecodeError) as e:
                parse_errors += 1
                step = AgentStep(
                    idx=idx,
                    thought="",
                    action="parse_error",
                    action_input={},
                    error=str(e),
                    elapsed_ms=int((time.time() - t0) * 1000),
                )
                steps.append(step)
                logger.warning(f"[agent] 第 {idx} 步动作解析失败（{parse_errors}/{MAX_PARSE_ERRORS}）: {e}")
                if parse_errors >= MAX_PARSE_ERRORS:
                    raise AgentError(f"动作连续 {parse_errors} 次无法解析: {e}") from e
                messages.append(
                    {
                        "role": "user",
                        "content": f"上一步输出无法解析为动作 JSON（{e}）。请只输出一个 JSON 对象："
                        '{"thought": "...", "action": "...", "action_input": {...}}',
                    }
                )
                continue

            if action == "final":
                claimed_sufficient = bool(action_input.get("sufficient"))
                reason = str(action_input.get("reason") or "")
                step = AgentStep(
                    idx=idx,
                    thought=thought,
                    action="final",
                    action_input={"sufficient": claimed_sufficient, "reason": reason},
                    elapsed_ms=int((time.time() - t0) * 1000),
                )
                steps.append(step)
                answer, abstained = self._compose(question, list(evidence.values()), claimed_sufficient, reason)
                return AgentResult(
                    answer=answer,
                    confidence=compute_confidence(answer, list(evidence.values())),
                    steps=steps,
                    evidence=list(evidence.values()),
                    # 成文阶段核实：判不够但资料其实有答案 → 纠正为可答（防 false abstain）
                    sufficient=not abstained,
                )

            # 工具动作
            query = str(action_input.get("query") or "").strip()
            step = AgentStep(
                idx=idx,
                thought=thought,
                action=action,
                action_input={"query": query},
                elapsed_ms=int((time.time() - t0) * 1000),
            )
            if not query:
                step.error = "缺少 query 参数"
                messages.append({"role": "user", "content": "工具调用缺少 query 参数，请重新输出动作 JSON。"})
            else:
                output = self.tools[action].run(query, top_k=top_k)
                for c in output.chunks:
                    evidence.setdefault(_chunk_key(c), c)
                step.observation = output.observation
                remaining = max_steps - idx
                hint = (
                    f"\n（剩余检索次数 {remaining}。不要猜测编号或重复同一 query；"
                    "若现有资料已足以完整回答，下一步直接 final）"
                )
                logger.info(
                    f"[agent] 第 {idx} 步 {action} query={query[:40]!r} "
                    f"hits={len(output.chunks)} evidence={len(evidence)} remain={remaining}"
                )
                messages.append({"role": "user", "content": output.observation + hint})
            steps.append(step)

        # 步数用尽仍未 final：不丢弃已收集的证据——有证据则基于证据成文，
        # 无证据则拒答；trace 里补一个 final 步，标明是预算收尾而非模型决策
        sufficient = bool(evidence)
        reason = "达到最大检索步数，基于已收集证据成文" if sufficient else "达到最大检索步数仍无任何证据"
        steps.append(
            AgentStep(
                idx=max_steps + 1,
                thought="步数上限收尾",
                action="final",
                action_input={"sufficient": sufficient, "reason": reason, "budget_forced": True},
            )
        )
        answer, abstained = self._compose(question, list(evidence.values()), sufficient, reason)
        return AgentResult(
            answer=answer,
            confidence=compute_confidence(answer, list(evidence.values())),
            steps=steps,
            evidence=list(evidence.values()),
            sufficient=not abstained,
        )

    # ---- 最终成文（非推理模型，保持延迟/成本优势）----
    def _compose(self, question: str, evidence: list[Chunk], claimed_sufficient: bool, reason: str) -> tuple[str, bool]:
        """返回 (答案, 是否确认为拒答)。

        claimed_sufficient=False 时走「先核实再决定」提示：资料里其实有答案
        就正常作答（防决策代理 false abstain），确实没有才输出 ABSTAIN: 前缀。
        """
        evidence_text = self._format_evidence(evidence)
        if claimed_sufficient:
            prompt = COMPOSE_PROMPT.format(question=question, evidence=evidence_text)
        else:
            prompt = VERIFY_PROMPT.format(
                question=question, reason=reason or "已检索资料可能不足以回答", evidence=evidence_text
            )
        resp = self.llm.generate(
            LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.llm_generate_max_tokens,
                temperature=settings.llm_temperature,
                name="agent-compose",
                model=settings.llm_generate_model or None,  # 空 = 沿用 LLM_MODEL（旧行为）
                # 成文与普通生成路径同源（非推理模型 + LLM_GENERATE_MAX_TOKENS），
                # 同样挂预算兜底，避免截断空正文时静默交出空答案
                meta={"budget_guard": True},
            )
        )
        answer = parse_llm_response(resp).content.strip()
        if claimed_sufficient or not answer.startswith("ABSTAIN:"):
            return answer, False
        # 真正拒答：去掉协议前缀，交给上游（仍保留拒答措辞与 marker）
        return answer[len("ABSTAIN:") :].strip() or "现有资料中未找到该问题的答案。", True

    @staticmethod
    def _format_evidence(evidence: list[Chunk]) -> str:
        if not evidence:
            return "（没有检索到任何资料）"
        budget = settings.agent_evidence_char_budget
        lines: list[str] = []
        used = 0
        for i, c in enumerate(evidence, 1):
            loc = " ".join(x for x in (c.chapter, c.article) if x)
            block = f"[{i}] {loc}\n{c.text}" if loc else f"[{i}]\n{c.text}"
            if used + len(block) > budget:
                lines.append(block[: max(0, budget - used)])
                break
            lines.append(block)
            used += len(block)
        return "\n\n".join(lines)
