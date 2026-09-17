"""docs-seeker - 答案生成器（应用服务）"""

from loguru import logger

from docs_seeker.core.config import prompts, settings
from docs_seeker.domain.interfaces.llm import LLMProvider
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.domain.models.llm import LLMRequest, LLMResponse
from docs_seeker.infra.llm.client import get_llm_client

_DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的文档问答助手。请根据以下检索到的文档内容回答用户问题。\n"
    "要求：\n"
    "1. 只基于文档内容回答，不要编造\n"
    "2. 文档中没有相关内容时明确告知\n"
    "3. 回答简洁、准确、有条理\n"
    "4. 引用文档时标注来源"
)


class Generator:
    def __init__(self, llm: LLMProvider | None = None):
        # 允许注入 LLM（deps 组装点传入）；缺省时走全局客户端单例
        self.llm = llm or get_llm_client()

    def _build_messages(
        self, question: str, docs: list[Chunk], conversation_history: list[dict] | None = None
    ) -> list[dict]:
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(
                f"【文档{i + 1}】\n来源: {doc.source}\n章节: {doc.chapter_title or doc.chapter}\n内容: {doc.text[:500]}"
            )
        context = "\n\n".join(context_parts)
        system_prompt = (prompts.get("generator") or {}).get("system") or _DEFAULT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-4:])
        messages.append({"role": "user", "content": f"基于以下文档回答问题：\n\n{context}\n\n问题：{question}"})
        return messages

    def _call(
        self,
        messages: list[dict],
        max_tokens: int,
        stream: bool = False,
        name: str = "generate-response",
        timeout: float | None = None,
        budget_guard: bool = False,
    ) -> LLMResponse:
        """单次调用 LLM 客户端，返回 ``LLMResponse`` 信封。

        温度取自 ``LLM_TEMPERATURE``（默认 0.3 = 历史口径；评估/A-B 用 0，
        否则采样噪声会盖过待测改动的量级）；模型可取 ``LLM_GENERATE_MODEL``
        覆盖（分层路由：生成走非推理模型；空 = 沿用 LLM_MODEL，旧行为）；
        ``timeout`` 非空时覆盖全局超时（判断类调用传短超时快速失败）。

        ``budget_guard=True`` 时启用 transport 的预算兜底（截断空正文 → 放大预算
        重试一次，机制见 ``middleware.BudgetGuardMiddleware``）。

        降级（``fallback_used``）不再静默：这里记 warning，便于评估归因。
        """
        response = self.llm.generate(
            LLMRequest(
                messages=messages,
                max_tokens=max_tokens,
                temperature=settings.llm_temperature,
                stream=stream,
                name=name,
                model=settings.llm_generate_model or None,
                timeout=timeout,
                meta={"budget_guard": True} if budget_guard else {},
            )
        )
        if response.fallback_used:
            logger.warning(f"本次调用走了降级模型（name={name}, provider={response.provider}）")
        return response

    def _call_with_budget_guard(
        self, messages: list[dict], name: str = "generate-response", timeout: float | None = None
    ) -> str:
        """生成答案（带预算兜底）。

        机制已收编到 transport 的 ``BudgetGuardMiddleware``（生成与查询改写共用一份）：
        正文为空且 ``finish_reason == 'length'`` 时，用 ``LLM_RETRY_MAX_TOKENS``
        放大预算重试**一次**。这里只负责发起调用 + 兜底后仍为空时如实记 warning ——
        绝不把 reasoning_content 当正文、也不静默编造。
        """
        budget = settings.llm_generate_max_tokens
        response = self._call(messages, budget, name=name, timeout=timeout, budget_guard=True)
        if not response.text:
            logger.warning(
                f"生成正文为空（finish={response.finish_reason}, max_tokens={budget}）"
                "——检查 LLM_MODEL 是否为推理模型、LLM_GENERATE_MAX_TOKENS 是否偏小"
            )
        return response.text

    def generate(
        self,
        question: str,
        docs: list[Chunk],
        conversation_history: list[dict] | None = None,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        """生成答案；返回 (answer, confidence)。

        显式传入 ``max_tokens`` 时沿用旧口径（单次调用、不做放大重试），供测试
        与特殊调用方使用；不传则走 ``LLM_GENERATE_MAX_TOKENS`` + 预算兜底。
        ``timeout`` 非空时透传客户端覆盖全局超时（agent 循环内判断类调用传
        ``LLM_JUDGE_TIMEOUT_SECONDS``，宁可快速失败走回退也不要卡住循环）。
        """
        messages = self._build_messages(question, docs, conversation_history)
        try:
            if max_tokens is not None:
                response = self._call(messages, max_tokens, timeout=timeout)
                answer = response.text
            else:
                answer = self._call_with_budget_guard(messages, timeout=timeout)
            confidence = compute_confidence(answer, docs)
            logger.info(f"答案生成: confidence={confidence} docs={len(docs)} len={len(answer)}")
            return answer, confidence
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"答案生成失败: {e}", "low"

    def generate_stream(self, question: str, docs: list[Chunk], conversation_history: list[dict] | None = None):
        """流式生成：逐段产出增量文本（str）。

        异常不在此捕获（由调用方决定如何收尾），最后一段文本产出后自然结束。
        正文提取走 ``LLMResponse.iter_text()``（兼容 reasoning 模型先出
        reasoning_content、以及 include_usage 末包 choices 为空）；预算不足时正文
        可能整段为空 —— 此时记 warning，避免再次静默退化。
        """
        messages = self._build_messages(question, docs, conversation_history)
        response = self._call(messages, settings.llm_generate_max_tokens, stream=True)
        produced = False
        for delta in response.iter_text():
            produced = True
            yield delta
        if not produced:
            logger.warning(
                f"流式生成正文为空（max_tokens={settings.llm_generate_max_tokens}）"
                "——疑似 reasoning 吃满预算，检查 LLM_GENERATE_MAX_TOKENS"
            )


def compute_confidence(answer: str, docs: list[Chunk]) -> str:
    """按答案措辞与命中文档质量评估置信度（与一次性生成共用同一口径）"""
    if not answer.strip():
        # 空答案不得报 medium/high（推理模型预算耗尽时会走到这里）
        return "low"
    if "未找到" in answer or "无法" in answer:
        return "low"
    if len(docs) >= 3 and _score_avg(docs) > 0.3:
        return "high"
    return "medium"


def _score_avg(docs: list[Chunk]) -> float:
    scores = [d.score for d in docs if isinstance(d.score, (int, float))]
    return sum(scores) / len(scores) if scores else 0
