"""docs-seeker - 答案生成器（应用服务）"""

from loguru import logger

from docs_seeker.core.config import prompts
from docs_seeker.domain.interfaces.llm import LLMProvider
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.infrastructure.llm.gateway import get_llm_gateway

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
        # 允许注入 LLM（deps 组装点传入）；缺省时走全局网关单例
        self.llm = llm or get_llm_gateway()

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

    def generate(
        self, question: str, docs: list[Chunk], conversation_history: list[dict] | None = None
    ) -> tuple[str, str]:
        messages = self._build_messages(question, docs, conversation_history)
        try:
            # name：Langfuse generation 观测名（稳定、动词开头，便于过滤/评估器定位）
            response = self.llm.generate(messages=messages, max_tokens=600, temperature=0.3, name="generate-response")
            answer = response.choices[0].message.content.strip()
            confidence = compute_confidence(answer, docs)
            logger.info(f"答案生成: confidence={confidence} docs={len(docs)} len={len(answer)}")
            return answer, confidence
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"答案生成失败: {e}", "low"

    def generate_stream(self, question: str, docs: list[Chunk], conversation_history: list[dict] | None = None):
        """流式生成：逐段产出增量文本（str）。

        异常不在此捕获（由调用方决定如何收尾），最后一段文本产出后自然结束。
        """
        messages = self._build_messages(question, docs, conversation_history)
        stream = self.llm.generate(messages=messages, max_tokens=600, temperature=0.3, stream=True, name="generate-response")
        for chunk in stream:
            delta = _extract_delta(chunk)
            if delta:
                yield delta


def _extract_delta(chunk) -> str:
    """从 OpenAI 风格流式 chunk 中提取增量文本（兼容 reasoning 模型先出 reasoning_content 的情形）"""
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def compute_confidence(answer: str, docs: list[Chunk]) -> str:
    """按答案措辞与命中文档质量评估置信度（与一次性生成共用同一口径）"""
    if "未找到" in answer or "无法" in answer:
        return "low"
    if len(docs) >= 3 and _score_avg(docs) > 0.3:
        return "high"
    return "medium"


def _score_avg(docs: list[Chunk]) -> float:
    scores = [d.score for d in docs if isinstance(d.score, (int, float))]
    return sum(scores) / len(scores) if scores else 0
