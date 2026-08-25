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

    def generate(self, question: str, docs: list[Chunk], conversation_history: list[dict] | None = None) -> tuple[str, str]:
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
        try:
            response = self.llm.generate(messages=messages, max_tokens=600, temperature=0.3)
            answer = response.choices[0].message.content.strip()
            if "未找到" in answer or "无法" in answer:
                confidence = "low"
            elif len(docs) >= 3 and _score_avg(docs) > 0.3:
                confidence = "high"
            else:
                confidence = "medium"
            logger.info(f"答案生成: confidence={confidence} docs={len(docs)} len={len(answer)}")
            return answer, confidence
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"答案生成失败: {e}", "low"


def _score_avg(docs: list[Chunk]) -> float:
    scores = [d.score for d in docs if isinstance(d.score, (int, float))]
    return sum(scores) / len(scores) if scores else 0
