"""
docs-seeker - 查询分解
将复杂问题分解为子问题，提高检索召回率
"""

from loguru import logger

from docs_seeker.core.config import prompts
from docs_seeker.domain.interfaces.llm import LLMProvider
from docs_seeker.domain.models.query import Query
from docs_seeker.infrastructure.llm.gateway import get_llm_gateway

_DEFAULT_PROMPT = (
    "你是一个查询分解助手。将以下问题分解为 2-4 个更具体的子问题，用于多路检索。\n"
    "只返回子问题列表，每行一个，不要编号，不要解释。\n"
    "如果问题已经足够简单，直接返回原问题。"
)


class QueryDecomposer:
    """查询分解器：用 LLM 将复杂问题拆分为多个子问题"""

    def __init__(self, llm: LLMProvider | None = None):
        # 允许注入 LLM（deps 组装点传入）；缺省时走全局网关单例
        self.llm = llm or get_llm_gateway()

    def decompose(self, question: str) -> Query:
        """分解查询，返回 Query（含子问题列表，含原始问题）"""
        prompt_template = (prompts.get("query_decomposer") or {}).get("system") or _DEFAULT_PROMPT
        prompt = f"{prompt_template}\n\n问题：{question}"
        try:
            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                name="query-decompose",
            )
            content = response.choices[0].message.content.strip()
            sub_questions = [q.strip() for q in content.split("\n") if q.strip()]
            if not sub_questions:
                return Query(text=question, sub_queries=[question])
            if question not in sub_questions:
                sub_questions.insert(0, question)
            logger.info(f"查询分解: '{question[:30]}...' → {len(sub_questions)} 个子问题")
            return Query(text=question, sub_queries=sub_questions[:4])
        except Exception as e:
            logger.warning(f"查询分解失败，使用原问题: {e}")
            return Query(text=question, sub_queries=[question])
