"""
docs-seeker - 查询分解（领域服务）
将复杂问题分解为子问题，提高检索召回率
"""

from loguru import logger

from docs_seeker.core.config import prompts, settings
from docs_seeker.domain.interfaces.llm import LLMProvider, LLMRequest, LLMResponse
from docs_seeker.domain.models.query import Query
from docs_seeker.infra.llm.gateway import get_llm_gateway

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

    def _call(self, prompt: str, max_tokens: int, name: str, budget_guard: bool = False):
        return self.llm.generate(
            LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=settings.llm_decompose_temperature,
                name=name,
                meta={"budget_guard": True} if budget_guard else {},
            )
        )

    def _split_lines(self, response: LLMResponse) -> list[str]:
        """从响应取正文并按行切分（空正文 → 空列表）。"""
        return [q.strip() for q in response.text.strip().split("\n") if q.strip()]

    def decompose(self, question: str) -> Query:
        """分解查询，返回 Query（含子问题列表，含原始问题）"""
        prompt_template = (prompts.get("query_decomposer") or {}).get("system") or _DEFAULT_PROMPT
        prompt = f"{prompt_template}\n\n问题：{question}"
        budget = settings.llm_decompose_max_tokens
        try:
            response = self._call(prompt, budget, "query-decompose", budget_guard=True)
            sub_questions = self._split_lines(response)
            if not sub_questions:
                # 放大预算重试由 BudgetGuardMiddleware 负责（生成 / 改写共用一份实现）；
                # 这里只处理「兜底后仍为空」——不能静默退化成单路检索（历史上正是这样
                # 丢掉了查询分解），如实记 warning 并回退原问题。
                logger.warning(
                    f"查询分解仍为空（finish={response.finish_reason}, max_tokens={budget}）"
                    "→ 退回原问题单路检索（多路召回能力本次未生效）"
                )
                return Query(text=question, sub_queries=[question])
            if question not in sub_questions:
                sub_questions.insert(0, question)
            logger.info(f"查询分解: '{question[:30]}...' → {len(sub_questions)} 个子问题")
            return Query(text=question, sub_queries=sub_questions[:4])
        except Exception as e:
            logger.warning(f"查询分解失败，使用原问题: {e}")
            return Query(text=question, sub_queries=[question])
