"""内置护栏：提示注入 / 话题白名单 / 输出脱敏。

模式表仍在 ``core/security.py``（本层只负责「按挂载点与作用对象决定拦不拦」）。
"""

from __future__ import annotations

from loguru import logger

from docs_seeker.config.settings import settings
from docs_seeker.core.security import check_injection_patterns, check_off_topic, desensitize
from docs_seeker.services.guards.base import (
    SUBJECT_ANSWER,
    SUBJECT_USER_INPUT,
    GuardChain,
    GuardContext,
    GuardVerdict,
)


class InjectionGuard:
    """提示注入检测。

    用户输入命中 → 可拒答（边界挂载）；检索文档 / 送模型的 messages 命中 → **仅告警**。
    后者是换语料后补上的缺口：RAG 里真正的注入通道是**检索回来的文档正文**。
    """

    name = "injection_guard"

    def inspect(self, text: str, ctx: GuardContext) -> GuardVerdict:
        ok, reason = check_injection_patterns(text)
        if ok:
            return GuardVerdict(text=text)
        if ctx.can_block:
            return GuardVerdict(allowed=False, text=text, reason=reason)
        logger.warning(f"[guard:{self.name}] {ctx.subject}@{ctx.mount} 命中注入模式 —— 仅告警，不干预答案（{reason}）")
        return GuardVerdict(text=text, reason=reason)


class TopicPolicyGuard:
    """话题白名单：只对**用户输入**生效（检索文档本身不适用职责范围判断）。"""

    name = "topic_policy"

    def inspect(self, text: str, ctx: GuardContext) -> GuardVerdict:
        if ctx.subject != SUBJECT_USER_INPUT:
            return GuardVerdict(text=text)
        ok, reason = check_off_topic(text)
        if ok:
            return GuardVerdict(text=text)
        if ctx.can_block:
            return GuardVerdict(allowed=False, text=text, reason=reason)
        logger.warning(f"[guard:{self.name}] {ctx.subject}@{ctx.mount} 命中话题限制（{reason}）")
        return GuardVerdict(text=text, reason=reason)


class PIIRedactionGuard:
    """输出脱敏：只对**最终答案**生效（改写信封里的文本）。"""

    name = "pii_redaction"

    def inspect(self, text: str, ctx: GuardContext) -> GuardVerdict:
        if ctx.subject != SUBJECT_ANSWER:
            return GuardVerdict(text=text)
        cleaned, found = desensitize(text)
        if found:
            logger.info(f"[guard:{self.name}] 输出脱敏：{'、'.join(found)}")
        return GuardVerdict(text=cleaned, reason="; ".join(found))


_REGISTRY = {
    "injection_guard": InjectionGuard,
    "topic_policy": TopicPolicyGuard,
    "pii_redaction": PIIRedactionGuard,
}


def build_guard_chain(names: list[str] | None = None) -> GuardChain:
    """按名字构建护栏链路；``names`` 为空时取配置 ``LLM_GUARDS``（再空 = 全部内置）。"""
    if names is None:
        names = [n.strip() for n in (settings.llm_guards or "").split(",") if n.strip()]
    if not names:
        names = list(_REGISTRY)
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        logger.warning(f"未知的 LLM guard {unknown}，已忽略")
    return GuardChain([_REGISTRY[n]() for n in names if n in _REGISTRY])


_guard_chain: GuardChain | None = None


def get_guard_chain() -> GuardChain:
    """进程内单例：边界挂载与 client 内挂载共用同一份护栏配置。"""
    global _guard_chain
    if _guard_chain is None:
        _guard_chain = build_guard_chain()
    return _guard_chain
