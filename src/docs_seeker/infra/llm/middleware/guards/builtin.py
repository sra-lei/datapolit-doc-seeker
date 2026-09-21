"""内置护栏：模式表 + 护栏实现 + 链路装配。

下半部是模式表与纯检测函数（提示注入 / 话题白名单 / 敏感信息脱敏）；上半部
护栏类按**挂载点与作用对象**决定拦不拦 / 改不改写。
"""

from __future__ import annotations

import re

from loguru import logger

from docs_seeker.config.settings import settings
from docs_seeker.infra.llm.middleware.guards.base import (
    SUBJECT_ANSWER,
    SUBJECT_USER_INPUT,
    GuardChain,
    GuardContext,
    GuardVerdict,
)

# ============================ 模式表与检测函数 ============================

INJECTION_PATTERNS = [
    r"忽略(上述|之前|以上|前面|系统)",
    r"ignore\s+(the\s+)?(above|previous|all)",
    r"忘记.*指令",
    r"forget\s+(the\s+)?instructions",
    r"你是.*不是.*助手",
    r"you\s+are\s+(not|no\s+longer)\s+(an?\s+)?assistant",
    r"扮演.*角色",
    r"act\s+as\s+(a|an)",
    r"输出.*系统.*提示",
    r"(print|output|show|display)\s+(the\s+)?(system\s+)?prompt",
    r"切换.*模式",
    r"switch\s+mode",
    r"忽略.*限制",
    r"ignore\s+restrictions",
]

OFF_TOPIC_PATTERNS = [
    r"(写|生成|编|创作).*(诗|小说|故事|代码|歌词|文章|剧本)",
    r"(write|generate|create|compose).*(poem|story|code|lyrics|article)",
    r"(怎么|如何|怎样).*(攻击|破解|入侵|黑入|越狱)",
    r"(how\s+to|teach\s+me).*(hack|attack|crack|bypass)",
    r"(翻译|translate)\s",
]


def check_injection_patterns(question: str) -> tuple[bool, str]:
    """提示注入检测（不含话题白名单）"""
    question_lower = question.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question_lower):
            return False, "检测到提示注入模式，请求已拒绝"
    return True, ""


def check_off_topic(question: str) -> tuple[bool, str]:
    """话题白名单检测（是否超出职责范围）"""
    question_lower = question.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, question_lower):
            return False, "该问题超出了我的职责范围（公司政策查询）"
    return True, ""


def check_injection(question: str) -> tuple[bool, str]:
    """输入侧综合检测：先提示注入、后话题白名单。

    （保留此组合入口以兼容既有调用方；护栏链路分别用上面两个细分函数。）
    """
    ok, reason = check_injection_patterns(question)
    if not ok:
        return ok, reason
    return check_off_topic(question)


SENSITIVE_PATTERNS = [
    (re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"), "身份证号"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "手机号"),
    (re.compile(r"\b0\d{2,3}-?\d{7,8}\b"), "电话号"),
    (re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b"), "邮箱"),
    (re.compile(r"(?:月薪|年薪|薪资|工资|底薪)\s*\d{3,7}"), "薪资信息"),
    (re.compile(r"\b\d{16,19}\b"), "疑似银行卡号"),
]


def desensitize(text: str) -> tuple[str, list]:
    found = []
    clean_text = text
    for pattern, label in SENSITIVE_PATTERNS:
        matches = pattern.findall(clean_text)
        if matches:
            found.append(f"{label} (共{len(matches)}处)")
            clean_text = pattern.sub("***", clean_text)
    return clean_text, found


def sanitize_output(text: str) -> str:
    clean_text, _ = desensitize(text)
    return clean_text


# ============================== 内置护栏实现 ==============================


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
