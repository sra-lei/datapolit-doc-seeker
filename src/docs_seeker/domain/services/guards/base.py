"""应用级护栏（guard）协议与链路。

护栏是可插拔的「文本检查 / 改写」中间件。按**挂载点**与**作用对象**决定行为：

| 挂载点 | 作用对象 | 策略 |
|---|---|---|
| `pipeline_boundary`（chat 入口/出口） | 用户输入 / 检索文档 / 最终答案 | 用户输入可**短路拒答**；答案可**改写**（脱敏）；文档**仅告警** |
| `gateway_inner`（transport middleware） | 送 provider 的完整 messages | **仅告警不短路** —— messages 里混着系统提示与检索证据，误判代价高于收益（评审已决 2026-09-17） |

新增护栏只需实现 ``inspect(text, ctx) -> GuardVerdict`` 并注册（见 ``builtin.py``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# 挂载点
MOUNT_BOUNDARY = "pipeline_boundary"
MOUNT_GATEWAY = "gateway_inner"

# 作用对象
SUBJECT_USER_INPUT = "user_input"
SUBJECT_DOCUMENT = "document"
SUBJECT_ANSWER = "answer"
SUBJECT_LLM_MESSAGES = "llm_messages"


@dataclass(frozen=True)
class GuardContext:
    """护栏上下文：告诉护栏「这次检查的是哪儿的什么东西、能不能拦」"""

    mount: str = MOUNT_BOUNDARY
    subject: str = SUBJECT_USER_INPUT
    can_block: bool = False


USER_INPUT_CTX = GuardContext(mount=MOUNT_BOUNDARY, subject=SUBJECT_USER_INPUT, can_block=True)
DOCUMENT_CTX = GuardContext(mount=MOUNT_BOUNDARY, subject=SUBJECT_DOCUMENT, can_block=False)
ANSWER_CTX = GuardContext(mount=MOUNT_BOUNDARY, subject=SUBJECT_ANSWER, can_block=False)
LLM_MESSAGES_CTX = GuardContext(mount=MOUNT_GATEWAY, subject=SUBJECT_LLM_MESSAGES, can_block=False)


@dataclass
class GuardVerdict:
    """护栏裁决。

    ``allowed=False`` → 调用方按策略处理（边界处拒答）；``text`` 为**可能被改写**的
    文本（脱敏等），调用方应使用裁决后的 ``text``。
    """

    allowed: bool = True
    text: str = ""
    reason: str = ""


@runtime_checkable
class Guard(Protocol):
    name: str

    def inspect(self, text: str, ctx: GuardContext) -> GuardVerdict: ...


class GuardChain:
    """按顺序执行护栏：遇到拦截立即短路，改写逐级传递。"""

    def __init__(self, guards: list[Guard]):
        self.guards = list(guards)

    def inspect(self, text: str, ctx: GuardContext) -> GuardVerdict:
        current = text
        for guard in self.guards:
            verdict = guard.inspect(current, ctx)
            if not verdict.allowed:
                return GuardVerdict(allowed=False, text=current, reason=verdict.reason or f"被 {guard.name} 拦截")
            if verdict.text:
                current = verdict.text
        return GuardVerdict(allowed=True, text=current)

    def inspect_many(self, texts: list[str], ctx: GuardContext) -> GuardVerdict:
        """把多段文本拼起来检查（如检索到的多篇文档正文）"""
        return self.inspect("\n".join(t for t in texts if t), ctx)

    @property
    def names(self) -> list[str]:
        return [g.name for g in self.guards]
