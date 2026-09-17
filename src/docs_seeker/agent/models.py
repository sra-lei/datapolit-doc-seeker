"""Agent 领域模型（M1）。

合并自两侧骨架：
- Role 枚举与 used_fallback 来自对话循环骨架；
- LLMMessage 是 client 响应的中立映射（隔离 vendor 字段）；
- AgentStep 是每一步的可审计记录；AgentResult 带证据列表与充分性裁决。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from docs_seeker.domain.models.chunk import Chunk


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """LLM 一步响应的中立映射。"""

    content: str = ""
    reasoning: str | None = None  # reasoning_content：只入 trace，不回灌 prompt
    tool_calls: list[dict] = field(default_factory=list)  # native 协议时用；M1 走 content-JSON
    finish_reason: str | None = None


@dataclass
class AgentStep:
    """每一步的可审计记录（M1 验收的 trace）。"""

    idx: int
    thought: str  # 为什么选这个动作
    action: str  # retrieve / lookup_article / final / parse_error
    action_input: dict
    observation: str = ""  # tool_result：runner 本地执行工具得到，不是 LLM 返回的
    elapsed_ms: int = 0
    error: str | None = None  # 语义级失败（JSON 解析坏等），拼进下一轮自纠


@dataclass
class AgentResult:
    answer: str
    confidence: str  # M1：沿用 compute_confidence 启发式；后续换成充分性判断输出
    steps: list[AgentStep]
    evidence: list[Chunk] = field(default_factory=list)
    sufficient: bool = True  # 成文阶段核实后的最终裁决（防 false abstain）
    used_fallback: bool = False  # 是否回退了旧单轮管线（chat_service 标记）
