from dataclasses import dataclass
from enum import Enum


@dataclass
class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class LLMMessage:  # gateway 响应的中立映射
    content: str  # 正文
    reasoning: str | None  # reasoning_content，只入 trace，不回灌 prompt
    tool_calls: list[dict]  # native 协议时用；M1 走 JSON 协议则为空
    finish_reason: str | None


@dataclass
class AgentStep:  # 每一步的可审计记录（M1 验收要的 trace）
    idx: int
    thought: str  # 为什么选这个动作
    action: str  # retrieve / lookup_article / final_answer / abstain
    action_input: dict
    observation: str  # ★ tool_result 是 runner 本地执行工具得到的，不是 LLM 返回的
    elapsed_ms: int
    error: str | None = None  # 语义级失败（JSON 解析坏等），拼进下一轮自纠


@dataclass
class AgentResult:
    answer: str
    confidence: str  # M1：沿用旧启发式兜底；后续换成充分性判断输出
    steps: list[AgentStep]
    used_fallback: bool  # 是否回退了旧单轮管线
