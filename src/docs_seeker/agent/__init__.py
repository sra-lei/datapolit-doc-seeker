"""docs-seeker - Agentic RAG（M1）

开环单轮检索 → 闭环：思考 → 选工具检索 → 观察证据 → 判断充分性 → 成文/拒答。

分层约束（见 docs/agentic-rag-m1-plan.md）：
- 传输可靠性（重试/熔断/超时/降级）全部复用 infra LLMGateway，本包不重复实现；
- 本包只做编排：动作协议解析、工具执行、预算/步数治理、可审计 trace；
- 默认 AGENT_ENABLED=false 关闭；chat_service 里任何异常都回退旧单轮管线。

模块：
- agent_loop：流式对话实验脚本（M1 未接工具，保留探索）；
- models/adapter：中立消息模型与 gateway 响应映射；
- tools：retrieve / lookup_article 运行时工具（+ M2 native schema 占位）；
- prompts/runner：content-JSON 协议与 ReAct 循环（M1 主路径）。
"""

from docs_seeker.agent.models import AgentResult, AgentStep, LLMMessage, Role
from docs_seeker.agent.runner import AgentError, AgentRunner

__all__ = [
    "AgentRunner",
    "AgentResult",
    "AgentStep",
    "LLMMessage",
    "Role",
    "AgentError",
]
