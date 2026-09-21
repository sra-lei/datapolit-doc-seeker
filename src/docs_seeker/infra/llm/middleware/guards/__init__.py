"""应用级护栏（guard）：可插拔的文本检查 / 改写中间件。

护栏全部内聚在 LLM middleware 子包：协议/链路在 ``base.py``，模式表在
``security.py``，内置实现在 ``builtin.py``，挂到 transport 链上的客户端
适配器在 ``adapter.py``（``LLMGuardMiddleware``）。

两个挂载点共用同一份护栏配置：

- **pipeline 边界**（``ChatService``）：用户输入可拒答、答案可改写（脱敏）、
  检索文档仅告警；
- **client 内**（``LLMGuardMiddleware``，transport 链最外层）：扫描送 provider
  的 messages，**仅告警不短路**，让 agent 循环内的每次 LLM 调用也经过护栏。
"""

from docs_seeker.infra.llm.middleware.guards.adapter import LLMGuardMiddleware
from docs_seeker.infra.llm.middleware.guards.base import (
    ANSWER_CTX,
    DOCUMENT_CTX,
    LLM_MESSAGES_CTX,
    MOUNT_BOUNDARY,
    MOUNT_GATEWAY,
    SUBJECT_ANSWER,
    SUBJECT_DOCUMENT,
    SUBJECT_LLM_MESSAGES,
    SUBJECT_USER_INPUT,
    USER_INPUT_CTX,
    Guard,
    GuardChain,
    GuardContext,
    GuardVerdict,
)
from docs_seeker.infra.llm.middleware.guards.builtin import (
    InjectionGuard,
    PIIRedactionGuard,
    TopicPolicyGuard,
    build_guard_chain,
    get_guard_chain,
)

__all__ = [
    "ANSWER_CTX",
    "DOCUMENT_CTX",
    "LLM_MESSAGES_CTX",
    "LLMGuardMiddleware",
    "MOUNT_BOUNDARY",
    "MOUNT_GATEWAY",
    "SUBJECT_ANSWER",
    "SUBJECT_DOCUMENT",
    "SUBJECT_LLM_MESSAGES",
    "SUBJECT_USER_INPUT",
    "USER_INPUT_CTX",
    "Guard",
    "GuardChain",
    "GuardContext",
    "GuardVerdict",
    "InjectionGuard",
    "PIIRedactionGuard",
    "TopicPolicyGuard",
    "build_guard_chain",
    "get_guard_chain",
]
