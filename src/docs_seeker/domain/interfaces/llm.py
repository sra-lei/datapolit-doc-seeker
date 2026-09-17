"""docs-seeker - LLM 接口

LLM 调用契约。数据模型（``LLMRequest`` / ``LLMResponse``）在 ``domain.models.llm``，
本模块只定义抽象接口 ``LLMProvider``：上层依赖它，不感知具体 provider、SDK 结构或
客户端策略（重试 / 熔断 / 降级 / 护栏都藏在客户端与 middleware 之后）。

设计约定（2026-09 重构 Phase 0，方案见 ``docs/llm-gateway-guard-refactor.md``）：
- 参数**不做白名单**：未知 provider 参数走 ``extra`` 原样透传；
- 返回**不降维**：``raw`` 保留原始响应，便捷字段只是视图；
- 调用**可归因**：降级 / 重试 / 生效的 middleware 在信封上显式可见。
"""

from abc import ABC, abstractmethod

from docs_seeker.domain.models.llm import LLMRequest, LLMResponse


class LLMProvider(ABC):
    """LLM 抽象接口：答案生成 / 查询分解 / agent 推理的统一调用入口"""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """调用 LLM 生成

        Args:
            request: LLM 调用请求实体（消息、预算、温度、超时、透传参数等）

        Returns:
            LLMResponse：便捷字段 + ``raw`` 原始响应 + 调用元信息
        """
        raise NotImplementedError
