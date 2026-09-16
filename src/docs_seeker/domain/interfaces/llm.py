"""docs-seeker - LLM 接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMRequest:
    """LLM 调用请求实体

    将 generate 的 7 个散落参数收敛为一个结构体，避免参数列表膨胀
    和调用方写错参数顺序。构造后字段不可变（frozen=True 会导致
    stream_options 注入失败，故不加 frozen）。
    """

    messages: list
    max_tokens: int = 600
    temperature: float = 0.3
    stream: bool = False
    name: str = "llm-call"
    model: str | None = None
    timeout: float | None = None


class LLMProvider(ABC):
    """LLM 抽象接口：答案生成 / 查询分解的统一调用入口"""

    @abstractmethod
    def generate(self, request: LLMRequest) -> Any:
        """调用 LLM 生成

        Args:
            request: LLM 调用请求实体（消息、预算、温度、超时等）

        Returns:
            OpenAI 风格响应对象（response.choices[0].message.content）
        """
        raise NotImplementedError
