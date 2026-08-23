"""docs-seeker - LLM 接口"""
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """LLM 抽象接口：答案生成 / 查询分解的统一调用入口"""

    @abstractmethod
    def generate(
        self,
        messages: list,
        max_tokens: int = 600,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> Any:
        """调用 LLM 生成

        Args:
            messages: OpenAI 风格消息列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            stream: 是否流式

        Returns:
            OpenAI 风格响应对象（response.choices[0].message.content）
        """
        raise NotImplementedError
