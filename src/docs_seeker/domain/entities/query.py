"""docs-seeker - 查询实体"""
from dataclasses import dataclass, field


@dataclass
class Query:
    """查询实体：原始问题 + 分解后的子问题列表"""

    text: str
    sub_queries: list[str] = field(default_factory=list)
