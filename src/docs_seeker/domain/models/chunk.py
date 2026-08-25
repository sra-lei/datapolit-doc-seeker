"""docs-seeker - 文档块实体（检索返回的基本单元）"""
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """文档块：正文片段 + 结构化元信息

    对应 Milvus collection 中的一条记录，三个检索器（dense/bm25/summary）
    的返回结果统一为该实体；score/sources 由融合或各检索器补充。
    """

    id: str = ""
    text: str = ""
    source: str = ""
    pages: str = ""
    chapter: str = ""
    chapter_title: str = ""
    section: str = ""
    section_title: str = ""
    article: str = ""
    article_title: str = ""
    distance: float = 0.0
    score: float = 0.0
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """从 Milvus 返回的 dict 构造实体（自动忽略未知字段）"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)
