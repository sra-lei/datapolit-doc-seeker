"""docs-seeker - 文档实体（文档级信息）"""
from dataclasses import dataclass, field


@dataclass
class Document:
    """文档实体：摘要集合中的文档级元信息

    用于摘要引导检索的第一阶段（在摘要 collection 中命中相关文档/章节），
    再由 chapter 信息去正文集合召回对应 Chunk。
    """

    id: str = ""
    title: str = ""
    chapter: str = ""
    chapter_title: str = ""
    summary: str = ""
    chunk_ids: list[str] = field(default_factory=list)
