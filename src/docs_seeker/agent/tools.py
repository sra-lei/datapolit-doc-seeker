"""Agent 可用工具（M1 两个，都复用现有检索器，不引入新检索实现）。

- retrieve：三路混合检索（dense + BM25 + 摘要引导，RRF 融合），语义查法；
- lookup_article：结构化定位（第X章/节/条/款/项），先解析问题里的结构词，
  带 meta_filter 调同一个混合检索器；解析不出结构词时明确提示改用 retrieve。

工具返回 (observation, chunks)：observation 是给下一步 LLM 看的紧凑摘要，
chunks 进 runner 的累计证据（去重、字符预算截断后用于最终成文）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from docs_seeker.infra.retrieval.metadata_filter import matches_metadata
from docs_seeker.models.chunk import Chunk
from docs_seeker.services.metadata import parse_question_metadata

_SNIPPET_CHARS = 350


@dataclass
class ToolOutput:
    observation: str
    chunks: list[Chunk]


class AgentTool(Protocol):
    name: str
    description: str

    def run(self, query: str, top_k: int) -> ToolOutput: ...


def _format_observation(chunks: list[Chunk], empty_hint: str) -> str:
    if not chunks:
        return f"未检索到相关内容。{empty_hint}"
    lines = [f"检索到 {len(chunks)} 条相关资料："]
    for i, c in enumerate(chunks, 1):
        loc = " ".join(x for x in (c.chapter, c.article) if x)
        head = f"[{i}] {loc} " if loc else f"[{i}] "
        snippet = c.text.replace("\n", " ")[:_SNIPPET_CHARS]
        lines.append(head + snippet)
    return "\n".join(lines)


class RetrieveTool:
    name = "retrieve"
    description = "三路混合语义检索。参数 query：改写后的检索词（适合同义表述、概念性问题）"

    def __init__(self, retriever):
        self._retriever = retriever

    def run(self, query: str, top_k: int) -> ToolOutput:
        chunks = self._retriever.search(query, top_k=top_k)
        return ToolOutput(_format_observation(chunks, "可换一种表述再试 retrieve。"), chunks)


class LookupArticleTool:
    name = "lookup_article"
    description = (
        "按结构化定位词（第X章/第X节/第X条/第X款/第X项）精确检索。参数 query：必须包含明确的章/节/条/款/项编号"
    )

    def __init__(self, retriever):
        self._retriever = retriever

    def run(self, query: str, top_k: int) -> ToolOutput:
        meta = parse_question_metadata(query)
        if not meta:
            return ToolOutput(
                "问题里没有解析出章/节/条/款/项编号，无法精确定位；请改用 retrieve。",
                [],
            )
        chunks = self._retriever.search(query, top_k=top_k, meta_filter=meta)
        # 混合检索器在结构过滤无命中时会静默回退全量语义检索。对「精确定位」工具来说，
        # 那些不匹配编号的近似结果不是本条内容——标出来且不进证据，防止污染成文。
        matched = [c for c in chunks if matches_metadata(c.to_dict(), meta)]
        if not matched:
            return ToolOutput(
                "未找到该章/节/条/款/项编号的内容（返回的语义近似结果不计入证据）；"
                "请确认编号，或改用 retrieve 做语义检索。",
                [],
            )
        return ToolOutput(
            _format_observation(matched, "该编号可能不存在，可改用 retrieve 做语义检索。"),
            matched,
        )


def default_tools(retriever) -> dict[str, AgentTool]:
    tools = [RetrieveTool(retriever), LookupArticleTool(retriever)]
    return {t.name: t for t in tools}


# ---- M2 预留：native function calling 的工具 schema（占位，M1 未使用）----
# 搜索互联网/翻译是骨架阶段的规划工具；M1 闭环只用上面的 retrieve / lookup_article。
TOOL_SCHEMAS = [
    {"name": "search", "description": "搜索互联网"},
    {"name": "translate", "description": "翻译文本"},
]
