"""RAGPipeline 的结构化元数据接线：从原问题解析 → 透传到每次子问题检索。

不依赖 Milvus / LLM / Redis（decomposer / retriever / generator 全为测试替身）。
"""

from docs_seeker.domain.models.query import Query
from docs_seeker.domain.services.rag_pipeline import RAGPipeline


class FakeDecomposer:
    def __init__(self, sub_queries):
        self._sub_queries = sub_queries

    def decompose(self, question: str) -> Query:
        return Query(text=question, sub_queries=list(self._sub_queries))


class FakeRetriever:
    def __init__(self):
        self.calls: list[dict] = []

    def search(self, query, top_k=10, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return []


class FakeGenerator:
    def generate(self, question, docs, conversation_history=None):
        return "答案", "high"


def _pipeline(sub_queries):
    retriever = FakeRetriever()
    pipeline = RAGPipeline(
        retriever=retriever,
        decomposer=FakeDecomposer(sub_queries),
        generator=FakeGenerator(),
    )
    return pipeline, retriever


def test_metadata_parsed_from_original_question_and_applied_to_all_sub_queries():
    """子问题（LLM 改写）常丢掉条号 → 必须从原问题解析，且对所有子问题统一生效"""
    pipeline, retriever = _pipeline(["第三十六条的内容", "领导水平指什么"])
    pipeline.run("第三十六条是什么内容？", top_k=10)

    assert [c["query"] for c in retriever.calls] == ["第三十六条的内容", "领导水平指什么"]
    assert all(c["meta_filter"] == {"article": ["第三十六条", "第36条"]} for c in retriever.calls)


def test_pipeline_without_structure_word_passes_none():
    pipeline, retriever = _pipeline(["公司对员工的要求"])
    pipeline.run("公司对员工有哪些要求？", top_k=10)

    assert retriever.calls[0]["meta_filter"] is None
