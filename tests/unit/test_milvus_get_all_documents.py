"""MilvusStore.get_all_documents：全量查询必须带回主键 id。

背景（P2-12）：BM25 索引来自全量查询，早前 `output_fields` 漏了 `id` →
BM25 的 chunk 全是空 id → 上游按 `chunk.id or ""` 去重时整条 BM25 召回被归并成
一条（检索质量静默下降，且没有任何报错）。本文件把「必须请求 id」锁进测试。
"""

from unittest.mock import patch

from docs_seeker.infrastructure.database.milvus_client import MilvusStore


class _FakeClient:
    """记录 query 参数并返回带主键的行。"""

    def __init__(self):
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return [{"id": "doc_1", "text": "正文", "chapter": "第一章"}]


def _store(fake: _FakeClient) -> MilvusStore:
    with patch("docs_seeker.infrastructure.database.milvus_client.MilvusClient", return_value=fake):
        return MilvusStore()


def test_get_all_documents_requests_primary_key():
    fake = _FakeClient()
    _store(fake).get_all_documents("chartermate_docs")

    assert "id" in fake.calls[0]["output_fields"]
    # 其余建索引要用的字段仍在
    assert {"text", "source", "chapter", "article"} <= set(fake.calls[0]["output_fields"])


def test_get_all_documents_returns_id_through():
    fake = _FakeClient()
    rows = _store(fake).get_all_documents("chartermate_docs")

    assert rows[0]["id"] == "doc_1"


def test_get_all_documents_returns_empty_on_failure():
    class _Broken:
        def query(self, **kwargs):
            raise RuntimeError("milvus down")

    assert _store(_Broken()).get_all_documents("chartermate_docs") == []
