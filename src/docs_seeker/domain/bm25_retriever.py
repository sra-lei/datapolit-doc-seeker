"""
docs-seeker - BM25 稀疏检索
基于 jieba 分词 + TF-IDF/BM25 算法
"""
import math
from collections import Counter
import jieba
from loguru import logger

from docs_seeker.infra.milvus_store import get_milvus_store
from docs_seeker.config import settings


class BM25Retriever:
    """BM25 检索器：从 Milvus 拉全量文档 + 本地 BM25 计算"""

    def __init__(self):
        self.milvus = get_milvus_store()
        self.collection_name = settings.collection_name
        self._docs: list[dict] = []
        self._bm25_index: dict = {}
        self._avgdl: float = 0.0
        self._doc_count: int = 0

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in jieba.cut_for_search(text) if len(w.strip()) > 1]

    def build_index(self):
        """从 Milvus 拉全量文档，构建 BM25 索引"""
        docs = self.milvus.get_all_documents(self.collection_name)
        self._docs = docs
        self._doc_count = len(docs)
        self._avgdl = sum(len(d.get("text", "")) for d in docs) / max(1, len(docs))

        # 构建 inverted index
        self._bm25_index = {}
        for i, doc in enumerate(docs):
            tokens = self._tokenize(doc.get("text", ""))
            tf = Counter(tokens)
            for term, freq in tf.items():
                if term not in self._bm25_index:
                    self._bm25_index[term] = []
                self._bm25_index[term].append((i, freq))
        logger.info(f"BM25 索引构建完成: docs={self._doc_count} terms={len(self._bm25_index)}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 检索

        Args:
            query: 用户查询文本
            top_k: 返回数量

        Returns:
            [{id, text, source, chapter, ..., score}, ...]
        """
        if not self._docs:
            self.build_index()
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        k1 = 1.5
        b = 0.75
        scores: list[tuple[int, float]] = []

        for i, doc in enumerate(self._docs):
            doc_len = len(doc.get("text", ""))
            score = 0.0
            for term in query_tokens:
                if term not in self._bm25_index:
                    continue
                # TF
                tf = 0
                for doc_id, freq in self._bm25_index[term]:
                    if doc_id == i:
                        tf = freq
                        break
                if tf == 0:
                    continue
                # IDF
                df = len(self._bm25_index[term])
                idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)
                # BM25 score
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self._avgdl))
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scores[:top_k]:
            doc = self._docs[i].copy()
            doc["score"] = score
            results.append(doc)
        logger.info(f"BM25 检索完成: query='{query[:30]}...' top_k={top_k} hits={len(results)}")
        return results
