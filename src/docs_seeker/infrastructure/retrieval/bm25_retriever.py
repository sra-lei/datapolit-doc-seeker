"""
docs-seeker - BM25 稀疏检索
基于 jieba 分词 + TF-IDF/BM25 算法

索引生命周期（v2 修复）：
- 进程内共享：所有 BM25Retriever 实例共用同一份索引（class-level 共享态），
  避免 ChatService / Warmup 各自持有一份全量索引（每份都等于
  一次 Milvus 全量扫描 + 全量分词）。
- 懒构建 + 线程安全：首次检索时用 double-checked locking 构建一次，后续请求
  只做 O(1) 的已构建判断，不会每次请求都全量扫描 Milvus。
- 新鲜度：索引构建后，每隔 bm25_refresh_seconds 用 Milvus 的 row count
  （statistics 调用，远便宜于全量 scan）做一次变更探测，文档数不一致才触发
  重建；doc-kit 入库后也可显式调用 refresh() 立即重建。
"""

import math
import threading
import time
from collections import Counter
from typing import Any

import jieba
from langfuse import get_client, observe
from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.retriever import Retriever
from docs_seeker.domain.models.chunk import Chunk
from docs_seeker.infrastructure.database.milvus_client import get_milvus_store


class BM25Retriever(Retriever):
    """BM25 检索器：从 Milvus 拉全量文档 + 本地 BM25 计算（进程内共享索引）"""

    # ---------------- 进程内共享索引（class-level） ----------------
    # 无论创建多少个 BM25Retriever 实例，都只维护这一份索引；
    # 重建时整体替换引用（不原地修改），检索侧持有本地快照即可保证一致性。
    _shared_docs: list[dict] = []
    _shared_index: dict[str, list[tuple[int, int]]] = {}
    _shared_avgdl: float = 0.0
    _shared_doc_count: int = 0
    _shared_built: bool = False
    _shared_built_at: float = 0.0  # 上次构建时间（monotonic）
    _shared_last_checked_at: float = 0.0  # 上次新鲜度探测时间（monotonic）
    _shared_lock = threading.Lock()  # 串行化首次构建 / 刷新

    def __init__(self):
        self.milvus = get_milvus_store()
        self.collection_name = settings.collection_name

    # ---------------- 索引构建 ----------------

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in jieba.cut_for_search(text) if len(w.strip()) > 1]

    def _do_build(self) -> None:
        """执行真正的全量构建（调用方必须已持有 _shared_lock）"""
        docs = self.milvus.get_all_documents(self.collection_name, limit=settings.bm25_max_docs)
        avgdl = sum(len(d.get("text", "")) for d in docs) / max(1, len(docs))

        index: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            tf = Counter(self._tokenize(doc.get("text", "")))
            for term, freq in tf.items():
                index.setdefault(term, []).append((i, freq))

        # 整体替换引用，保证检索侧快照一致性。
        # 注意：必须写 class 属性（type(self)），若写 self.xxx 会生成实例属性，
        # 导致其它实例读不到共享索引而各自重建。
        cls = type(self)
        cls._shared_docs = docs
        cls._shared_index = index
        cls._shared_avgdl = avgdl
        cls._shared_doc_count = len(docs)
        cls._shared_built = True
        cls._shared_built_at = time.monotonic()
        logger.info(f"BM25 索引构建完成: docs={len(docs)} terms={len(index)}")

    def build_index(self, force: bool = False) -> None:
        """构建 / 重建共享索引。

        - force=False（默认）：仅当索引尚未构建时构建（double-checked，线程安全），
          已构建则直接返回 —— 这是"每次请求不重建"的保证。
        - force=True：无条件全量重建（启动预热 / 显式刷新用）。
        """
        with self._shared_lock:
            if force or not self._shared_built:
                self._do_build()

    def refresh(self) -> None:
        """强制重建索引。

        文档（doc-kit）入库/更新后可调用本方法，使 BM25 立即感知最新数据；
        否则将由 search 内的周期新鲜度检查兜底。
        """
        self.build_index(force=True)

    def _maybe_refresh(self) -> None:
        """廉价新鲜度检查：每隔 bm25_refresh_seconds 至多探测一次。

        通过 Milvus row count（statistics 调用，远低于全量 scan）与索引文档数
        对比，不一致才触发重建，避免每次请求都做全量扫描。
        bm25_refresh_seconds <= 0 时关闭自动刷新。
        """
        interval = settings.bm25_refresh_seconds
        if interval <= 0 or not self._shared_built:
            return
        now = time.monotonic()
        if now - self._shared_last_checked_at < interval:
            return
        type(self)._shared_last_checked_at = now
        try:
            current = self.milvus.count(self.collection_name)
        except Exception:
            return
        if current >= 0 and current != self._shared_doc_count:
            logger.info(f"检测到文档数变化 {self._shared_doc_count} -> {current}，重建 BM25 索引")
            self.build_index(force=True)

    # ---------------- 检索 ----------------

    @observe(name="retrieve-bm25", as_type="retriever", capture_input=False, capture_output=False)
    def search(self, query: str, top_k: int = 10, **kwargs: Any) -> list[Chunk]:
        """BM25 检索

        Args:
            query: 用户查询文本
            top_k: 返回数量

        Returns:
            按相关性降序的 Chunk 列表
        """
        get_client().update_current_span(input={"query": query, "top_k": top_k})
        # 线程安全懒构建：已构建时仅一次加锁 + 标志判断，开销可忽略
        self.build_index()
        if not self._shared_built or not self._shared_docs:
            return []

        # 周期新鲜度检查（有节流，不阻塞每次请求）
        self._maybe_refresh()
        if not self._shared_docs:
            return []

        # 本地快照：重建是整体替换引用，快照保证本次检索看到一致的数据
        docs = self._shared_docs
        index = self._shared_index
        avgdl = self._shared_avgdl
        doc_count = self._shared_doc_count

        query_tokens = self._tokenize(query)
        k1 = 1.5
        b = 0.75
        scores: list[tuple[int, float]] = []

        for i, doc in enumerate(docs):
            doc_len = len(doc.get("text", ""))
            score = 0.0
            for term in query_tokens:
                postings = index.get(term)
                if not postings:
                    continue
                # TF
                tf = 0
                for doc_id, freq in postings:
                    if doc_id == i:
                        tf = freq
                        break
                if tf == 0:
                    continue
                # IDF
                df = len(postings)
                idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
                # BM25 score
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scores[:top_k]:
            chunk = Chunk.from_dict(docs[i])
            chunk.score = score
            results.append(chunk)
        logger.info(f"BM25 检索完成: query='{query[:30]}...' top_k={top_k} hits={len(results)}")
        get_client().update_current_span(output={"hits": len(results)})
        return results
