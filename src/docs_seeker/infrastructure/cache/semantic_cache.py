"""
docs-seeker - 语义缓存
基于 Redis Stack 向量搜索，按问题相似度匹配缓存
"""

import hashlib
import json
from array import array

from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.core.metrics import cache_hits_total, cache_misses_total
from docs_seeker.infrastructure.cache.redis_client import get_redis_client
from docs_seeker.infrastructure.embedding.embedder import get_embedder

_INDEX_NAME = "qa_cache_idx"


class SemanticCache:
    """基于 Redis Stack 向量搜索的语义缓存

    - 开关：环境变量 `SEMANTIC_CACHE_ENABLED`（默认 true；false 时完全不连接 Redis 做缓存读写）
    - 索引维度以真实 embedding 长度为准（不硬编码），模型/维度配置变化时自动重建索引
    """

    def __init__(self):
        self.redis = get_redis_client()
        self._hits = 0
        self._misses = 0
        self._available = False
        self._dim: int | None = None
        self.enabled = settings.semantic_cache_enabled

        if not self.enabled:
            logger.info("语义缓存已禁用（SEMANTIC_CACHE_ENABLED=false）")
            return

        try:
            self.redis.ft(_INDEX_NAME).info()
            self._dim = self._read_index_dim()
            self._available = True
            logger.info(f"语义缓存索引已存在 (DIM={self._dim})")
        except Exception:
            try:
                self._create_index(self._require_dim())
                self._available = True
            except Exception as e:
                logger.warning(f"语义缓存初始化失败，将降级为无缓存模式: {e}")
                self._available = False

    # ---------- 工具 ----------

    @staticmethod
    def _encode_vector(vector: list[float]) -> bytes:
        """float32 编码（与索引 TYPE=FLOAT32 对应）"""
        return array("f", vector).tobytes()

    def _get_embedding(self, text: str) -> list[float]:
        return get_embedder().get_embedding(text)

    def _require_dim(self) -> int:
        """获取真实 embedding 维度（首次调用会请求一次向量化服务）"""
        if self._dim is None:
            # 注意：向量化服务拒绝空文本，需用非空探测文本获取维度
            self._dim = len(self._get_embedding("维度探测"))
        return self._dim

    def _read_index_dim(self) -> int | None:
        """读取已有索引的向量维度"""
        try:
            info = self.redis.ft(_INDEX_NAME).info()
            attrs = info.get("attributes", []) if isinstance(info, dict) else getattr(info, "attributes", [])
            for attr in attrs or []:
                if not isinstance(attr, dict):
                    continue
                vector_index = attr.get("vector_index")
                if isinstance(vector_index, dict):
                    dim = vector_index.get("dims")
                    if dim:
                        return int(dim)
        except Exception as e:
            logger.warning(f"读取语义缓存索引维度失败: {e}")
        return None

    def _create_index(self, dim: int):
        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        schema = (
            TextField("$.question", no_stem=True, as_name="question"),
            TextField("$.answer", no_stem=True, as_name="answer"),
            TextField("$.confidence", no_stem=True, as_name="confidence"),
            VectorField(
                "$.embedding",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
                as_name="embedding",
            ),
        )
        definition = IndexDefinition(prefix=["qa:"], index_type=IndexType.JSON)
        self.redis.ft(_INDEX_NAME).create_index(schema, definition=definition)
        logger.info(f"Redis 语义缓存索引已创建 (DIM={dim})")

    def _ensure_dim(self, embedding: list[float]) -> None:
        """向量维度与索引不一致时删除并重建索引（自愈）"""
        dim = len(embedding)
        if self._dim == dim:
            return
        logger.warning(f"语义缓存向量维度漂移 {self._dim} -> {dim}，重建索引")
        try:
            self.redis.ft(_INDEX_NAME).dropindex()
        except Exception:
            pass
        self._create_index(dim)
        self._dim = dim

    # ---------- 读写 ----------

    def search(self, question: str) -> dict | None:
        if not self.enabled or not self._available:
            return None
        try:
            query_embedding = self._get_embedding(question)
            self._ensure_dim(query_embedding)
            # Redis Stack 向量检索（KNN）必须使用 dialect=2
            q = "(*)=>[KNN 1 @embedding $vec AS score]"
            results = self.redis.ft(_INDEX_NAME).search(
                q,
                query_params={"vec": self._encode_vector(query_embedding)},
                dialect=2,
            )
            if results.docs:
                score = float(results.docs[0].score)
                similarity = 1 - score
                if similarity >= settings.similarity_threshold:
                    self._hits += 1
                    cache_hits_total.inc()
                    logger.info(f"语义缓存命中 (相似度: {similarity:.3f})")
                    doc = results.docs[0]
                    return {
                        "answer": doc["answer"],
                        "confidence": doc["confidence"],
                        "sources": json.loads(doc["sources"] or "[]"),
                    }
        except Exception as e:
            logger.warning(f"语义缓存查询失败: {e}")
        self._misses += 1
        cache_misses_total.inc()
        return None

    def store(self, question: str, result: dict):
        if not self.enabled or not self._available:
            return
        try:
            embedding = self._get_embedding(question)
            self._ensure_dim(embedding)
            key = f"qa:{hashlib.md5(question.encode()).hexdigest()}"
            doc = {
                "question": question,
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", ""),
                "sources": json.dumps(result.get("sources", []), ensure_ascii=False),
                "embedding": embedding,
            }
            self.redis.json().set(key, "$", doc)
            self.redis.expire(key, settings.cache_ttl_hours * 3600)
        except Exception as e:
            logger.warning(f"语义缓存写入失败: {e}")

    def clear(self):
        if not self.enabled:
            return
        for key in self.redis.scan_iter("qa:*"):
            self.redis.delete(key)
        self._hits = 0
        self._misses = 0
        logger.info("语义缓存已清空")

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "enabled": self.enabled,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total:.1%}" if total > 0 else "0.0%",
            "threshold": settings.similarity_threshold,
        }


_semantic_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache
