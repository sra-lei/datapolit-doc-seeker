"""
docs-seeker - 语义缓存
基于 Redis Stack 向量搜索，按问题相似度匹配缓存
"""
import time
import json
import hashlib
import redis
from openai import OpenAI
from loguru import logger

from docs_seeker.config import settings
from docs_seeker.infra.embedder import get_embedder


class SemanticCache:
    """基于 Redis Stack 向量搜索的语义缓存"""

    def __init__(self):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._hits = 0
        self._misses = 0
        self._available = False

        try:
            self.redis.ft("qa_cache_idx").info()
            self._available = True
        except Exception:
            try:
                self._create_index()
                self._available = True
            except Exception as e:
                logger.warning(f"语义缓存初始化失败，将降级为无缓存模式: {e}")
                self._available = False

    def _create_index(self):
        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        schema = (
            TextField("$.question", no_stem=True, as_name="question"),
            TextField("$.answer", no_stem=True, as_name="answer"),
            TextField("$.confidence", no_stem=True, as_name="confidence"),
            VectorField(
                "$.embedding", "FLAT",
                {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"},
                as_name="embedding",
            ),
        )
        definition = IndexDefinition(prefix=["qa:"], index_type=IndexType.JSON)
        self.redis.ft("qa_cache_idx").create_index(schema, definition=definition)
        logger.info("Redis 语义缓存索引已创建")

    def _get_embedding(self, text: str) -> list:
        return get_embedder().get_embedding(text)

    def search(self, question: str) -> dict | None:
        if not self._available:
            return None
        try:
            query_embedding = self._get_embedding(question)
            q = "(*)=>[KNN 1 @embedding $vec AS score]"
            results = self.redis.ft("qa_cache_idx").search(
                q, query_params={"vec": query_embedding.tobytes() if hasattr(query_embedding, "tobytes") else bytes(query_embedding)}
            )
            if results.docs:
                score = float(results.docs[0].score)
                similarity = 1 - score
                if similarity >= settings.similarity_threshold:
                    self._hits += 1
                    logger.info(f"语义缓存命中 (相似度: {similarity:.3f})")
                    return {
                        "answer": results.docs[0].answer,
                        "confidence": results.docs[0].confidence,
                        "sources": json.loads(results.docs[0].get("sources", "[]")),
                    }
        except Exception as e:
            logger.warning(f"语义缓存查询失败: {e}")
        self._misses += 1
        return None

    def store(self, question: str, result: dict):
        if not self._available:
            return
        try:
            embedding = self._get_embedding(question)
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
        for key in self.redis.scan_iter("qa:*"):
            self.redis.delete(key)
        self._hits = 0
        self._misses = 0
        logger.info("语义缓存已清空")

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
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
