"""语义缓存单元测试（mock Redis 与向量化，不依赖真实外部服务）

覆盖：向量编码格式、禁用开关、KNN dialect=2 与字节向量参数（Query 对象传递）、
命中路径（redis-py 8.x 的 doc.json 打包结构）、维度漂移自动重建索引。
"""
from unittest.mock import patch

from docs_seeker.core.config import settings
from docs_seeker.infrastructure.cache.semantic_cache import SemanticCache


class FakeDoc:
    """模拟 redis-py 8.x 的 Document：JSON 索引的搜索结果字段在 .json 属性"""

    def __init__(self, **kwargs):
        self.json = kwargs["json"]
        self.score = kwargs.get("score", 0.0)


class FakeSearchResult:
    def __init__(self, docs=None):
        self.docs = docs or []


class FakeIndex:
    """模拟 Redis Search 索引对象"""

    def __init__(self, existing=False, dim=None, search_docs=None):
        self.existing = existing
        self.dim = dim
        self.search_docs = search_docs
        self.created_schema = None
        self.dropped = False
        self.last_search_kwargs = None

    def info(self):
        if not self.existing:
            raise Exception("no index")
        # redis-py 8.x：attributes 为扁平列表
        return {
            "attributes": [
                [
                    "identifier", "$.embedding", "attribute", "embedding", "type", "VECTOR",
                    "algorithm", "FLAT", "data_type", "FLOAT32", "dim", self.dim, "distance_metric", "COSINE",
                ]
            ]
        }

    def create_index(self, schema, definition=None):
        self.created_schema = schema

    def dropindex(self):
        self.dropped = True

    def search(self, query, **kwargs):
        self.last_query = query
        self.last_search_kwargs = kwargs
        return FakeSearchResult(self.search_docs)


class FakeRedis:
    def __init__(self, index=None):
        self.index = index or FakeIndex()
        self.stored = {}

    def ft(self, name):
        return self.index

    def json(self):
        class _Json:
            def __init__(self, outer):
                self._outer = outer

            def set(self, key, path, doc):
                self._outer.stored[key] = doc

        return _Json(self)

    def expire(self, key, seconds):
        pass


class FakeEmbedder:
    DIM = 8

    def get_embedding(self, text):
        return [0.1] * self.DIM


def _patch(cache_enabled=True, index=None):
    fake_redis = FakeRedis(index=index)
    p_redis = patch("docs_seeker.infrastructure.cache.semantic_cache.get_redis_client", return_value=fake_redis)
    p_embedder = patch("docs_seeker.infrastructure.cache.semantic_cache.get_embedder", return_value=FakeEmbedder())
    p_enabled = patch.object(settings, "semantic_cache_enabled", cache_enabled)
    return fake_redis, p_redis, p_embedder, p_enabled


def test_encode_vector_is_float32_bytes():
    vec = [1.0, 2.0, 3.0, 4.0]
    encoded = SemanticCache._encode_vector(vec)
    assert isinstance(encoded, bytes)
    assert len(encoded) == 4 * len(vec)


def test_disabled_cache_never_queries_redis():
    fake_redis, p_redis, p_embedder, p_enabled = _patch(cache_enabled=False)
    with p_redis, p_embedder, p_enabled:
        cache = SemanticCache()
        assert cache.enabled is False
        assert cache._available is False
        assert cache.search("任何问题") is None
        cache.store("任何问题", {"answer": "x", "confidence": "high"})  # 不应抛异常
        assert fake_redis.index.created_schema is None  # 未尝试建索引


def test_search_uses_dialect2_and_bytes_vector():
    fake_redis, p_redis, p_embedder, p_enabled = _patch(index=FakeIndex(existing=False))
    with p_redis, p_embedder, p_enabled:
        cache = SemanticCache()
        assert cache._available is True
        assert cache._dim == FakeEmbedder.DIM
        assert cache.search("测试问题") is None  # 无结果 → miss
        kwargs = fake_redis.index.last_search_kwargs
        # redis-py 8.x：dialect 通过 Query 对象传递，不再作为 search() 关键字参数
        assert fake_redis.index.last_query._dialect == 2
        vec = kwargs["query_params"]["vec"]
        assert isinstance(vec, bytes)
        assert len(vec) == 4 * FakeEmbedder.DIM


def test_search_hit_returns_cached_answer():
    # redis-py 8.x：JSON 索引搜索结果整体打包在 doc.json（字符串）
    hit_doc = FakeDoc(
        json='{"answer": "缓存答案", "confidence": "high", "sources": "[{\\"id\\": \\"1\\", \\"text\\": \\"t\\"}]"}',
        score=0.05,  # similarity = 0.95 >= 0.92（阈值）
    )
    index = FakeIndex(existing=True, dim=FakeEmbedder.DIM, search_docs=[hit_doc])
    fake_redis, p_redis, p_embedder, p_enabled = _patch(index=index)
    with p_redis, p_embedder, p_enabled:
        cache = SemanticCache()
        result = cache.search("重复问题")
        assert result is not None
        assert result["answer"] == "缓存答案"
        assert result["confidence"] == "high"
        assert result["sources"] == [{"id": "1", "text": "t"}]
        assert cache.stats["hits"] == 1


def test_read_index_dim_supports_flat_list_attributes():
    """redis-py 8.x：ft().info() 的 attributes 是扁平列表，需能解析出 dim"""
    flat = [
        ["identifier", "$.question", "attribute", "question", "type", "TEXT", "WEIGHT", 1.0, "NOSTEM"],
        [
            "identifier", "$.embedding", "attribute", "embedding", "type", "VECTOR",
            "algorithm", "FLAT", "data_type", "FLOAT32", "dim", 1024, "distance_metric", "COSINE",
        ],
    ]

    class _InfoIndex:
        def info(self):
            return {"attributes": flat}

    class _InfoRedis:
        def ft(self, name):
            return _InfoIndex()

    with patch("docs_seeker.infrastructure.cache.semantic_cache.get_redis_client", return_value=_InfoRedis()):
        cache = SemanticCache()
        assert cache._dim == 1024
        assert cache._available is True


def test_dim_drift_rebuilds_index():
    index = FakeIndex(existing=True, dim=16)  # 与真实维度 8 不一致 → 触发重建
    fake_redis, p_redis, p_embedder, p_enabled = _patch(index=index)
    with p_redis, p_embedder, p_enabled:
        cache = SemanticCache()
        assert cache._dim == 16
        cache.search("触发维度漂移")
        assert index.dropped is True
        assert cache._dim == FakeEmbedder.DIM
        assert index.last_query._dialect == 2
