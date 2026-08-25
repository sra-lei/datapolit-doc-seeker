"""RAG 使用统计单元测试（mock Redis，不依赖真实 Redis）"""
from unittest.mock import patch

from docs_seeker.infrastructure.usage.tracker import UsageTracker


class FakeRedis:
    """支持 record/stats/top 所需 Redis 命令的假客户端"""

    def __init__(self):
        self.data = {}
        self.zsets = {}
        self.sets = {}

    # ---- pipeline（record 使用） ----
    def pipeline(self):
        return _Pipeline(self)

    # ---- 基础命令 ----
    def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]

    def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)
        return 1

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def get(self, key):
        return self.data.get(key)

    def zincrby(self, key, amount, member):
        z = self.zsets.setdefault(key, {})
        z[member] = z.get(member, 0) + amount
        return z[member]

    def zrevrange(self, key, start, end, withscores=False):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1], reverse=True)
        if withscores:
            return items[start : end + 1]
        return [m for m, _ in items[start : end + 1]]


class _Pipeline:
    def __init__(self, store: FakeRedis):
        self._store = store
        self._ops = []

    def incr(self, key):
        self._ops.append(("incr", key))
        return self

    def sadd(self, key, member):
        self._ops.append(("sadd", key, member))
        return self

    def execute(self):
        for op in self._ops:
            if op[0] == "incr":
                self._store.incr(op[1])
            else:
                self._store.sadd(op[1], op[2])


class FakeCache:
    def search(self, question):
        return None  # 永不命中


def _tracker(redis: FakeRedis) -> UsageTracker:
    p = patch("docs_seeker.infrastructure.usage.tracker.get_redis_client", return_value=redis)
    p.start()
    return UsageTracker()


def test_record_only_tracks_rag_paths():
    redis = FakeRedis()
    t = _tracker(redis)
    t.record("u1", "/v1/chat", 200)
    t.record("u1", "/v1/retrieve", 500)
    t.record("u1", "/v1/health", 200)  # 不统计
    assert redis.data["rag:usage:total"] == 2
    assert redis.data["rag:usage:success"] == 1
    assert "u1" in redis.sets["rag:usage:users"]


def test_record_anonymous_user():
    redis = FakeRedis()
    t = _tracker(redis)
    t.record("", "/v1/chat", 200)
    assert redis.data["rag:usage:user:anonymous:total"] == 1


def test_record_question_normalizes_and_counts():
    redis = FakeRedis()
    t = _tracker(redis)
    t.record_question("  怎么  报销  ")
    t.record_question("怎么 报销")
    z = redis.zsets["rag:usage:top"]
    assert z["怎么 报销"] == 2


def test_record_question_filters_short():
    redis = FakeRedis()
    t = _tracker(redis)
    t.record_question(" ")
    assert "rag:usage:top" not in redis.zsets


def test_top_questions_sorted_with_cached_flag():
    redis = FakeRedis()
    t = _tracker(redis)
    redis.zincrby("rag:usage:top", 3, "问题A")
    redis.zincrby("rag:usage:top", 1, "问题B")
    with patch("docs_seeker.infrastructure.cache.semantic_cache.get_semantic_cache", return_value=FakeCache()):
        items = t.top_questions(limit=10)
    assert [q["question"] for q in items] == ["问题A", "问题B"]
    assert all(q["cached"] is False for q in items)


def test_stats_aggregation():
    redis = FakeRedis()
    t = _tracker(redis)
    t.record("u1", "/v1/chat", 200)
    t.record("u1", "/v1/chat", 500)
    t.record("u2", "/v1/retrieve", 200)
    stats = t.stats()
    assert stats["total_calls"] == 3
    assert stats["success_calls"] == 2
    assert stats["active_users"] == 2
    assert stats["users"][0]["user_id"] == "u1"
    assert stats["users"][0]["calls"] == 2


def test_redis_down_degrades_gracefully():
    def _boom():
        raise ConnectionError("redis down")

    with patch("docs_seeker.infrastructure.usage.tracker.get_redis_client", side_effect=_boom):
        t = UsageTracker()
        t.record("u1", "/v1/chat", 200)  # 不应抛异常
        assert t.stats()["total_calls"] == 0
        assert t.top_questions(10) == []
