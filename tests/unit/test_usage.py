"""RAG 使用统计单元测试（FakeStore，不依赖真实 Redis）"""

from docs_seeker.domain.interfaces.usage import CallStats, UserCalls
from docs_seeker.domain.services.usage import UsageTracker


class FakeStore:
    """实现 UsageStore 业务操作的内存假存储"""

    def __init__(self):
        self.total = 0
        self.success = 0
        self.users: dict[str, list[int]] = {}  # uid -> [total, success]
        self.top: dict[str, int] = {}
        self.fail = False  # True 时所有操作抛异常（模拟 Redis 不可用）

    def _guard(self):
        if self.fail:
            raise ConnectionError("redis down")

    def record_call(self, uid, ok):
        self._guard()
        self.total += 1
        if ok:
            self.success += 1
        rec = self.users.setdefault(uid, [0, 0])
        rec[0] += 1
        if ok:
            rec[1] += 1

    def record_question(self, question):
        self._guard()
        self.top[question] = self.top.get(question, 0) + 1

    def top_questions(self, limit):
        self._guard()
        items = sorted(self.top.items(), key=lambda kv: kv[1], reverse=True)
        return [(q, c) for q, c in items[:limit]]

    def call_stats(self):
        self._guard()
        users = [UserCalls(user_id=uid, total=rec[0], success=rec[1]) for uid, rec in self.users.items()]
        return CallStats(total=self.total, success=self.success, users=users)


class FakeCache:
    def search(self, question):
        return None  # 永不命中


def _tracker(store: FakeStore | None = None) -> UsageTracker:
    return UsageTracker(store=store or FakeStore(), cache=FakeCache())


def test_record_only_tracks_rag_paths():
    store = FakeStore()
    t = _tracker(store)
    t.record("u1", "/v1/chat", 200)
    t.record("u1", "/v1/chat", 500)
    t.record("u1", "/v1/health", 200)  # 不统计
    assert store.total == 2
    assert store.success == 1
    assert "u1" in store.users


def test_record_anonymous_user():
    store = FakeStore()
    t = _tracker(store)
    t.record("", "/v1/chat", 200)
    assert store.users["anonymous"][0] == 1


def test_record_question_normalizes_and_counts():
    store = FakeStore()
    t = _tracker(store)
    t.record_question("  怎么  报销  ")
    t.record_question("怎么 报销")
    assert store.top["怎么 报销"] == 2


def test_record_question_filters_short():
    store = FakeStore()
    t = _tracker(store)
    t.record_question(" ")
    assert store.top == {}


def test_top_questions_sorted_with_cached_flag():
    store = FakeStore()
    store.top = {"问题A": 3, "问题B": 1}
    t = _tracker(store)
    items = t.top_questions(limit=10)
    assert [q["question"] for q in items] == ["问题A", "问题B"]
    assert all(q["cached"] is False for q in items)


def test_stats_aggregation():
    store = FakeStore()
    t = _tracker(store)
    t.record("u1", "/v1/chat", 200)
    t.record("u1", "/v1/chat", 500)
    t.record("u2", "/v1/chat", 200)
    stats = t.stats()
    assert stats["total_calls"] == 3
    assert stats["success_calls"] == 2
    assert stats["active_users"] == 2
    assert stats["users"][0]["user_id"] == "u1"
    assert stats["users"][0]["calls"] == 2


def test_redis_down_degrades_gracefully():
    store = FakeStore()
    store.fail = True
    t = _tracker(store)
    t.record("u1", "/v1/chat", 200)  # 不应抛异常
    assert t.stats()["total_calls"] == 0
    assert t.top_questions(10) == []
