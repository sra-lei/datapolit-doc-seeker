"""RAG 使用统计单元测试（FakeStore，不依赖真实 Redis）"""

from docs_seeker.domain.services.usage import UsageTracker


class FakeStore:
    """实现 UsageStore 原语的内存假存储"""

    def __init__(self):
        self.total = 0
        self.success = 0
        self.users: dict[str, list[int]] = {}  # uid -> [total, success]
        self.top: dict[str, int] = {}
        self.fail = False  # True 时所有操作抛异常（模拟 Redis 不可用）

    def _guard(self):
        if self.fail:
            raise ConnectionError("redis down")

    def incr_total(self):
        self._guard()
        self.total += 1

    def incr_success(self):
        self._guard()
        self.success += 1

    def incr_user_total(self, uid):
        self._guard()
        self.users.setdefault(uid, [0, 0])[0] += 1

    def incr_user_success(self, uid):
        self._guard()
        self.users.setdefault(uid, [0, 0])[1] += 1

    def add_user(self, uid):
        self._guard()
        self.users.setdefault(uid, [0, 0])

    def get_total(self):
        self._guard()
        return self.total

    def get_success(self):
        self._guard()
        return self.success

    def get_users(self):
        self._guard()
        return set(self.users)

    def get_user_total(self, uid):
        self._guard()
        return self.users.get(uid, [0, 0])[0]

    def get_user_success(self, uid):
        self._guard()
        return self.users.get(uid, [0, 0])[1]

    def incr_top(self, question):
        self._guard()
        self.top[question] = self.top.get(question, 0) + 1

    def top_questions(self, limit):
        self._guard()
        items = sorted(self.top.items(), key=lambda kv: kv[1], reverse=True)
        return [(q, float(c)) for q, c in items[:limit]]


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
