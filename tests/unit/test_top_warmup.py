"""热门问题预热器单元测试（Fake 依赖，不依赖真实 Redis / LLM / Milvus）

覆盖：互斥跳过、未命中缓存的预热与落库、全命中跳过、锁机制不可用时的降级、
异常路径下的锁释放。直接调 ``warmup_once()``，不起后台线程。
"""

from docs_seeker.infra.llm.middleware.guards import GuardVerdict
from docs_seeker.models.chunk import Chunk
from docs_seeker.services.top_warmup import TopQuestionWarmup


class FakeLock:
    """记录 acquire/release 的假锁；held=True 模拟被其他实例持有，fail=True 模拟机制不可用"""

    def __init__(self, held: bool = False, fail: bool = False):
        self.held = held
        self.fail = fail
        self.acquired: list[tuple[str, int]] = []
        self.released: list[str] = []

    def acquire(self, key, ttl_seconds):
        if self.fail:
            raise ConnectionError("redis down")
        self.acquired.append((key, ttl_seconds))
        return not self.held

    def release(self, key):
        self.released.append(key)


class FakeUsage:
    def __init__(self, items):
        self.items = items
        self.limits: list[int] = []

    def top_questions(self, limit=10):
        self.limits.append(limit)
        return self.items


class FakeCache:
    def __init__(self):
        self.stored: list[tuple[str, dict]] = []

    def store(self, question, result):
        self.stored.append((question, result))


class FakeGuards:
    def inspect(self, text, ctx):
        return GuardVerdict(allowed=True, text=text)


class FakePipeline:
    def __init__(self, chunks):
        self.chunks = chunks
        self.ran: list[str] = []

    def run(self, question, top_k=10, **kwargs):
        self.ran.append(question)
        return "答案正文", "high", self.chunks, ["子问题"]


class FakeService:
    def __init__(self, chunks=None):
        self.pipeline = FakePipeline(chunks or [Chunk(id="c1", text="正文", source="doc.md", score=0.5)])
        self.guards = FakeGuards()
        self.cache = FakeCache()


def _warmup(usage, service=None, lock=None) -> tuple[TopQuestionWarmup, FakeService]:
    service = service or FakeService()
    warmup = TopQuestionWarmup(
        service=service,  # type: ignore[arg-type]
        usage_tracker=usage,  # type: ignore[arg-type]
        lock=lock or FakeLock(),
    )
    return warmup, service


def test_lock_held_by_another_instance_skips():
    usage = FakeUsage([{"question": "问题A", "count": 3, "cached": False}])
    lock = FakeLock(held=True)
    warmup, service = _warmup(usage, lock=lock)

    warmup.warmup_once()

    assert service.pipeline.ran == []  # 未预热
    assert service.cache.stored == []
    assert lock.released == []  # 没拿到锁就不释放


def test_uncached_questions_are_warmed_and_stored():
    usage = FakeUsage(
        [
            {"question": "问题A", "count": 3, "cached": False},
            {"question": "问题B", "count": 2, "cached": False},
        ]
    )
    lock = FakeLock()
    warmup, service = _warmup(usage, lock=lock)

    warmup.warmup_once()

    assert service.pipeline.ran == ["问题A", "问题B"]
    assert [q for q, _ in service.cache.stored] == ["问题A", "问题B"]
    assert service.cache.stored[0][1]["answer"] == "答案正文"
    assert service.cache.stored[0][1]["confidence"] == "high"
    assert lock.released == ["rag:warmup:lock"]


def test_already_cached_questions_are_skipped():
    usage = FakeUsage([{"question": "问题A", "count": 3, "cached": True}])
    warmup, service = _warmup(usage)

    warmup.warmup_once()

    assert service.pipeline.ran == []
    assert service.cache.stored == []


def test_lock_mechanism_down_degrades_to_single_instance():
    """锁机制不可用（Redis 掉线）→ 按单实例继续执行，不阻塞预热"""
    usage = FakeUsage([{"question": "问题A", "count": 3, "cached": False}])
    warmup, service = _warmup(usage, lock=FakeLock(fail=True))

    warmup.warmup_once()

    assert service.pipeline.ran == ["问题A"]  # 仍然预热
    assert service.cache.stored  # 仍然落库


def test_release_failure_does_not_raise():
    usage = FakeUsage([{"question": "问题A", "count": 3, "cached": False}])

    class _BrokenRelease(FakeLock):
        def release(self, key):
            raise ConnectionError("redis down")

    warmup, service = _warmup(usage, lock=_BrokenRelease())

    warmup.warmup_once()  # 不应抛异常

    assert service.cache.stored


def test_one_failed_question_does_not_block_the_rest():
    usage = FakeUsage(
        [
            {"question": "会失败的问题", "count": 3, "cached": False},
            {"question": "问题B", "count": 2, "cached": False},
        ]
    )
    service = FakeService()
    original_run = service.pipeline.run

    def _flaky_run(question, **kwargs):
        if question == "会失败的问题":
            raise RuntimeError("生成失败")
        return original_run(question, **kwargs)

    service.pipeline.run = _flaky_run  # type: ignore[method-assign]
    warmup, _ = _warmup(usage, service=service)

    warmup.warmup_once()

    assert [q for q, _ in service.cache.stored] == ["问题B"]  # 后续问题继续预热


def test_stop_sets_event():
    usage = FakeUsage([])
    warmup, _ = _warmup(usage)
    warmup.stop()
    assert warmup._stop.is_set()
