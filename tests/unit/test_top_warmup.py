"""热门问题预热器单元测试（Fake 依赖，不依赖真实 Redis / LLM / Milvus）

覆盖：未命中缓存的预热与落库、全命中跳过、单题失败不阻断后续。
直接调 ``warmup_once()``，不起后台线程。
"""

from docs_seeker.llm.middleware.guards import GuardVerdict
from docs_seeker.models.chunk import Chunk
from docs_seeker.services.top_warmup import TopQuestionWarmup


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


def _warmup(usage, service=None) -> tuple[TopQuestionWarmup, FakeService]:
    service = service or FakeService()
    warmup = TopQuestionWarmup(
        service=service,  # type: ignore[arg-type]
        usage_tracker=usage,  # type: ignore[arg-type]
    )
    return warmup, service


def test_uncached_questions_are_warmed_and_stored():
    usage = FakeUsage(
        [
            {"question": "问题A", "count": 3, "cached": False},
            {"question": "问题B", "count": 2, "cached": False},
        ]
    )
    warmup, service = _warmup(usage)

    warmup.warmup_once()

    assert service.pipeline.ran == ["问题A", "问题B"]
    assert [q for q, _ in service.cache.stored] == ["问题A", "问题B"]
    assert service.cache.stored[0][1]["answer"] == "答案正文"
    assert service.cache.stored[0][1]["confidence"] == "high"


def test_already_cached_questions_are_skipped():
    usage = FakeUsage([{"question": "问题A", "count": 3, "cached": True}])
    warmup, service = _warmup(usage)

    warmup.warmup_once()

    assert service.pipeline.ran == []
    assert service.cache.stored == []


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
