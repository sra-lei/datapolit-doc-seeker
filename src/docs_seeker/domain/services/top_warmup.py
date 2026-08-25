"""
docs-seeker - 热门问题预热器（P4）
对 TopN 热门问题中未命中语义缓存的，定期主动跑一遍 RAG 流程写入缓存，
使高频问题从"第一次提问"起即命中缓存，减少重复检索 + LLM 生成的 token 花费。

注意：预热直接走 pipeline + cache.store（不经 chat()），避免：
- 重复的问题计数（record_question）
- 重复的注入检测/脱敏链路
数据写入与 chat 路径保持一致的格式（CACHE_FIELDS + sanitize_output）。
"""
import threading

from loguru import logger

from docs_seeker.core.config import settings
from docs_seeker.core.security import sanitize_output
from docs_seeker.domain.services.chat_service import CACHE_FIELDS, ChatService
from docs_seeker.infrastructure.cache.redis_client import get_redis_client
from docs_seeker.infrastructure.usage import get_usage_tracker

_LOCK_KEY = "rag:warmup:lock"


class TopQuestionWarmup:
    """后台线程：定时对热门问题预热语义缓存"""

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._service: ChatService | None = None

    def _get_service(self) -> ChatService:
        if self._service is None:
            self._service = ChatService()
        return self._service

    def start(self, service: ChatService | None = None) -> None:
        if not settings.top_warmup_enabled:
            logger.info("热门问题预热已禁用（TOP_WARMUP_ENABLED=false）")
            return
        if service is not None:
            # 注入与 API 路径共享的 ChatService（共用同一份 retriever / BM25 索引）
            self._service = service
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="top-warmup", daemon=True)
        self._thread.start()
        logger.info(
            f"热门问题预热器已启动（每 {settings.top_warmup_interval_hours}h，Top{settings.top_warmup_size}）"
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(settings.top_warmup_interval_hours * 3600):
            try:
                self.warmup_once()
            except Exception as e:
                logger.error(f"[warmup] 预热执行失败: {e}")

    def warmup_once(self) -> None:
        """执行一轮预热：TopN 中未命中缓存的问题 → pipeline + cache.store"""
        if not self._acquire_lock():
            logger.info("[warmup] 另一实例正在预热，跳过")
            return
        try:
            top = get_usage_tracker().top_questions(limit=settings.top_warmup_size)
            todo = [q for q in top if not q["cached"]]
            if not todo:
                logger.info("[warmup] TopN 均已命中缓存，跳过")
                return

            logger.info(f"[warmup] 预热 {len(todo)}/{len(top)} 个热门问题")
            service = self._get_service()
            for item in todo:
                question = item["question"]
                logger.info(f"[warmup] 预热: {question[:40]}")
                try:
                    answer, confidence, chunks, _ = service.pipeline.run(question, top_k=10)
                    answer = sanitize_output(answer)
                    source_dicts = [
                        {k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS}
                        for chunk in chunks
                    ]
                    service.cache.store(
                        question,
                        {"answer": answer, "confidence": confidence, "sources": source_dicts},
                    )
                    logger.info(f"[warmup] 已写入缓存: {question[:40]}")
                except Exception as e:
                    logger.error(f"[warmup] 预热失败 {question[:40]}: {e}")
        finally:
            self._release_lock()

    def _acquire_lock(self) -> bool:
        try:
            redis = get_redis_client()
            return bool(redis.set(_LOCK_KEY, "1", nx=True, ex=3600))
        except Exception:
            # Redis 不可用：单实例直接执行
            return True

    def _release_lock(self) -> None:
        try:
            get_redis_client().delete(_LOCK_KEY)
        except Exception:
            pass


_top_warmup: TopQuestionWarmup | None = None


def get_top_warmup() -> TopQuestionWarmup:
    global _top_warmup
    if _top_warmup is None:
        _top_warmup = TopQuestionWarmup()
    return _top_warmup
