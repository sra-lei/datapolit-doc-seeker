"""
docs-seeker - 热门问题预热器（P4）
对 TopN 热门问题中未命中语义缓存的，定期主动跑一遍 RAG 流程写入缓存，
使高频问题从"第一次提问"起即命中缓存，减少重复检索 + LLM 生成的 token 花费。

注意：预热直接走 pipeline + cache.store（不经 chat()），避免：
- 重复的问题计数（record_question）
- 重复的注入检测/脱敏链路
数据写入与 chat 路径保持一致的格式（CACHE_FIELDS + 护栏链路脱敏）。

依赖（ChatService / UsageTracker / DistributedLock）全部由组合根注入；
互斥失败与锁机制不可用的降级口径属于业务决策，留在本模块。
"""

import threading

from loguru import logger

from docs_seeker.config.settings import settings
from docs_seeker.infra.guards import ANSWER_CTX
from docs_seeker.interfaces.lock import DistributedLock
from docs_seeker.services.chat_service import CACHE_FIELDS, ChatService
from docs_seeker.services.usage import UsageTracker

_LOCK_KEY = "rag:warmup:lock"
# 锁 TTL：一轮预热最多持有的时间（远大于单轮耗时，避免锁提前过期导致并发预热）
_LOCK_TTL_SECONDS = 3600


class TopQuestionWarmup:
    """后台线程：定时对热门问题预热语义缓存"""

    def __init__(self, service: ChatService, usage_tracker: UsageTracker, lock: DistributedLock):
        self._service = service
        self._usage = usage_tracker
        self._lock = lock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not settings.top_warmup_enabled:
            logger.info("热门问题预热已禁用（TOP_WARMUP_ENABLED=false）")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="top-warmup", daemon=True)
        self._thread.start()
        logger.info(f"热门问题预热器已启动（每 {settings.top_warmup_interval_hours}h，Top{settings.top_warmup_size}）")

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
            top = self._usage.top_questions(limit=settings.top_warmup_size)
            todo = [q for q in top if not q["cached"]]
            if not todo:
                logger.info("[warmup] TopN 均已命中缓存，跳过")
                return

            logger.info(f"[warmup] 预热 {len(todo)}/{len(top)} 个热门问题")
            for item in todo:
                question = item["question"]
                logger.info(f"[warmup] 预热: {question[:40]}")
                try:
                    answer, confidence, chunks, _ = self._service.pipeline.run(question, top_k=10)
                    answer = self._service.guards.inspect(answer, ANSWER_CTX).text
                    source_dicts = [{k: v for k, v in chunk.to_dict().items() if k in CACHE_FIELDS} for chunk in chunks]
                    self._service.cache.store(
                        question,
                        {"answer": answer, "confidence": confidence, "sources": source_dicts},
                    )
                    logger.info(f"[warmup] 已写入缓存: {question[:40]}")
                except Exception as e:
                    logger.error(f"[warmup] 预热失败 {question[:40]}: {e}")
        finally:
            self._release_lock()

    def _acquire_lock(self) -> bool:
        """获取互斥锁；锁机制不可用时降级为「单实例直接执行」（可用性优先）"""
        try:
            return self._lock.acquire(_LOCK_KEY, ttl_seconds=_LOCK_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"[warmup] 获取分布式锁失败（按单实例继续）: {e}")
            return True

    def _release_lock(self) -> None:
        """释放锁；失败不影响正确性（TTL 会兜底过期）"""
        try:
            self._lock.release(_LOCK_KEY)
        except Exception:
            pass
