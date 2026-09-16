"""docs-seeker - 结构化日志（loguru + JSON 按日轮换）"""

import json
import sys
import traceback as tb
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


class _JSONFileSink:
    """JSON 文件 sink：按日创建文件、自动清理过期日志"""

    def __init__(self, log_dir: Path, retention_days: int = 30):
        self._dir = log_dir
        self._dir.mkdir(exist_ok=True)
        self._retention = retention_days
        self._file = None
        self._date: str | None = None

    def _rotate(self, record_time) -> None:
        today = record_time.strftime("%Y-%m-%d")
        if today == self._date:
            return
        if self._file:
            self._file.close()
        self._date = today
        self._file = open(self._dir / f"docs-seeker-{today}.log", "a", encoding="utf-8")
        self._purge_old()

    def _purge_old(self) -> None:
        cutoff = datetime.now() - timedelta(days=self._retention)
        for f in self._dir.glob("docs-seeker-*.log"):
            try:
                file_date = datetime.strptime(f.stem, "docs-seeker-%Y-%m-%d")
            except ValueError:
                continue
            if file_date < cutoff:
                f.unlink()

    def __call__(self, message) -> None:
        record = message.record
        self._rotate(record["time"])
        entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "line": record["line"],
            "message": record["message"],
        }
        if record["extra"]:
            entry["extra"] = record["extra"]
        if record["exception"]:
            exc = record["exception"]
            entry["exception"] = "".join(tb.format_exception(type(exc.value), exc.value, exc.traceback))
        self._file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._file.flush()


def setup_logging(level: str = "INFO") -> None:
    """配置全局 logger：
    - stderr: 彩色格式（开发时实时查看）
    - logs/docs-seeker-YYYY-MM-DD.log: JSON 格式按日轮换
    """
    logger.remove()

    # stderr — 彩色输出
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        backtrace=True,
        diagnose=False,
    )

    # logs/ — JSON 按日轮换
    logger.add(
        _JSONFileSink(Path("logs")),
        level=level,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
