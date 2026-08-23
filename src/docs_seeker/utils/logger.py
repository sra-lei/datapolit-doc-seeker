"""docs-seeker - 结构化日志（loguru）"""
import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """配置全局 logger：移除默认 sink，输出结构化格式"""
    logger.remove()
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
