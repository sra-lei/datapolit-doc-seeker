"""日志初始化的容错契约（2026-09-17 线上事故回归锁）。

事故：容器以 ``read_only: true`` 运行，``setup_logging`` 在只读根文件系统里
``mkdir logs`` → ``OSError: [Errno 30] Read-only file system`` → 应用启动失败 →
无限重启（服务整段不可用）。

契约：**文件 sink 建不起来不得让应用启动失败** —— 降级为仅 stderr 并打 warning；
可写时仍要正常产出 JSON 日志。
"""

from __future__ import annotations

from loguru import logger

from docs_seeker.infra.logger import logging as app_logging


def test_setup_logging_survives_unavailable_file_sink(monkeypatch, capsys) -> None:
    """文件 sink 不可用（只读文件系统）时不得抛异常，且 stderr 仍能出日志"""

    def _boom(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(app_logging, "_JSONFileSink", _boom)

    app_logging.setup_logging("INFO")  # 不抛 = 不阻塞启动
    logger.warning("降级后仍可打日志")
    assert "降级后仍可打日志" in capsys.readouterr().err

    app_logging.setup_logging("INFO")  # 还原全局 logger 状态


def test_setup_logging_writes_json_file_when_writable(tmp_path) -> None:
    app_logging.setup_logging("INFO", log_dir=tmp_path)
    logger.info("写入文件")
    logger.complete()  # enqueue=True：等待落盘

    files = list(tmp_path.glob("docs-seeker-*.log"))
    assert files, "可写目录下应产出 JSON 日志文件"
    assert "写入文件" in files[0].read_text(encoding="utf-8")

    app_logging.setup_logging("INFO")  # 还原全局 logger 状态
