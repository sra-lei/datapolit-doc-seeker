"""
docs-seeker - Langfuse 链路追踪接入点（可选，未配置时自动降级为 no-op）

遵循 Langfuse 官方 Agent Skill（github.com/langfuse/skills）与官方文档最佳实践：
- 一个 chat 请求 = 一条 trace（名称 chat-response，动词开头、稳定、不含动态值）；
  多轮会话用 session_id 分组，用户维度用 user_id 归因；
- 未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 时客户端自动进入 no-op 模式，
  所有追踪调用零开销降级，不影响业务；
- 必须在加载 .env 之后再触发任何 langfuse 客户端初始化（SDK 直接读取
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL / LANGFUSE_HOST 环境变量）。
"""

import os

from dotenv import load_dotenv

# langfuse 客户端为惰性初始化（首次 get_client() 时才读取环境变量），
# 但按 skill 最佳实践仍先确保 .env 注入 os.environ，再引入 langfuse。
load_dotenv()

from langfuse import Langfuse, get_client  # noqa: E402  # 必须在 load_dotenv() 之后导入

# 稳定的观测命名（best practice：动词开头、低基数，便于过滤/仪表盘/评估器复用）
TRACE_NAME = "chat-response"
# 业务维度标签：tags 在观测创建时不可变，适合标记请求来源功能
FEATURE_TAG = "chat"


def tracing_enabled() -> bool:
    """Langfuse 是否启用：需要公钥 + 私钥，且未被 LANGFUSE_TRACING_ENABLED=false 显式关闭。"""
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_langfuse() -> Langfuse:
    """返回全局 langfuse 客户端（未配置时为 disabled 客户端，调用均为 no-op）。"""
    return get_client()


def shutdown_langfuse() -> None:
    """进程退出前冲刷并关闭 langfuse 客户端（长驻服务在 lifespan 收尾时调用）。"""
    if not tracing_enabled():
        return
    try:
        get_client().shutdown()
    except Exception:
        # 关闭失败不应影响进程退出
        pass
