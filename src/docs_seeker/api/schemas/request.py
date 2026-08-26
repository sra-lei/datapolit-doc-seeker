"""请求模型"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    conversation_history: list[ChatMessage] | None = None
    top_k: int = Field(10, ge=1, le=30)
    use_cache: bool = True
    # True（默认）= SSE 流式输出；False = 一次性 JSON 返回（兼容旧客户端/评估服务）
    stream: bool = True
    # Langfuse 追踪属性（可选）：多轮会话分组（Sessions 视图）与用户归因
    session_id: str | None = Field(None, max_length=200)
    user_id: str | None = Field(None, max_length=200)
