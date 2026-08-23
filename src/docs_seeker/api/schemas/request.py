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


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=30)
    use_summary: bool = True
