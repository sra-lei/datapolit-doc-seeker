"""docs-seeker - 请求/响应模型"""
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    conversation_history: list[ChatMessage] | None = None
    top_k: int = Field(10, ge=1, le=30)
    use_cache: bool = True


class SourceDoc(BaseModel):
    id: str = ""
    text: str = ""
    source: str = ""
    chapter: str = ""
    chapter_title: str = ""
    section: str = ""
    section_title: str = ""
    score: float = 0.0
    sources: list[str] = []


class ChatResponse(BaseModel):
    answer: str
    confidence: str = "medium"
    sources: list[SourceDoc] = []
    cached: bool = False
    query_decomposed: list[str] | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=30)
    use_summary: bool = True


class RetrieveResponse(BaseModel):
    docs: list[SourceDoc] = []
    total: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    milvus_connected: bool = False
    redis_connected: bool = False
