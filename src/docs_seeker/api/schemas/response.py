"""响应模型"""
from pydantic import BaseModel


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


class RetrieveResponse(BaseModel):
    docs: list[SourceDoc] = []
    total: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    milvus_connected: bool = False
    redis_connected: bool = False


class CacheStats(BaseModel):
    enabled: bool = True
    hits: int = 0
    misses: int = 0
    hit_rate: str = "0.0%"
    threshold: float = 0.92


class LLMStats(BaseModel):
    total_calls: int = 0
    success_calls: int = 0
    fallback_calls: int = 0
    circuit_state: str = "closed"
    circuit_failures: int = 0


class StatsResponse(BaseModel):
    cache: CacheStats
    llm: LLMStats


class MilvusIndexInfo(BaseModel):
    index_name: str = ""
    field_name: str = ""
    index_type: str = ""
    metric_type: str = ""


class MilvusCollectionStats(BaseModel):
    name: str = ""
    exists: bool = False
    count: int = -1
    dim: int | None = None
    index: MilvusIndexInfo | None = None


class MilvusStatsResponse(BaseModel):
    connected: bool = False
    server_version: str = ""
    collections: dict[str, MilvusCollectionStats] = {}
