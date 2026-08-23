"""
docs-seeker - 配置管理
所有配置通过环境变量读取，禁止硬编码
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # DeepSeek (Chat)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # 阿里百炼 (Embedding)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 与 doc-kit 入库模型一致（text-embedding-v4 = 1024 维）；库已用 v4 重建
    embedding_model: str = "text-embedding-v4"

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    collection_name: str = "chartermate_docs"
    summary_collection_name: str = "chartermate_summaries"

    # Redis（语义缓存）
    redis_url: str = "redis://localhost:6379/0"
    semantic_cache_enabled: bool = True
    similarity_threshold: float = 0.92
    cache_ttl_hours: int = 24

    # 应用配置
    log_level: str = "INFO"


settings = Settings()
