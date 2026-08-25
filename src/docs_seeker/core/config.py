"""
docs-seeker - 核心配置（pydantic-settings 环境变量 + yaml 资源加载）
所有配置通过环境变量读取，禁止硬编码；prompts/retrieval 从同目录 yaml 加载。
"""

from pathlib import Path

import yaml
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
    # 运行环境：development / production（production 下禁用 /docs、/redoc、/openapi.json）
    environment: str = "development"

    # 热门问题 Top10（P4）：记录 + 预热语义缓存（ChatWidget 欢迎语用）
    top_warmup_enabled: bool = True
    top_warmup_interval_hours: int = 6
    top_warmup_size: int = 10

    # BM25 索引
    # 新鲜度检查间隔（秒）：每隔该时长用 Milvus row count 探测文档数是否变化，
    # 变化才重建索引；0 表示关闭自动刷新（只靠启动预热 + 手动 refresh()）
    bm25_refresh_seconds: int = 300
    # 建索引单次拉取文档上限（与 MilvusStore.get_all_documents 的 limit 对齐）
    bm25_max_docs: int = 10000


settings = Settings()

_CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(name: str) -> dict:
    """读取 core 目录下的 yaml 配置，缺失/解析失败时返回空 dict"""
    path = _CONFIG_DIR / name
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Prompt 模板（generator/query_decomposer 等使用）
prompts = _load_yaml("prompts.yaml")

# 检索策略配置（RRF 权重/k、单路召回参数等）
retrieval_config = _load_yaml("retrieval.yaml")

__all__ = ["Settings", "settings", "prompts", "retrieval_config"]
