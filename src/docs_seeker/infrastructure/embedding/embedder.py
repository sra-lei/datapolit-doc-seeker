"""
docs-seeker - 向量化模块（只读）
仅提供查询向量化，不含索引构建（入库由 doc-kit 负责）
"""

from loguru import logger
from openai import OpenAI

from docs_seeker.core.config import settings
from docs_seeker.domain.interfaces.embedder import EmbeddingProvider


class Embedder(EmbeddingProvider):
    """查询向量化服务（调阿里百炼）"""

    _instance: "Embedder | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.embedding_model = settings.embedding_model
        self.embedding_client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        self._initialized = True
        logger.info(f"Embedder 初始化完成 | model={self.embedding_model}")

    def get_embedding(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ") for t in texts[i : i + batch_size]]
            response = self.embedding_client.embeddings.create(model=self.embedding_model, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    @classmethod
    def reset(cls):
        cls._instance = None


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
