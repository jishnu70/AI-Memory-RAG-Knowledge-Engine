# src/services/embedding_services.py

from functools import lru_cache
from typing import Optional

from fastembed import TextEmbedding


class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None) -> None:
        model_name = model_name or "BAAI/bge-small-en-v1.5"
        self._embedding_model = TextEmbedding(model_name=model_name)

    @property
    def info(self) -> dict:
        """Returns the underlying embedding model."""
        return {
            "model_name": self._embedding_model.model_name,
            "embedding_size": self._embedding_model.embedding_size,
        }

    def embed(self, payload: list[str]) -> list[list[float]]:
        return [arr.tolist() for arr in self._embedding_model.embed(payload)]

    def get_sample_embedding(self) -> list[float]:
        """Returns a sample embedding vector for the model."""
        sample_text = "This is a sample text to generate an embedding."
        return self.embed([sample_text])[0]


@lru_cache(maxsize=1)
def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """Returns a singleton instance of the EmbeddingService."""
    return EmbeddingService(model_name=model_name)
