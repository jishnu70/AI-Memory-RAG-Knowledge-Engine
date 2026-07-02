# src/schemas/embed_data.py

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class EmbedData:
    embedding: list[float]
    text: str
    metadata: dict[str, Any]

    def __post_init__(self):
        if (
            not isinstance(self.embedding, list)
            or not self.embedding
            or not all(isinstance(i, (float, int)) for i in self.embedding)
        ):
            raise ValueError("embedding must be a list of floats")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def to_dict(self) -> dict:
        return asdict(self)
