# src/services/vector_services.py

from datetime import datetime, timezone
from uuid import uuid4

import qdrant_client
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorService:
    """
    A service for handling vector operations.
    For now it will be minimum with only things i have learnt till now.
    """

    def __init__(self, collection_name: str, sample_embedding: list[float]):
        self._collection_name = collection_name
        self._qdrant_client = qdrant_client.QdrantClient(path="./qdrant/qdrant.db")
        self.create_collection(sample_embedding)

    def create_collection(self, sample_embedding: list[float]):
        """
        Create a collection in Qdrant if it doesn't exist.
        """
        if not self._qdrant_client.collection_exists(self._collection_name):
            self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=len(sample_embedding), distance=Distance.COSINE
                ),
            )

    def delete_collection(self):
        """
        Delete the collection from Qdrant.
        """
        self._qdrant_client.delete_collection(self._collection_name)

    def close(self, delete_collection: bool = False):
        """
        Close the Qdrant client connection.
        """
        if delete_collection:
            self.delete_collection()
        self._qdrant_client.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _create_points(
        self, embedded_list: list[list[float]], original_chunk: list[str], **kwargs
    ) -> list[PointStruct]:
        point_list: list[PointStruct] = []
        payload = {}
        payload["document_id"] = kwargs.get(
            "document_id",
        )
        payload["is_file_source"] = kwargs.get("is_file_source", False)
        payload["vector_created_at"] = kwargs.get("vector_created_at", self._now())
        if filename := kwargs.get("filename", None):
            payload["filename"] = filename

        for idx, (emb, ori) in enumerate(zip(embedded_list, original_chunk)):
            payload["original_chunk"] = ori
            payload["chunk_index"] = idx

            point_list.append(PointStruct(id=str(uuid4()), vector=emb, payload=payload))

        return point_list

    # def save
    def save(
        self, *, embedded_list: list[list[float]], original_chunk: list[str], **kwargs
    ):
        if not embedded_list:
            return
        points = self._create_points(embedded_list, original_chunk, **kwargs)
        self._qdrant_client.upsert(self._collection_name, points=points)

    # def retrieve
    def retrieve(
        self, *, query: list[float], top_k: int = 5, **kwargs
    ) -> QueryResponse:
        response = self._qdrant_client.query_points(
            collection_name=self._collection_name, query=query, limit=top_k
        )

        return response
