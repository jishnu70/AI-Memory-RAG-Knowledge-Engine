# src/services/vector_services.py
from uuid import uuid4

import qdrant_client
from fastembed import TextEmbedding
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorService:
    """
    A service for handling vector operations.
    For now it will be minimum with only things i have learnt till now.
    """

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self._qdrant_client = qdrant_client.QdrantClient(path="./qdrant/qdrant.db")
        self._embedding_model = TextEmbedding()
        self.create_collection()

    def create_collection(self):
        """
        Create a collection in Qdrant if it doesn't exist.
        """
        sample_embedding = self.embed_text("test")[0]
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

    def embed_text(self, text: str) -> list[list[float]]:
        """
        Embed the given text using the embedding model.
        """
        return [arr.tolist() for arr in self._embedding_model.embed([text])]

    def store(self, text: str):
        """
        Store the given text in the collection after embedding it.
        """
        vector = self.embed_text(text)[0]  # Get the first (and only) embedding vector
        """
        There is only one because we are embedding a single text.
        If we were embedding multiple texts, we would have multiple vectors.
        When chunking there will be multiple vectors for a single text, but for now we are not doing that.
        """
        self._qdrant_client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(id=str(uuid4()), vector=vector, payload={"text": text})
            ],
        )

    def search(self, query: str):
        query_vector = self.embed_text(query)[0]
        search_result = self._qdrant_client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=5,
        )
        return search_result
