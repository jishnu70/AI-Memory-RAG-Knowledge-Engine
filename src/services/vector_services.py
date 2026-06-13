# src/services/vector_services.py
from typing import Union
from uuid import uuid4

import qdrant_client
from fastapi import UploadFile
from fastembed import TextEmbedding
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.services.chunking_technique import TECHNIQUES, ChunkingTechnique


class VectorService:
    """
    A service for handling vector operations.
    For now it will be minimum with only things i have learnt till now.
    """

    def __init__(self, collection_name: str, chunk_technique: ChunkingTechnique):
        self._collection_name = collection_name
        self._qdrant_client = qdrant_client.QdrantClient(path="./qdrant/qdrant.db")
        self._embedding_model = TextEmbedding()
        self._chunk_technique = chunk_technique
        self.create_collection()

    def create_collection(self):
        """
        Create a collection in Qdrant if it doesn't exist.
        """
        sample_embedding = self.embed_text("test")
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

    def embed_text(self, text: str) -> list[float]:
        """
        Embed the given text using the embedding model.
        iter and next is used to get the first (and only) embedding
        from the list returned by the embed method.

        I used __next__ as it is easy to read and understand,
            but we can also use next() function like this:
                return next(iter(self._embedding_model.embed([text]))).tolist()
        """
        return iter(self._embedding_model.embed([text])).__next__().tolist()

    def chunk(
        self,
        text: Union[str, bytes],
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        technique: TECHNIQUES = "recursive",
        **kwargs,
    ) -> list[str]:
        """
        Chunk the given text into smaller pieces of the specified size.
        This is a simple implementation and can be improved to handle edge cases.
        """
        if isinstance(text, str):
            return self._chunk_technique.chunk(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                technique=technique,
            )
        return self._chunk_technique.chunk(
            text=text.decode("utf-8"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            technique=technique,
            **kwargs,
        )

    async def store(self, data: Union[str, UploadFile]):
        """
        Store the given text or document in the collection after embedding it.
         - If the input is a string, it will be chunked and each chunk will be embedded and stored as a point in the collection.
         - If the input is an UploadFile, it will be read, chunked, and each chunk will be embedded and stored as a point in the collection.
         - The original text of each chunk will be stored as payload for retrieval during search.
         - Each point will have a unique ID generated using uuid4.
         - The embedding of each chunk will be stored as the vector representation in the collection.
         - The method is asynchronous to handle file reading operations efficiently.
         - The chunking technique can be specified to control how the text is split into chunks.
         - The method ensures that the collection is created before storing any points.

        Args:
            data (Union[str, UploadFile]): The text or document to be stored.

        Returns:
            None
        """
        chunks: list[str] = []
        if isinstance(data, UploadFile):
            byte = await data.read()
            chunks.extend(
                self.chunk(
                    text=byte,
                    chunk_size=512,
                    chunk_overlap=20,
                    technique="markdown",
                    filename=data.filename,
                )
            )
        elif isinstance(data, str):
            chunks.extend(
                self.chunk(
                    text=data,
                    chunk_size=512,
                    chunk_overlap=20,
                    technique="recursive",
                )
            )

        """
        Store the given text in the collection after embedding it.

        There is only one because we are embedding a single text.
        If we were embedding multiple texts, we would have multiple vectors.
        When chunking there will be multiple vectors for a single text, but for now we are not doing that.
        """
        points: list[PointStruct] = []
        for idx, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid4()),  # generate a unique ID for each point
                    vector=self.embed_text(
                        chunk
                    ),  # embed the chunk to get the vector representation
                    payload={
                        "chunk": chunk,
                        "chunk_index": idx,
                    },  # store the original text as payload for retrieval
                )
            )
        self._qdrant_client.upsert(collection_name=self._collection_name, points=points)

    def search(self, query: str) -> QueryResponse:
        """
        Search for the given query in the collection and return the nearest points.

        Args:
            query (str): The query text to search for.

        Returns:
            QueryResponse: The search results containing the nearest points to the query.
        """
        query_vector = self.embed_text(query)
        search_result = self._qdrant_client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=5,
        )
        return search_result
