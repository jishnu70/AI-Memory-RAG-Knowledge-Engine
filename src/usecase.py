# src/usecase.py

import logging
from typing import Optional
from uuid import uuid4

from docling_core.types.doc.document import DoclingDocument
from fastapi import UploadFile
from typing_extensions import deprecated

from src.schemas.embed_data import EmbedData
from src.services.chunking_technique import TECHNIQUES, ChunkingTechnique
from src.services.document_loader import DocumentLoader
from src.services.embedding_service import EmbeddingService
from src.services.file_service import FileService
from src.services.llm_provider import LLMProvider
from src.services.vector_services import VectorService

logger = logging.getLogger(__name__)


class UseCase:
    def __init__(
        self,
        chunking_strategy: ChunkingTechnique,
        file_service: FileService,
        document_loader: DocumentLoader,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        llm_provider: LLMProvider,
    ):
        self._chunking_strategy = chunking_strategy
        self._file_service = file_service
        self._document_loader = document_loader
        self._embedding_service = embedding_service
        self._vector_service = vector_service
        self._llm_provider = llm_provider
        self._is_closed = False

    @property
    def embedding_service_info(self) -> dict:
        """Returns the embedding service instance."""
        return self._embedding_service.info

    def close(self, delete_collection: bool = False) -> None:
        """Close any resources held by the usecase instance."""
        if not self._is_closed:
            self._vector_service.close(delete_collection=delete_collection)
            self._is_closed = True

    async def _handle_file_upload(self, data: list[UploadFile]) -> None:
        """Handle file upload and processing based on the type of data."""
        try:
            file_path = await self._file_service.save_files(data)
            await self.process_document_paths(file_path)
        except Exception as e:
            logger.error(f"Error in handle_file_upload: {e}")
            raise e

    def _handle_chunking_and_embedding(
        self,
        content: str,
        chunking_strategy: TECHNIQUES = "default",
    ) -> tuple[list[str], list[list[float]]]:
        """Handle chunking and embedding of the content."""
        chunks = self._chunking_strategy.chunk(
            text=content, technique=chunking_strategy
        )
        embeddings = self._embedding_service.embed(chunks)
        return chunks, embeddings

    def _handle_vector_addition(self, data: list[EmbedData]) -> None:
        """Handle adding vectors to the vector service."""
        self._vector_service.save(data=data)

    async def process_documents(self, files: list[UploadFile]) -> None:
        """Process a list of files."""
        await self._handle_file_upload(files)

    @deprecated("WARNING: This method is EXPERIMENTAL and subject to breaking changes.")
    def _handling_document_chunking_embedding_and_metadata(self, doc: DoclingDocument):
        doc_id = str(uuid4())
        filename: Optional[str] = None
        if doc.origin and hasattr(doc.origin, "filename"):
            filename = doc.origin.filename
        elif hasattr(doc, "name"):
            filename = doc.name
        if not filename:
            logger.warning(f"Document {doc} does not have a filename. Skipping.")
            return

        payload_texts: list[str] = []
        embedded_dtos: list[EmbedData] = []

        # 2. Extract and build everything in a single, efficient loop
        for chunk_idx, (chunk_txt, meta) in enumerate(
            self._chunking_strategy.chunk_docling_document(doc)
        ):
            payload_texts.append(chunk_txt)
            embedded_dtos.append(
                EmbedData(
                    embedding=[],
                    text=chunk_txt,
                    metadata={
                        "document_id": doc_id,
                        "filename": filename,
                        "chunk_index": chunk_idx,
                        "source": "document",
                        **meta.model_dump(),
                    },
                )
            )

        # check the raw chunks and metadata
        for idx, dto in enumerate(embedded_dtos):
            logger.warning(
                f"Chunk {idx}: text length={len(dto.text)}, metadata={dto.metadata} \n chunk_text: {dto.text}",
            )

        # 3. Pass the pre-built lists directly to your service
        all_embed_datas: list[EmbedData] = self._embedding_service.embed_dto(
            payload=payload_texts,
            embedded_dto=embedded_dtos,
        )

        self._vector_service.save(data=all_embed_datas)

    async def process_document_paths(self, paths: list[str]) -> None:
        """Process a list of paths."""
        try:
            documents = self._document_loader.load_documents(paths)
            for document in documents:
                try:
                    self._handling_document_chunking_embedding_and_metadata(document)
                except Exception as e:
                    logger.error(f"Error processing document {document.name}: {e}")
                    continue  # Continue processing other documents even if one fails
        except Exception as e:
            logger.error(f"Critical error loading document path cluster: {e}")
            raise
        finally:
            await self._file_service.delete_files(paths)

    def process_texts(self, texts: list[str]) -> None:
        """Process a list of texts."""
        for text in texts:
            try:
                chunks, embeddings = self._handle_chunking_and_embedding(text)
                self._handle_vector_addition(
                    data=[
                        EmbedData(
                            embedding=embed_data,
                            text=txt,
                            metadata={
                                "source": "text",
                                "chunk_index": chunk_index,
                                "text_length": len(txt),
                            },
                        )
                        for chunk_index, (txt, embed_data) in enumerate(
                            zip(chunks, embeddings)
                        )
                    ]
                )
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                raise e

    def query(self, query_text: str, top_k: int = 5) -> str:
        """Query the vector service and return results."""
        try:
            _, embeddings = self._handle_chunking_and_embedding(query_text)
            vector_result = self._vector_service.retrieve(
                query_vector=embeddings[0], top_k=top_k
            )
            for item in vector_result:
                print(
                    f"score={item.score:.4f} | items={list(item.payload.items())} | preview={str(item.payload)[:100]}"  # type: ignore
                )
            original_chunks = [
                item.payload.get("text", "")  # type: ignore
                for item in vector_result
            ]
            results = ""

            for i, chunk in enumerate(original_chunks):
                results += f"""
            Context {i + 1}

            {chunk}

            """
            llm_response = self._llm_provider.generate(
                context=results, user_input=query_text
            )
            print("=" * 80)
            print(results)
            print("=" * 80)
            return llm_response
        except Exception as e:
            logger.error(f"Error querying vector service: {e}")
            raise e
