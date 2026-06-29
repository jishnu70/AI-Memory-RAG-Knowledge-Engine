# src/usecase.py

import asyncio
import logging
from typing import Optional
from uuid import uuid4

from fastapi import UploadFile

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
            await asyncio.to_thread(self.process_document_paths, file_path)
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

    def _handle_vector_addition(
        self, embeddings: list[list[float]], original_chunk: list[str], **metadata
    ) -> None:
        """Handle adding vectors to the vector service."""
        self._vector_service.save(
            embedded_list=embeddings, original_chunk=original_chunk, **metadata
        )

    async def process_documents(self, files: list[UploadFile]) -> None:
        """Process a list of files."""
        await self._handle_file_upload(files)

    def process_document_paths(self, paths: list[str]) -> None:
        """Process a list of paths."""
        try:
            documents = self._document_loader.load_documents(paths)
            for document in documents:
                try:
                    markdown_content = document.export_to_markdown()
                    doc_chunks, embeddings = self._handle_chunking_and_embedding(
                        markdown_content, "recursive"
                    )

                    if not doc_chunks:
                        logger.warning(
                            f"No chunks generated for document at {document.name}. Skipping."
                        )
                        continue

                    page_count = len(document.pages) if document.pages else 0
                    filename: Optional[str] = None

                    if document.origin and hasattr(document.origin, "filename"):
                        filename = document.origin.filename
                    elif hasattr(document, "name"):
                        filename = document.name

                    if not filename:
                        logger.warning(
                            f"Filename not found for document at {document.name}. Using 'unknown'."
                        )
                        continue

                    self._handle_vector_addition(
                        embeddings=embeddings,
                        original_chunk=doc_chunks,
                        document_id=str(uuid4()),
                        source="document",
                        document_parser="docling-parser",
                        filename=filename,
                        page_count=page_count,
                    )
                except Exception as e:
                    logger.error(f"Error processing document {document.name}: {e}")
                    continue  # Continue processing other documents even if one fails
        except Exception as e:
            logger.error(f"Critical error loading document path cluster: {e}")
            raise

    def process_texts(self, texts: list[str]) -> None:
        """Process a list of texts."""
        for text in texts:
            try:
                chunks, embeddings = self._handle_chunking_and_embedding(text)
                self._handle_vector_addition(
                    embeddings=embeddings,
                    original_chunk=chunks,
                    source="text",
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
            original_chunks = [
                item.payload.get("original_chunk", "")  # type: ignore
                for item in vector_result
            ]
            results = "\n\n".join(original_chunks)
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
