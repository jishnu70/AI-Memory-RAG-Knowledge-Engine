# src/services/document_loader.py
import logging
import pathlib
from collections.abc import Iterator
from typing import Generator, Union

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self):
        self._converter = DocumentConverter()

    def _path_exists(self, path: str) -> bool:
        """
        Check if a given path exists.

        Args:
            path (str): The path to check.
        Returns:
            bool: True if the path exists, False otherwise.
        """
        return pathlib.Path(path).exists()

    def _is_file(self, path: str) -> bool:
        """
        Check if a given path is a file.

        Args:
            path (str): The path to check.
        Returns:
            bool: True if the path is a file, False otherwise.
        """
        return pathlib.Path(path).is_file()

    def _convert_all(self, path: list[str]) -> Iterator[ConversionResult]:
        """
        Load documents from a given path.

        Args:
            path (str): The path to load documents from.
        Yields:
            DoclingDocument: A document loaded from the path.
        """
        return self._converter.convert_all(path)

    def load_documents(
        self, path: Union[str, list[str]]
    ) -> Generator[DoclingDocument, None, None]:
        """
        Load documents from a given path.

        Args:
            path (Union[str, list[str]]): The path to load documents from.
        Yields:
            DoclingDocument: A document loaded from the path.
        """
        input_paths = [path] if isinstance(path, str) else path
        final_file_list: list[str] = []

        for p in input_paths:
            if not self._path_exists(p):
                raise FileNotFoundError(f"Path does not exist: {p}")

            if self._is_file(p):
                final_file_list.append(p)
            else:
                # Recursively find all files in the directory
                discovered_files = [
                    str(file_path)
                    for file_path in pathlib.Path(p).rglob("*")
                    if file_path.is_file()
                ]
                final_file_list.extend(discovered_files)

        for doc in self._convert_all(final_file_list):
            if (
                doc.status == ConversionStatus.SUCCESS
            ):  # stream only successful conversions
                yield doc.document
            else:  # log failed conversions for debugging purposes
                logger.warning(
                    "Failed to convert document %s. Errors: %s",
                    doc.input.file,
                    doc.errors,
                )
