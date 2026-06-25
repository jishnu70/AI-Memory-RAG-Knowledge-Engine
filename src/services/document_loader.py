# src/services/document_loader.py
from typing import Union

from llama_index.core import Document, SimpleDirectoryReader

from src.exceptions.service_exceptions import DocumentInvalidParametersException


class DocumentLoader:
    def __init__(self):
        pass

    def _read_inputs(self, path: Union[list[str], str]) -> list[Document]:
        """
        Read the content of a file and return it as a string.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The content of the file.
        """
        if isinstance(path, list):
            for p in path:
                if not p.endswith(f".{p.split('.')[-1]}"):
                    raise DocumentInvalidParametersException(
                        "All paths must be strings."
                    )
            content = SimpleDirectoryReader(input_files=path).load_data()
            return content
        content = SimpleDirectoryReader(input_dir=path).load_data()
        return content

    def load_files(self, files_path: Union[list[str], str]) -> list[Document]:
        """
        Load the content of a file and return it as a string.

        Args:
            file_path (str): The path to the file.

        Returns:
            list[Document]: The content of the file.
        """
        if isinstance(files_path, str):
            files_path = [files_path]
        return self._read_inputs(path=files_path)

    def load_directory(self, dir_path: str) -> list[Document]:
        """
        Load the content of a file and return it as a string.

        Args:
            file_path (str): The path to the file.

        Returns:
            list[Document]: The content of the file.
        """
        return self._read_inputs(path=dir_path)
