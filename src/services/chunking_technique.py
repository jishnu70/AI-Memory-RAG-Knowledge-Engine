# src/services/chunking_technique.py


import io
from typing import Literal

TECHNIQUES = Literal["simple", "character", "recursive", "markdown"]


class ChunkingTechnique:
    """
    A class to handle different chunking techniques for text processing.
    """

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
        """
        Chunk the given text into smaller pieces of specified size.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The size of each chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    @staticmethod
    def langchain_character_text_splitter(
        text: str, chunk_size: int = 512, chunk_overlap: int = 20
    ) -> list[str]:
        """
        Use LangChain's RecursiveCharacterTextSplitter to split the text into chunks.

        Args:
            text (str): The text to be split.
            chunk_size (int): The size of each chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        from langchain_text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_text(text)

    @staticmethod
    def langchain_recursive_character_text_splitter(
        text: str, chunk_size: int = 512, chunk_overlap: int = 20
    ) -> list[str]:
        """
        Use LangChain's RecursiveCharacterTextSplitter to split the text into chunks.

        Args:
            text (str): The text to be split.
            chunk_size (int): The size of each chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separators_regex=False,
        )
        return splitter.split_text(text)

    @staticmethod
    def chunk_document(
        file_bytes: bytes,
        payload: dict[str, str],
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        technique: TECHNIQUES = "recursive",
    ) -> list[str]:
        """
        Chunk the content of the given document into smaller pieces of specified size.

        Args:
            file_bytes (bytes): The content of the document in bytes.
            payload (dict[str, str]): The metadata of the document, including filename.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
            technique (TECHNIQUES): The chunking technique to be used.

        Returns:
            list[str]: A list of text chunks from the document.
        """
        if payload.get("filename") is None:
            return []

        txt = ""

        match payload.get("filename", "").split(".")[-1].lower():
            case "pdf":
                import pymupdf

                content = io.BytesIO(file_bytes)
                pdf_reader = pymupdf.open(stream=content, filetype="pdf")
                for page in pdf_reader:
                    txt += page.get_text() + "\n"  # type: ignore[union-attr]
            case "txt" | "md" | _:
                txt = file_bytes.decode("utf-8")

        match technique:
            case "simple":
                return ChunkingTechnique.chunk_text(txt, chunk_size)
            case "character":
                return ChunkingTechnique.langchain_character_text_splitter(
                    txt, chunk_size, chunk_overlap=chunk_overlap
                )
            case "markdown":
                from langchain_text_splitters import MarkdownTextSplitter

                splitter = MarkdownTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                return splitter.split_text(txt)
            case "recursive" | _:
                return ChunkingTechnique.langchain_recursive_character_text_splitter(
                    txt, chunk_size, chunk_overlap=chunk_overlap
                )

    def chunk(
        self,
        *,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        technique: TECHNIQUES = "recursive",
        **kwargs,
    ) -> list[str]:
        """
        Chunk the given text into smaller pieces of specified size using the specified technique.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
            technique (TECHNIQUES): The chunking technique to be used.

        Returns:
            list[str]: A list of text chunks.
        """
        match technique:
            case "simple":
                return self.chunk_text(text, chunk_size)
            case "character":
                return self.langchain_character_text_splitter(
                    text, chunk_size, chunk_overlap=chunk_overlap
                )
            case "markdown":
                return self.chunk_document(
                    file_bytes=text.encode("utf-8"),
                    payload={"filename": kwargs.get("filename", "")},
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    technique="markdown",
                )
            case "recursive" | _:
                return self.langchain_recursive_character_text_splitter(
                    text, chunk_size, chunk_overlap=chunk_overlap
                )


def get_chunking_technique() -> ChunkingTechnique:
    """
    Get an instance of the ChunkingTechnique class.

    Returns:
        ChunkingTechnique: An instance of the ChunkingTechnique class.
    """
    return ChunkingTechnique()
