# src/services/chunking_technique.py


from typing import Literal

TECHNIQUES = Literal["simple", "character", "recursive", "sentenceSplitter"]


class ChunkingTechnique:
    """
    A class to handle different chunking techniques for text processing.
    """

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> list[str]:
        """
        Chunk the given text into smaller pieces of specified size.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The size of each chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _langchain_character_text_splitter(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 20
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

    def _langchain_recursive_character_text_splitter(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 20
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

    def _llamaindex_sentence_splitter(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 50
    ):
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_text(text)

    def chunk(
        self,
        *,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        technique: TECHNIQUES = "recursive",
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
                return self._chunk_text(text, chunk_size)
            case "character":
                return self._langchain_character_text_splitter(
                    text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            case "sentenceSplitter":
                return self._llamaindex_sentence_splitter(
                    text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            case "recursive" | _:
                return self._langchain_recursive_character_text_splitter(
                    text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )


def get_chunking_technique() -> ChunkingTechnique:
    """
    Get an instance of the ChunkingTechnique class.

    Returns:
        ChunkingTechnique: An instance of the ChunkingTechnique class.
    """
    return ChunkingTechnique()
