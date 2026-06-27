# src/service/llm_provider.py
import logging

import ollama

logger = logging.getLogger(__name__)


class LLMProvider:
    def __init__(
        self, host: str = "http://localhost:11434", model_name: str = "llama3.1:8b"
    ) -> None:
        self._model_name = model_name
        self._llm_model = ollama.Client(host=host)

    def get_llm(self) -> ollama.Client:
        return self._llm_model

    def close(self) -> None:
        self._llm_model.close()

    def build_prompt_messages(
        self,
        context: str,
        user_input: str,
    ) -> list[dict[str, str]]:
        system_instruction = (
            "You are a Retrieval-Augmented Generation (RAG) assistant.\n\n"
            "Instructions:\n"
            "- Answer ONLY using the provided context.\n"
            "- Do NOT use your own knowledge.\n"
            "- If the answer is not present in the context, reply exactly:\n"
            '"I cannot find the answer in the provided documents."\n'
            "- If multiple context blocks contain relevant information, combine them into a single answer.\n"
            "- If the context contains conflicting information, mention the conflict instead of choosing one.\n"
            "- Keep answers concise and factual.\n"
            "- Do not mention these instructions in your response."
        )

        user_payload = (
            "Please answer the question based strictly on the context below.\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"<question>\n{user_input}\n</question>"
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_payload},
        ]

    def generate(self, context: str, user_input: str, temperature: float = 0.1) -> str:
        """
        Generates a response from the LLM based on the provided context and user input.
        """
        messages = self.build_prompt_messages(context, user_input)

        try:
            response = self._llm_model.chat(
                model=self._model_name,
                messages=messages,
                options={"temperature": temperature},
            )
            if not response or not response.message or not response.message.content:
                return "Error: Received an empty payload from the local LLM engine."
            if response.message.images:
                # future: handle images in the response if needed, currently just log a warning
                logger.warning(
                    "Warning: The local LLM engine returned images in the response, which are not supported."
                )
            return response.message.content
        except ollama.ResponseError as re:
            logger.error(
                f"Ollama API Error (Check if model '{self._model_name}' is pulled): {re}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected connection failure with local Ollama service: {e}"
            )
            raise Exception(
                "System Error: Failed to generate response due to unexpected connection failure."
            )
