# src/schemas/req_in.py
from pydantic import BaseModel, Field, field_validator

from src.exceptions.body_exceptions import InvalidTextError


class BaseRequest(BaseModel):
    """Base request model with common validation logic for all request types."""

    @staticmethod
    def validate_content(v: str) -> str:
        if not v or not v.strip() or not isinstance(v, str):
            raise InvalidTextError("Input text cannot be empty or whitespace")
        return v


class TextRequest(BaseRequest):
    """Request model for processing text."""

    text: str = Field(..., description="The input text to process")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        return cls.validate_content(v)


class QueryRequest(BaseRequest):
    """Request model for querying text."""

    query: str = Field(..., description="The query to process")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        return cls.validate_content(v)
