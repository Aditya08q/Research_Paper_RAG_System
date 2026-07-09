"""
Request schemas for the API layer.

Pydantic validates incoming data before it ever reaches business logic —
e.g. an empty question string is rejected here rather than deep inside the
RAG pipeline.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body for POST /chat — a natural-language question about uploaded papers."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about the uploaded research papers.",
    )
