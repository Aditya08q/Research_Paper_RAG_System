from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body for POST /chat — a natural-language question about uploaded papers."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about the uploaded research papers.",
    )
