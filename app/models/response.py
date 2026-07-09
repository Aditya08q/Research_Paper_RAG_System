"""
Response schemas for the API layer.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Returned after a PDF is successfully uploaded, parsed, and indexed."""

    filename: str
    pages_extracted: int
    chunks_created: int
    message: str = "File uploaded and indexed successfully."


class SourceCitation(BaseModel):
    """
    A single piece of retrieved evidence backing the answer — this is what
    lets a user verify the LLM's answer against the original paper instead
    of trusting it blindly.
    """

    filename: str
    page_number: int
    chunk_text: str = Field(..., description="The retrieved chunk of text used as evidence.")
    similarity_score: float


class ChatResponse(BaseModel):
    """Returned from POST /chat — the answer plus the sources that support it."""

    answer: str
    sources: list[SourceCitation]
