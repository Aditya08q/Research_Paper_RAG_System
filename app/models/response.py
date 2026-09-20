from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_created: int
    message: str = "File uploaded and indexed successfully."


class SourceCitation(BaseModel):
    filename: str
    page_number: int
    chunk_text: str = Field(..., description="The retrieved chunk of text used as evidence.")
    similarity_score: float


class ChatResponse(BaseModel):

    answer: str
    sources: list[SourceCitation]
