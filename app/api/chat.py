"""
Chat endpoint — accepts a question and runs the query flow via the RAG
pipeline, returning an answer grounded in the uploaded papers.
"""

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.rag_pipeline import RAGPipelineError, ask

logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using only the content of uploaded research papers."""
    try:
        return ask(request.question)
    except RAGPipelineError as exc:
        logger.error("RAG pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
