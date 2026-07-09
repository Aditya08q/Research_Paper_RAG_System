"""
Application entrypoint.

Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api import chat, upload
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Research Paper RAG",
    description=(
        "A beginner-friendly Retrieval-Augmented Generation API for asking "
        "questions about uploaded research papers, powered by LangChain, "
        "ChromaDB, and Grok."
    ),
    version="0.1.0",
)

app.include_router(upload.router, tags=["Upload"])
app.include_router(chat.router, tags=["Chat"])


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Simple liveness check."""
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Research Paper RAG API starting up")
