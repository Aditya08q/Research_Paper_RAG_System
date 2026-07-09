"""
Embedding service.

Design choice: BAAI/bge-small-en-v1.5 is a small, CPU-friendly embedding
model that still performs well on retrieval benchmarks. Using LangChain's
HuggingFaceEmbeddings wrapper means the same object can be passed directly
into Chroma later — Chroma calls `.embed_documents()` and `.embed_query()`
on it automatically, so we never manually juggle raw vectors.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the embedding model.

    The model is loaded once per process (via lru_cache) because loading it
    from disk/HuggingFace hub on every request would be slow and wasteful —
    the weights don't change between calls.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model_name)
    model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        # bge models are trained to work well with normalized embeddings,
        # which makes cosine similarity comparisons more reliable.
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully")
    return model
