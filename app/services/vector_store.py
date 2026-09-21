from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding import get_embedding_model

logger = get_logger(__name__)


class VectorStoreError(Exception):

@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    logger.info("Opening Chroma collection '%s'", settings.chroma_collection_name)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(settings.chroma_persist_dir),
    )


def add_chunks(chunks: list[Document]) -> None:
   
    if not chunks:
        return

    try:
        store = get_vector_store()
        store.add_documents(chunks)
        logger.info("Added %d chunks to vector store", len(chunks))
    except Exception as exc:
        logger.error("Failed to add chunks to vector store: %s", exc)
        raise VectorStoreError("Could not store document embeddings.") from exc


def similarity_search(query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
    """
    Retrieve the most semantically similar chunks to a query.

    Args:
        query: The user's natural-language question.
        top_k: Number of chunks to retrieve. Defaults to settings.retrieval_top_k.

    Returns:
        A list of (Document, similarity_score) tuples, most similar first.

    Raises:
        VectorStoreError: if the similarity search fails.
    """
    k = top_k or settings.retrieval_top_k

    try:
        store = get_vector_store()
        # `with_relevance_scores` returns cosine similarity in [0, 1] rather
        # than raw distance, which is more intuitive to show to a user.
        results = store.similarity_search_with_relevance_scores(query, k=k)
        logger.info("Retrieved %d chunks for query (top_k=%d)", len(results), k)
        return results
    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        raise VectorStoreError("Could not search the document index.") from exc
