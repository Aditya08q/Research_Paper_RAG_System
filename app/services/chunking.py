"""
Chunking service.

Design choice: RecursiveCharacterTextSplitter tries to split on paragraph
breaks first, then sentences, then words — only falling back to a hard
character cut as a last resort. This keeps chunks semantically coherent
instead of slicing mid-sentence, which matters a lot for embedding quality:
a chunk that ends mid-thought produces a noisier, less useful vector.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def chunk_documents(pages: list[Document]) -> list[Document]:
    """
    Split page-level Documents into smaller overlapping chunks.

    Each output chunk keeps the same metadata (source filename, page number)
    as the page it came from, so a chunk retrieved later can still be traced
    back to an exact page for citation.

    Args:
        pages: List of page-level Documents, typically from `pdf_loader.load_pdf`.

    Returns:
        A list of smaller Document chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Try paragraph, then line, then sentence, then word, then character.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    logger.info(
        "Created %d chunks (size=%d, overlap=%d) from %d pages",
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
        len(pages),
    )
    return chunks
