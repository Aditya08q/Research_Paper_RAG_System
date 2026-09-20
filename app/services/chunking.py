from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def chunk_documents(pages: list[Document]) -> list[Document]:
    
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
